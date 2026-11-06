"""SGLang device-output -> IPC -> detector parity, without loading a new LLM."""
import argparse
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace
import msgspec
import torch


def source(pipe,layers,positions,dim):
    os.environ['SGLANG_FEATURE_OUTPUT_FACTORY']='inscope_server.detection.sglang_ipc_output:copy_to_ipc'
    os.environ['INSCOPE_GPU_FEATURE_POOL_BYTES']=str(256*1024**2)
    from sglang.srt.feature_capture.recorder import FeatureDeviceOutput
    from sglang.srt.feature_capture.plan import SlotPlan
    from sglang.srt.feature_capture.spec import FeatureRequestSpec,FeatureTarget
    from inscope_server.detection.sglang_ipc_output import _pools
    torch.cuda.set_device(0)
    n=len(positions)
    x=(torch.arange(n*len(layers)*dim,device='cuda:0',dtype=torch.float32).remainder_(127)/32).bfloat16().reshape(n,len(layers),dim)
    flat=torch.cat([x[:,j] for j in range(len(layers))])
    plan=tuple(SlotPlan(layer=l,rows=tuple(range(n)),segments=(('test',tuple(positions)),),out_start=j*n) for j,l in enumerate(layers))
    output=FeatureDeviceOutput(tensor=flat,plan=plan)
    def forbidden_copy(t):raise AssertionError('Feature D2H copy must not run')
    host=output.copy_to_host(forbidden_copy)
    req=SimpleNamespace(rid='test',is_retracted=False,customized_info=None,
        feature_capture=FeatureRequestSpec(targets=tuple(FeatureTarget(l) for l in layers),positions=tuple(positions),position_mode='explicit',encoding='raw_bytes'))
    host.consume(SimpleNamespace(reqs=[req]),None)
    pipe.send_bytes(msgspec.msgpack.encode(req.customized_info['features'][0]))
    if pipe.recv()!='ack':raise RuntimeError('Missing read completion')
    deadline=time.monotonic()+5
    while _pools[0].active_lease_count and time.monotonic()<deadline:time.sleep(.01)
    pipe.send(dict(leases=_pools[0].active_lease_count))
    if pipe.recv()!='closed':raise RuntimeError('Consumer mapping still open')
    _pools[0].shutdown()


def consumer(pipe,control,plan_path,target):
    try:
        from inscope_server.token_probe import AllPositionProbeBank
        from inscope_server.detection.native_backend import NativeCaptureBackend
        from inscope_server.detection.composer import CompiledComposer
        from inscope_server.detection.planner import AllPositionPlanner
        bank=AllPositionProbeBank(plan_path,f'cuda:{target}');bank.warmup()
        prompt=CompiledComposer('test',2048).compose(dict(composer_id='test',input_ids=[1]*1030,
            pre_position=1,observation_tail_position=1024,post_position=1026,suffix_position=1028))
        plan=AllPositionPlanner().plan(prompt,tuple(bank.layers));n=len(plan.positions)
        original=(torch.arange(n*len(bank.layers)*bank.dim,dtype=torch.float32).remainder_(127)/32).bfloat16().reshape(n,-1)
        index={v:i for i,v in enumerate(plan.positions)}
        cpu_features=(original.narrow(0,index[1],1024),original[[index[1026],index[1028]]])
        row=dict(region_positions=list(plan.region));expected=bank.compute([(0,row,cpu_features)])
        backend=NativeCaptureBackend(['ipc:///unused'],hidden_dim=bank.dim,model_path='test',layers=bank.layers,gpu_device=f'cuda:{target}')
        if not pipe.poll(90):raise TimeoutError('SGLang device-output producer timed out')
        payload=msgspec.msgpack.decode(pipe.recv_bytes());features=backend._receive_device(payload,plan)
        actual=bank.compute([(0,row,features)])
        full=bank.predict([(0,row,features)]);baseline=bank.predict([(0,row,cpu_features)])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(full['logits'],baseline['logits'],rtol=1e-5,atol=1e-5)
        placement=features.placement;features.release();pipe.send('ack')
        if not pipe.poll(10):raise TimeoutError('Lease recycling timed out')
        state=pipe.recv();del features;gc.collect();backend.gpu_receiver.close()
        result=dict(positions=len(expected),detectors=len(bank.models),placement=placement,
            exact_token_logits=True,max_token_error=float((actual-expected).abs().max()),
            max_dp_error=float((full['logits']-baseline['logits']).abs().max()),remaining_leases=state['leases'],
            feature_d2h_calls=0)
        control.send(result)
    except BaseException:
        import traceback
        control.send(dict(error=traceback.format_exc()))
        raise
    finally:pipe.close();control.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--target',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.plan.read_text());layers=plan['layers']
    checkpoint=torch.load(plan['checkpoints'][0]['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    dim=checkpoint['normalization']['mean'].numel()
    positions=tuple(sorted({0,*range(1,1025),1026,1028}))
    ctx=mp.get_context('spawn');producer_pipe,consumer_pipe=ctx.Pipe();report,send_report=ctx.Pipe(False)
    writer=ctx.Process(target=source,args=(producer_pipe,layers,positions,dim))
    reader=ctx.Process(target=consumer,args=(consumer_pipe,send_report,args.plan,args.target))
    writer.start();reader.start();producer_pipe.close();send_report.close()
    try:
        if not report.poll(120):raise TimeoutError('Device detector acceptance timed out')
        result=report.recv();reader.join(20)
        if reader.exitcode!=0 or 'error' in result:raise RuntimeError(str(result))
        # Producer must outlive the entire consumer process, including its CUDA
        # storage cache/context; an ack alone does not close imported mappings.
        consumer_pipe.send('closed');writer.join(20)
        result.update(producer_exitcode=writer.exitcode,consumer_exitcode=reader.exitcode)
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        if result['remaining_leases'] or writer.exitcode!=0:raise RuntimeError('Device feature lifecycle failed')
    finally:
        for process in (reader,writer):
            if process.is_alive():process.terminate();process.join(10)
        consumer_pipe.close();report.close()


if __name__=='__main__':main()
