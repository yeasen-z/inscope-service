"""Finite cross-process placement acceptance, with bounded synthetic payloads."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import torch


def producer(pipe,source,nbytes):
    from sglang.srt.multimodal.transport.cuda_ipc import MmItemMemoryPool
    from inscope_server.detection.gpu_transport import export_proxy
    torch.cuda.set_device(source)
    pool=MmItemMemoryPool(memory_size=nbytes+4*1024**2,recycle_interval=.01,base_gpu_id=source,
        consumer_count=1,max_inflight_slices=32)
    x=torch.arange(nbytes//2,device=f'cuda:{source}',dtype=torch.float32).remainder_(127).bfloat16().reshape(-1,1,1024)
    try:
        for mode in ('ipc','host')*4:
            if mode=='ipc':
                start=time.perf_counter();proxy=pool.wrap_tensor(x,use_pool_handle_cache=True)
                if proxy is None:raise RuntimeError('Finite benchmark pool exhausted')
                descriptor=export_proxy(proxy,torch.cuda.get_device_properties(source).uuid)
                pipe.send((mode,descriptor,time.perf_counter()-start))
            else:
                start=time.perf_counter();host=x.cpu();pipe.send((mode,host.view(torch.uint8).numpy().tobytes(),time.perf_counter()-start))
            if pipe.recv()!='ack':raise RuntimeError('Missing consumer acknowledgement')
        deadline=time.monotonic()+3
        while pool.active_lease_count and time.monotonic()<deadline:time.sleep(.01)
        pipe.send(dict(remaining_leases=pool.active_lease_count))
        if pipe.recv()!='closed':raise RuntimeError('Consumer must close imported mappings before producer exit')
    finally:pool.shutdown();pipe.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=int,default=0);p.add_argument('--target',type=int,default=1)
    p.add_argument('--consumer-threads',type=int,default=0);p.add_argument('--mib',type=int,default=64);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    nbytes=args.mib*1024**2;torch.cuda.set_device(args.target)
    from inscope_server.detection.gpu_transport import CudaFeatureReceiver
    receiver=CudaFeatureReceiver(f'cuda:{args.target}')
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    process=ctx.Process(target=producer,args=(child,args.source,nbytes));process.start();child.close()
    workers=[ThreadPoolExecutor(max_workers=1) for _ in range(args.consumer_threads)]
    result=dict(bytes=nbytes,source=args.source,target=args.target,consumer_threads=args.consumer_threads,timings={},full_payload_verified=True)
    try:
        samples={"ipc":[],"host":[]}
        for _ in range(8):
            if not parent.poll(60):raise TimeoutError('GPU transfer producer timed out')
            mode,payload,producer_seconds=parent.recv();start=time.perf_counter()
            if mode=='ipc':
                if workers:
                    value,proxy,placement=workers[(_//2)%len(workers)].submit(receiver.receive,payload).result(timeout=30)
                else:value,proxy,placement=receiver.receive(payload)
                torch.cuda.synchronize(args.target)
                seconds=time.perf_counter()-start
                expected=torch.arange(nbytes//2,device=f'cuda:{args.target}',dtype=torch.float32).remainder_(127).bfloat16().reshape(value.shape)
                if not torch.equal(value,expected):
                    raise ValueError('IPC payload changed')
                if proxy:
                    with torch.cuda.device(args.target):proxy.release_without_reconstruction()
            else:
                host=torch.frombuffer(bytearray(payload),dtype=torch.bfloat16)
                value=host.to(f'cuda:{args.target}');torch.cuda.synchronize(args.target)
                seconds=time.perf_counter()-start
                placement='host'
            samples[mode].append(dict(producer_seconds=producer_seconds,consumer_seconds=seconds,placement=placement))
            parent.send('ack')
        if not parent.poll(10):raise TimeoutError('Lease-recycle report timed out')
        result.update(parent.recv())
        del value
        proxy=None
        import gc;gc.collect();receiver.close();gc.collect()
        parent.send('closed');process.join(10)
        result['timings']=samples
        import statistics
        result['warm_median_seconds']={k:statistics.median(x['producer_seconds']+x['consumer_seconds'] for x in v[1:]) for k,v in samples.items()}
        if process.exitcode!=0 or result['remaining_leases']!=0:raise RuntimeError('Producer/lease lifecycle failed')
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:
        for worker in workers:worker.shutdown(wait=True)
        if process.is_alive():process.terminate();process.join(10)
        parent.close()


if __name__=='__main__':main()
