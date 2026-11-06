"""Dedicated GPU capture bridge, co-located or across NVLink from detectors.

The engine JSON is an explicit deployment artifact. This command never claims
an occupied GPU or changes a running engine. Start only on an owned GPU after
checking memory. Acknowledged feature-byte admission is separate from MRR64.
"""
from collections import deque
import argparse
import asyncio
import json
import os
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio


def device_payload(result):
    value=result.get('meta_info',{}).get('features')
    # TokenizerManager accumulates customized_info as a list of records.
    if isinstance(value,list):
        if len(value)!=1:raise ValueError('Expected exactly one device feature record')
        value=value[0]
    if not isinstance(value,dict) or value.get('schema')!='inscope.device_features.v1':
        raise ValueError('Native device feature record is absent or invalid')
    return value


async def send_host_frames(payload,rid,receiver,send):
    """Keep large D2H copies off the control/heartbeat loop, with bounded frames."""
    from .host_transfer import host_frames
    frames=host_frames(payload,rid,receiver)
    try:
        while True:
            future=asyncio.create_task(asyncio.to_thread(next,frames,None))
            try:frame=await asyncio.shield(future)
            except asyncio.CancelledError:
                # A running CUDA read must finish before closing/acking leases.
                await future
                raise
            if frame is None:break
            await send(frame)
    finally:frames.close()


async def serve(engine_args,bind,layers,pool_bytes,hidden_dim):
    os.environ['SGLANG_ENABLE_FEATURE_CAPTURE']='1'
    # Native sampled warmup covers capture buckets without sweeping every M.
    os.environ.setdefault('SGLANG_JIT_DEEPGEMM_FAST_WARMUP','1')
    os.environ['SGLANG_FEATURE_CAPTURE_RAW_BYTES']='1'
    os.environ['SGLANG_FEATURE_EARLY_EXIT_LAYER']=str(max(layers))
    os.environ['SGLANG_FEATURE_CAPTURE_MAX_TARGETS']=str(len(layers))
    os.environ['SGLANG_FEATURE_CAPTURE_MAX_BYTES']=str(max(pool_bytes*2,16*1024**3))
    os.environ['SGLANG_FEATURE_OUTPUT_FACTORY']='inscope_server.detection.sglang_ipc_output:copy_to_ipc'
    os.environ['INSCOPE_GPU_FEATURE_POOL_BYTES']=str(pool_bytes)
    os.environ['INSCOPE_GPU_POOL_FALLBACK']='error'
    if not bind.startswith('ipc://'):raise ValueError('CUDA feature capabilities require a local IPC endpoint')
    from sglang.srt.feature_capture.recorder import OUTPUT_FACTORY_API_VERSION
    if OUTPUT_FACTORY_API_VERSION!=1:raise RuntimeError('Unsupported native device-output factory API')
    from sglang.srt.entrypoints.engine import Engine
    engine=Engine(**engine_args)
    ctx=zmq.asyncio.Context();socket=ctx.socket(zmq.ROUTER);socket.setsockopt(zmq.LINGER,0)
    previous_umask=os.umask(0o077)
    try:socket.bind(bind)
    finally:os.umask(previous_umask)
    host_receiver=None
    tasks={};generations=set();reserved={};condition=asyncio.Condition();used=0
    # Per-request actual BF16 feature size, not a fixed global DP quota.
    usable=pool_bytes-1024*1024
    admission=usable//2
    waiters=deque()
    async def release(key):
        nonlocal used
        async with condition:
            used-=reserved.pop(key,0);condition.notify_all()
    async def one(identity,request):
        nonlocal used,host_receiver
        key=(identity,request['rid']);rid=request['rid'];sent=False;generation=None;features=None
        host_mode=request.get('transport')=='host';host_started=False
        try:
            selected=tuple(request['selected_positions'])
            if tuple(request['layer_ids'])!=tuple(layers):raise ValueError('Capture layer identity mismatch')
            size=len(selected)*len(layers)*hidden_dim*2
            if size>usable:raise ValueError('DP exceeds configured GPU feature pool; use host capture endpoint')
            async with condition:
                waiters.append(key)
                try:
                    await condition.wait_for(lambda:waiters[0]==key and
                        ((used==0) if size>admission else used+size<=admission))
                    used+=size;reserved[key]=size
                finally:
                    waiters.remove(key);condition.notify_all()
            generation=asyncio.create_task(engine.async_generate(input_ids=request['input_ids'],rid=rid,
                sampling_params=dict(max_new_tokens=0,temperature=0),feature_capture=dict(
                    layers=list(layers),site='block_output',positions=dict(mode='explicit',index=list(selected)),encoding='raw_bytes')))
            generations.add(generation);generation.add_done_callback(generations.discard)
            result=await asyncio.shield(generation)
            features=device_payload(result)
            if not features or features.get('schema')!='inscope.device_features.v1' or 'error' in features:
                raise RuntimeError((features or {}).get('error','GPU capture plugin unavailable'))
            if host_mode:
                from .gpu_transport import CudaFeatureReceiver
                if host_receiver is None:host_receiver=CudaFeatureReceiver('cuda:0')
                host_started=True
                async def send_frame(frame):
                    await socket.send_multipart([identity,msgspec.msgpack.encode(frame)])
                await send_host_frames(features,rid,host_receiver,send_frame)
            else:
                # Once dispatch is attempted, only the receiver may ack this
                # lease: cancellation must not race a delivered capability.
                sent=True
                await socket.send_multipart([identity,msgspec.msgpack.encode(dict(rids=[rid],device_features=features))])
        except asyncio.CancelledError:
            try:engine.tokenizer_manager.abort_request(rid)
            except Exception:pass
            if not sent and not host_started and generation is not None:
                from .host_transfer import discard_device_payload
                def drain(future):
                    try:discard_device_payload(device_payload(future.result()))
                    except Exception:pass
                if generation.done():drain(generation)
                else:generation.add_done_callback(drain)
            raise
        except Exception as error:
            await socket.send_multipart([identity,b'fyh-error-v2:'+msgspec.json.encode(dict(rid=rid,error=str(error)))])
        finally:
            tasks.pop(key,None)
            if not sent:await release(key)
    try:
        while True:
            identity,payload=await socket.recv_multipart()
            if payload.startswith(b'fyh-info:'):
                rid=payload[len(b'fyh-info:'):].decode()
                info=dict(rid=rid,model_path=engine_args['model_path'],tokenizer_path=engine_args.get('tokenizer_path',engine_args['model_path']),
                    lora_path=None,version='inscope-device-features-v1',context_length=engine_args.get('context_length'),
                    feature_capture_graph_layers=list(layers),feature_capture_early_exit_layer=max(layers),
                    device_transport='cuda_ipc',gpu_feature_pool_bytes=pool_bytes,reserved_bytes=used,feature_admission_bytes=admission)
                await socket.send_multipart([identity,b'fyh-info:'+msgspec.json.encode(info)]);continue
            if payload.startswith((b'fyh-release:',b'fyh-abort:')):
                prefix=b'fyh-release:' if payload.startswith(b'fyh-release:') else b'fyh-abort:'
                key=(identity,payload[len(prefix):].decode())
                if prefix==b'fyh-abort:' and key in tasks:tasks[key].cancel()
                await release(key);continue
            request=msgspec.msgpack.decode(payload);key=(identity,request['rid'])
            if key in tasks or key in reserved:
                await socket.send_multipart([identity,b'fyh-error-v2:'+msgspec.json.encode(dict(rid=request['rid'],error='Duplicate request identity'))]);continue
            tasks[key]=asyncio.create_task(one(identity,request))
    finally:
        for task in tasks.values():task.cancel()
        await asyncio.gather(*list(tasks.values()),return_exceptions=True)
        if host_receiver:host_receiver.close()
        socket.close(0);ctx.destroy(linger=0);engine.shutdown()


def main():
    p=argparse.ArgumentParser();p.add_argument('--engine-config',type=Path,required=True)
    p.add_argument('--bind',required=True);p.add_argument('--layers',type=int,nargs='+',required=True)
    p.add_argument('--pool-gib',type=float,default=4);args=p.parse_args()
    config=json.loads(args.engine_config.read_text())
    # Hidden dimension is bridge metadata, not an Engine keyword.
    hidden=config.pop('inscope_hidden_dim')
    async def run():
        # Pass bridge-only fields separately to avoid changing native model args.
        await serve(config,args.bind,args.layers,int(args.pool_gib*1024**3),hidden)
    asyncio.run(run())


if __name__=='__main__':main()
