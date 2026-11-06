"""Multiplexed BF16 binary capture over the existing sglang-inscope bridge.

No import from the training repo. Wire schema is fyh.feature_batch.v1; control
messages retain their deployed prefixes for compatibility with live services.
"""
import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio


class NativeCaptureBackend:
    def __init__(self,endpoints,*,hidden_dim,model_path,layers,early_exit=16,timeout=900,
                 token_low=3145728,token_high=5242880,max_requests=4096,max_frame_bytes=4194304,gpu_device=None,wire_transport="auto"):
        if not endpoints or not 0<=token_low<token_high or min(hidden_dim,max_requests,max_frame_bytes)<1:
            raise ValueError('Invalid native backend limits')
        if wire_transport not in ("auto","host"):raise ValueError("Unsupported wire transport")
        self.wire_transport=wire_transport
        self.endpoints=list(endpoints); self.dim=hidden_dim; self.model_path=model_path
        self.layers=tuple(layers); self.early_exit=early_exit; self.timeout=timeout
        self.low=token_low; self.high=token_high; self.max_requests=max_requests; self.max_frame_bytes=max_frame_bytes
        self.tokens=[0]*len(endpoints); self.requests=[0]*len(endpoints); self.blocked=[False]*len(endpoints)
        self.condition=asyncio.Condition(); self.context=None; self.sockets=[]; self.readers=[]; self.channels=[]
        self.gpu_pool=None
        self.gpu_receiver=None
        if gpu_device is not None:
            from .gpu_transport import CudaFeatureReceiver
            self.gpu_receiver=CudaFeatureReceiver(gpu_device)
            self.gpu_pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='inscope-device-copy')
        self.health_task=None
        self.release_tasks=set()
        self.errors={}; self.closed=False; self.verified=False

    async def start(self):
        if self.closed: raise RuntimeError('Native capture backend closed')
        if self.context is not None: return
        self.context=zmq.asyncio.Context()
        for i,endpoint in enumerate(self.endpoints):
            socket=self.context.socket(zmq.DEALER); socket.setsockopt(zmq.LINGER,0)
            socket.setsockopt(zmq.MAXMSGSIZE,self.max_frame_bytes+1024*1024)
            socket.connect(endpoint); self.sockets.append(socket); self.channels.append({})
            self.readers.append(asyncio.create_task(self._receive(i)))

    async def _receive(self,i):
        try:
            while True:
                payload=await self.sockets[i].recv()
                if payload.startswith(b'fyh-error-v2:'):
                    error=json.loads(payload[len(b'fyh-error-v2:'):]); rid=error['rid']
                    value=RuntimeError(error.get('error','Native capture failed'))
                elif payload.startswith(b'fyh-info:'):
                    value=json.loads(payload[len(b'fyh-info:'):]); rid=value['rid']
                elif payload.startswith(b'fyh-error:'):
                    raise RuntimeError('Unroutable native capture failure')
                else:
                    value=msgspec.msgpack.decode(payload)
                    rids=value.get('rids',[])
                    if len(rids)!=1: raise ValueError('Expected exactly one request per native frame')
                    rid=rids[0]
                channel=self.channels[i].get(rid)
                if channel is not None: channel.put_nowait(value)
                elif isinstance(value,dict) and 'device_features' in value:
                    self._discard_device(value['device_features'])
                    await self.sockets[i].send(b'fyh-release:'+rid.encode())
                # A ready wire backlog must not starve admission/GPU consumers.
                await asyncio.sleep(0)
        except asyncio.CancelledError: raise
        except Exception as error:
            self.errors[i]=error
            for channel in self.channels[i].values(): channel.put_nowait(error)
            async with self.condition: self.condition.notify_all()

    async def verify(self):
        await self.start()
        async def info(i):
            rid=uuid.uuid4().hex; q=asyncio.Queue(); self.channels[i][rid]=q
            try:
                await self.sockets[i].send(b'fyh-info:'+rid.encode())
                value=await asyncio.wait_for(q.get(),3)
                if isinstance(value,Exception): raise value
                if (value.get('model_path')!=self.model_path or
                    tuple(value.get('feature_capture_graph_layers',[]))!=self.layers or
                    value.get('feature_capture_early_exit_layer')!=self.early_exit):
                    raise ValueError('Native capture model/layer identity differs from manifest')
                return value
            finally: self.channels[i].pop(rid,None)
        results=await asyncio.gather(*(info(i) for i in range(len(self.endpoints))),return_exceptions=True)
        profiles=[]
        for i,value in enumerate(results):
            if isinstance(value,Exception):
                self.errors[i]=value
                for q in self.channels[i].values():q.put_nowait(RuntimeError('Native endpoint failed health check'))
            else:
                self.errors.pop(i,None);profiles.append(value)
        if not profiles:
            self.verified=False
            raise RuntimeError('No healthy native capture endpoints')
        for field in ('model_path','tokenizer_path','lora_path','json_model_override_args','version'):
            if any(p.get(field)!=profiles[0].get(field) for p in profiles):
                raise ValueError('Native endpoints differ: '+field)
        self.verified=True
        async with self.condition:self.condition.notify_all()
        return profiles

    def start_health_monitor(self):
        async def monitor():
            while not self.closed:
                await asyncio.sleep(5)
                try:await self.verify()
                except (RuntimeError,ValueError,TimeoutError):self.verified=False
        if self.health_task is None:self.health_task=asyncio.create_task(monitor())

    async def _acquire(self,n):
        if n>self.high: raise ValueError('Prompt exceeds endpoint admission budget')
        async with self.condition:
            while True:
                if self.closed: raise RuntimeError('Native capture backend closed')
                available=[]
                for i,count in enumerate(self.tokens):
                    if i in self.errors: continue
                    if self.blocked[i] and count<=self.low: self.blocked[i]=False
                    if count+n>self.high: self.blocked[i]=True
                    if not self.blocked[i] and self.requests[i]<self.max_requests: available.append(i)
                if available:
                    i=min(available,key=lambda x:(self.tokens[x],self.requests[x]))
                    self.tokens[i]+=n; self.requests[i]+=1
                    if self.tokens[i]>=self.high: self.blocked[i]=True
                    return i
                if len(self.errors)==len(self.endpoints): raise RuntimeError('All native capture endpoints failed')
                await self.condition.wait()

    async def capture(self,prompt,plan):
        if not self.verified: raise RuntimeError('Native capture identity has not been verified')
        if tuple(plan.layers)!=self.layers: raise ValueError('Capture layer plan mismatch')
        i=None; rid=uuid.uuid4().hex; completed=False
        try:
            async with asyncio.timeout(self.timeout):
                i=await self._acquire(len(prompt.input_ids))
                channel=asyncio.Queue(); self.channels[i][rid]=channel
                await self.sockets[i].send(msgspec.msgpack.encode(dict(rid=rid,input_ids=list(prompt.input_ids),
                    selected_positions=plan.positions,layer_ids=plan.layers,capture_dtype='bfloat16',stream_rows=True,transport=self.wire_transport)))
                values=None;destination=None;offset=0
                while True:
                    frame=await channel.get()
                    if isinstance(frame,Exception): raise frame
                    if 'device_features' in frame:
                        if self.gpu_receiver is None:raise ValueError('Device transport was not configured')
                        future=asyncio.get_running_loop().run_in_executor(self.gpu_pool,self._receive_device,frame['device_features'],plan)
                        try:result=await asyncio.shield(future)
                        except asyncio.CancelledError:
                            def discard(completed_future):
                                try:completed_future.result().release()
                                except Exception:pass
                            future.add_done_callback(discard)
                            raise
                        loop=asyncio.get_running_loop()
                        def release_bookkeeping():
                            def schedule():
                                if not self.closed:
                                    future=asyncio.ensure_future(self.sockets[i].send(b'fyh-release:'+rid.encode()))
                                    self.release_tasks.add(future);future.add_done_callback(self.release_tasks.discard)
                            loop.call_soon_threadsafe(schedule)
                        result.on_release=release_bookkeeping
                        completed=True;return result
                    records=frame.get('records',[])
                    if (frame.get('schema')!='fyh.feature_batch.v1' or frame.get('dtype')!='bfloat16'
                        or tuple(frame.get('layer_ids',[]))!=plan.layers or len(records)!=1):
                        raise ValueError('Native capture schema/layer mismatch')
                    record=records[0]; n=record['row_count']; wire=frame['hidden_bytes']
                    if (type(n) is not int or n<1 or record['rid']!=rid or record['row_start']!=0
                        or tuple(record['positions'])!=plan.positions[offset:offset+n]
                        or tuple(frame['shape'])!=(n,len(plan.layers),self.dim)
                        or len(wire)!=n*len(plan.layers)*self.dim*2 or len(wire)>self.max_frame_bytes):
                        raise ValueError('Native capture frame geometry/size mismatch')
                    if values is None:
                        values=torch.empty((len(plan.positions),len(plan.layers)*self.dim),dtype=torch.bfloat16)
                        destination=values.view(torch.uint8).numpy().reshape(-1)
                    begin=offset*values.shape[1]*2; end=begin+len(wire)
                    np.copyto(destination[begin:end],np.frombuffer(wire,dtype=np.uint8)); offset+=n
                    if record.get('complete',True):
                        if offset!=len(plan.positions): raise ValueError('Incomplete native capture')
                        completed=True; break
                index={p:j for j,p in enumerate(plan.positions)}
                region=values.narrow(0,index[plan.region[0]],len(plan.region))
                anchors=values[[index[p] for p in plan.anchors]]
                return region,anchors
        finally:
            if i is not None:
                leftover=self.channels[i].pop(rid,None)
                if not completed and leftover is not None:
                    while not leftover.empty():
                        frame=leftover.get_nowait()
                        if isinstance(frame,dict) and 'device_features' in frame:self._discard_device(frame['device_features'])
                if not completed:
                    try: await asyncio.wait_for(self.sockets[i].send(b'fyh-abort:'+rid.encode()),1)
                    except Exception: pass
                async with self.condition:
                    self.tokens[i]-=len(prompt.input_ids); self.requests[i]-=1; self.condition.notify_all()

    def _discard_device(self,payload):
        if self.gpu_receiver is None:return
        from .gpu_transport import import_proxy
        self.gpu_receiver.activate_context()
        with torch.cuda.device(self.gpu_receiver.index):
            for chunk in payload.get('chunks',[]):
                if 'descriptor' in chunk:
                    descriptor=chunk['descriptor'];proxy=import_proxy(descriptor)
                    self.gpu_receiver.pool_keys.add((self.gpu_receiver.index,tuple(descriptor['extra']['pool_handle'])))
                    proxy.release_without_reconstruction()

    def _receive_device(self,payload,plan):
        from .gpu_transport import DeviceFeatureLease
        self.gpu_receiver.activate_context()
        if (payload.get('schema')!='inscope.device_features.v1' or tuple(payload['layers'])!=plan.layers
            or tuple(payload['positions'])!=plan.positions or payload['hidden_dim']!=self.dim):
            raise ValueError('Device feature identity mismatch')
        values=None;borrowed=[];seen=set();placement='p2p'
        try:
            index={p:i for i,p in enumerate(plan.positions)}
            chunks=payload['chunks']
            for chunk in chunks:
                positions=tuple(chunk['positions'])
                if not positions or any(p not in index or p in seen for p in positions):
                    raise ValueError('Duplicate/unknown device feature position')
                if 'descriptor' in chunk:
                    tensor,proxy,placement=self.gpu_receiver.receive(chunk['descriptor'])
                    if proxy:borrowed.append(proxy)
                else:
                    wire=chunk['host_bytes']
                    if len(wire)!=len(positions)*len(plan.layers)*self.dim*2:raise ValueError('Bad fallback bytes')
                    tensor=torch.frombuffer(bytearray(wire),dtype=torch.bfloat16).reshape(len(positions),len(plan.layers),self.dim).to(self.gpu_receiver.device)
                    placement='host_fallback'
                if tensor.shape!=(len(positions),len(plan.layers),self.dim):raise ValueError('Invalid device feature shape')
                if len(chunks)==1 and positions==plan.positions:
                    values=tensor.flatten(1)
                else:
                    if values is None:values=torch.empty((len(plan.positions),len(plan.layers)*self.dim),device=self.gpu_receiver.device,dtype=torch.bfloat16)
                    ix=torch.tensor([index[p] for p in positions],device=self.gpu_receiver.device)
                    values.index_copy_(0,ix,tensor.flatten(1))
                seen.update(positions)
            if seen!=set(plan.positions):raise ValueError('Incomplete device feature capture')
            region=values.narrow(0,index[plan.region[0]],len(plan.region))
            anchors=values[[index[p] for p in plan.anchors]]
            result=DeviceFeatureLease(region,anchors,borrowed,self.gpu_receiver.index,placement)
            result.activate_context=self.gpu_receiver.activate_context
            with torch.cuda.device(self.gpu_receiver.index):
                result.ready_event=torch.cuda.Event();result.ready_event.record()
            return result
        except BaseException:
            if borrowed:
                with torch.cuda.device(self.gpu_receiver.index):
                    for proxy in borrowed:proxy.release_without_reconstruction()
            raise

    async def close(self):
        if self.health_task:
            self.health_task.cancel();await asyncio.gather(self.health_task,return_exceptions=True)
        if self.gpu_pool:await asyncio.to_thread(self.gpu_pool.shutdown,wait=True)
        await asyncio.sleep(0)
        if self.release_tasks:await asyncio.gather(*list(self.release_tasks),return_exceptions=True)
        self.closed=True
        async with self.condition: self.condition.notify_all()
        for channels in self.channels:
            for q in channels.values(): q.put_nowait(RuntimeError('Native backend closed'))
        for task in self.readers: task.cancel()
        await asyncio.gather(*self.readers,return_exceptions=True)
        for socket in self.sockets: socket.close(0)
        if self.context: self.context.destroy(linger=0)
        if self.gpu_receiver:self.gpu_receiver.close()
