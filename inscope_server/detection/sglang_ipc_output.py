"""Optional sglang-inscope device-output factory, before any D2H copy.

Activate only in a dedicated native capture worker via
SGLANG_FEATURE_OUTPUT_FACTORY=inscope_server.detection.sglang_ipc_output:copy_to_ipc.
A full pool fails explicitly by default; operators may select host fallback.
"""
import os
import threading
import torch
from .gpu_transport import export_proxy

_pools={};_lock=threading.Lock()


def pool_for(device):
    from sglang.srt.multimodal.transport.cuda_ipc import MmItemMemoryPool
    with _lock:
        if device not in _pools:
            size=int(os.environ.get('INSCOPE_GPU_FEATURE_POOL_BYTES',str(256*1024**2)))
            _pools[device]=MmItemMemoryPool(memory_size=size,recycle_interval=.01,
                base_gpu_id=device,consumer_count=1,max_inflight_slices=4096)
        return _pools[device]


def wrap_feature_rows(pool,tensor,positions):
    """Preserve whole-tensor fast path; fit fragments without dropping rows."""
    result=[];offset=0
    try:
        while offset<len(positions):
            count=len(positions)-offset
            while True:
                proxy=pool.wrap_tensor(tensor[offset:offset+count],use_pool_handle_cache=True)
                if proxy is not None:break
                native=getattr(pool,'_pool',None)
                if native is not None:
                    with native._lock,torch.cuda.device(pool.device_id):native._recycle_ready_leases_locked()
                    proxy=pool.wrap_tensor(tensor[offset:offset+count],use_pool_handle_cache=True)
                    if proxy is not None:break
                if count==1:raise RuntimeError('GPU feature pool exhausted even for one complete feature row')
                count=max(1,count//2)
            result.append((positions[offset:offset+count],proxy));offset+=count
        return result
    except BaseException:
        for _,proxy in result:pool.cancel_proxy(proxy)
        raise


class RequestState:
    def __init__(self):self.chunks=[];self.seen=set();self.proxies=[];self.published=False;self.failed=False
    def cancel(self):
        if not self.published:
            for pool,proxy in self.proxies:
                try:pool.cancel_proxy(proxy)
                except RuntimeError:pass
            self.proxies.clear()
    def __del__(self):
        try:self.cancel()
        except Exception:pass


class IpcFeatureOutput:
    def __init__(self,tensor,plan):self.tensor=tensor;self.plan=plan
    def consume(self,batch,commits):
        by_request={}
        for slot in self.plan:
            offset=slot.out_start
            for rid,positions in slot.segments:
                by_request.setdefault(rid,{})[slot.layer]=(positions,self.tensor[offset:offset+len(positions)])
                offset+=len(positions)
        requests={r.rid:r for r in batch.reqs}
        for rid,blocks in by_request.items():
            request=requests[rid]
            if request.is_retracted or request.feature_capture is None:continue
            spec=request.feature_capture;layers=tuple(spec.layers)
            positions=tuple(blocks[layers[0]][0])
            if any(tuple(blocks[l][0])!=positions for l in layers):
                raise RuntimeError('Device transport requires identical selected positions across layers')
            tensor=torch.stack([blocks[l][1] for l in layers],1).contiguous()
            if tensor.dtype!=torch.bfloat16:raise ValueError('Device feature transport requires BF16')
            state=getattr(request,'_inscope_device_features',None)
            if state is None:state=RequestState();request._inscope_device_features=state
            if state.failed:continue
            if any(p in state.seen for p in positions) and not state.published:
                state.cancel();state=RequestState();request._inscope_device_features=state
            device=tensor.device.index;pool=pool_for(device)
            try:
                pieces=wrap_feature_rows(pool,tensor,positions)
            except RuntimeError as error:
                state.failed=True;state.cancel()
                if request.customized_info is None:request.customized_info={}
                request.customized_info['features']=[dict(schema='inscope.device_features.v1',error=str(error))]
                continue
            uuid=torch.cuda.get_device_properties(device).uuid
            for selected,proxy in pieces:
                state.proxies.append((pool,proxy))
                state.chunks.append(dict(positions=selected,descriptor=export_proxy(proxy,uuid)))
            state.seen.update(positions)
            if state.seen==set(spec.positions):
                state.published=True
                if request.customized_info is None:request.customized_info={}
                request.customized_info['features']=[dict(schema='inscope.device_features.v1',
                    layers=layers,positions=spec.positions,hidden_dim=tensor.shape[-1],chunks=state.chunks)]


def copy_to_ipc(output,copy_tensor):
    # Deliberately do not invoke copy_tensor: features stay on the model GPU.
    return IpcFeatureOutput(output.tensor,output.plan)
