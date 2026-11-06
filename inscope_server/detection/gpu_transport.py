"""Placement-aware local CUDA IPC transport using sglang-inscope's lease pool.

Only small, schema-checked control metadata crosses IPC. Feature payloads stay
on device. Descriptors are local capabilities and must never be exposed on an
unauthenticated/public network. No pickle is accepted by this adapter.
"""
import math
import threading
import torch

SCHEMA='inscope.cuda_feature_lease.v1'


def export_proxy(proxy,producer_uuid):
    e=dict(proxy.proxy_state['ipc_extra'])
    for name in ('dtype','recons_dtype'): e[name]=str(e[name]).removeprefix('torch.')
    for name in ('shape','stride','recons_shape'): e[name]=list(e[name])
    return dict(schema=SCHEMA,producer_uuid=str(producer_uuid),extra=e,
        ready_byte_offset=proxy.ready_byte_offset,ack_byte_offset=proxy.ack_byte_offset,
        generation=proxy.generation,total_consumer_count=proxy.total_consumer_count)


def import_proxy(value):
    from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
    if value.get('schema')!=SCHEMA: raise ValueError('Unsupported CUDA lease schema')
    e=dict(value['extra'])
    if e['dtype']!='uint8' or e['recons_dtype']!='bfloat16':
        raise ValueError('Only BF16 native feature payloads are accepted')
    e['dtype']=torch.uint8; e['recons_dtype']=torch.bfloat16
    for name in ('shape','stride','recons_shape'):
        if not isinstance(e[name],(tuple,list)) or any(type(x) is not int or x<0 for x in e[name]):
            raise ValueError('Invalid CUDA feature geometry')
        e[name]=tuple(e[name])
    if len(e['shape'])!=1 or e['stride']!=(1,) or len(e['recons_shape'])!=3:
        raise ValueError('Expected packed [positions,layers,hidden] features')
    if math.prod(e['recons_shape'])*2!=e['nbytes'] or e['shape'][0]!=e['nbytes']:
        raise ValueError('Invalid CUDA feature byte count')
    if type(e['pool_byte_offset']) is not int or e['pool_byte_offset']<0 or e['storage_offset']!=0:
        raise ValueError('Invalid CUDA pool offset')
    handle=tuple(e['pool_handle'])
    if len(handle)!=8 or type(handle[2]) is not int or e['pool_byte_offset']+e['nbytes']>handle[2]:
        raise ValueError('CUDA feature outside exported pool')
    e['pool_handle']=handle; e['use_pool_handle_cache']=True
    for name in ('ready_byte_offset','ack_byte_offset'):
        x=value[name]
        if type(x) is not int or x<0 or x%4 or x+4>handle[2]:raise ValueError('Invalid CUDA lease control offset')
    if type(value['generation']) is not int or not 0<value['generation']<=0x7fffffff or value['total_consumer_count']!=1:
        raise ValueError('Invalid CUDA lease generation/consumer count')
    proxy=CudaIpcTensorTransportProxy.__new__(CudaIpcTensorTransportProxy)
    proxy._init_stream_ordered_consumer(ready_byte_offset=value['ready_byte_offset'],
        ack_byte_offset=value['ack_byte_offset'],generation=value['generation'],
        total_consumer_count=1,transport_name='CUDA IPC')
    proxy.proxy_state=dict(ipc_extra=e,tensor_data=None); proxy.reconstruct_tensor=None
    proxy._pool_storage=None; proxy._borrowed_storage=None
    proxy._borrowed_base_address=None; proxy._borrowed_device_id=None
    return proxy


class DeviceFeatureLease:
    """Tuple-compatible features; scheduler owns release after in-flight reads."""
    def __init__(self,region,anchors,proxies=(),device=None,placement='p2p'):
        self.values=(region,anchors);self.proxies=list(proxies);self.device=device
        self.ready_event=None
        self.activate_context=None
        self.on_release=None
        self.placement=placement;self.scheduler_owned=False;self.released=False
        self.lock=threading.Lock()
    def __getitem__(self,i):return self.values[i]
    def __iter__(self):return iter(self.values)
    def release(self):
        with self.lock:
            if self.released:return
            if self.proxies:
                if self.activate_context:self.activate_context()
                with torch.cuda.device(self.device):
                    if self.ready_event is not None:torch.cuda.current_stream(self.device).wait_event(self.ready_event)
                    for proxy in self.proxies:proxy.release_without_reconstruction()
            self.proxies.clear();self.released=True
            if self.on_release:self.on_release()


class CudaFeatureReceiver:
    def __init__(self,device='cuda'):
        self.device=torch.device(device)
        self.index=self.device.index if self.device.index is not None else torch.cuda.current_device()
        self.uuid=str(torch.cuda.get_device_properties(self.index).uuid)
        # Driver contexts are thread-local; cached IPC mappings can skip the
        # runtime calls that would initialize a new transfer worker thread.
        from cuda.bindings import driver
        from sglang.srt.utils.cuda_vmm_utils import check_drv
        self.driver=driver;self.check_drv=check_drv
        with torch.cuda.device(self.index):
            torch.empty(0,device=self.index)
            self.context=check_drv(driver.cuCtxGetCurrent(),'cuCtxGetCurrent(features)')
        if not self.context:raise RuntimeError('Missing feature receiver CUDA context')
        self.same_gpu=0;self.peer_copies=0;self.device_bytes=0;self.pool_keys=set()

    def activate_context(self):
        self.check_drv(self.driver.cuCtxSetCurrent(self.context),'cuCtxSetCurrent(features)')

    def receive(self,descriptor):
        self.activate_context()
        proxy=import_proxy(descriptor)
        self.pool_keys.add((self.index,tuple(descriptor["extra"]["pool_handle"])))
        same=descriptor['producer_uuid']==self.uuid
        with torch.cuda.device(self.index):
            if same:
                value=proxy.borrow_on_target_device(self.index)
                if value is None:raise RuntimeError('CUDA lease cannot be borrowed')
                self.same_gpu+=1
                return value,proxy,'same_gpu'
            value=proxy.reconstruct_on_target_device(self.index)
            self.peer_copies+=1;self.device_bytes+=value.numel()*value.element_size()
            return value,None,'p2p'

    def close(self):
        """Close owned mappings only after callers have released all views."""
        from sglang.srt.multimodal.transport.cuda_ipc import _pool_handle_cache_invalidate
        self.activate_context()
        torch.cuda.synchronize(self.index)
        for key in self.pool_keys:_pool_handle_cache_invalidate(key)
        self.pool_keys.clear()
