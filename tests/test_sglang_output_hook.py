import sys
from types import SimpleNamespace
import torch
import pytest
from sglang.srt.feature_capture.recorder import FeatureDeviceOutput,OUTPUT_FACTORY_API_VERSION


def test_default_capture_transport_unchanged(monkeypatch):
    monkeypatch.delenv('SGLANG_FEATURE_OUTPUT_FACTORY',raising=False)
    source=FeatureDeviceOutput(tensor=torch.ones(1,2),plan=())
    output=source.copy_to_host(lambda x:x+1)
    assert torch.equal(output.tensor,torch.full((1,2),2.))
    assert OUTPUT_FACTORY_API_VERSION==1


def test_operator_factory_receives_device_output_without_forcing_d2h(monkeypatch):
    seen=[]
    def factory(source,copy):seen.append(source);return 'device-output'
    monkeypatch.setitem(sys.modules,'test_device_factory',SimpleNamespace(make=factory))
    monkeypatch.setenv('SGLANG_FEATURE_OUTPUT_FACTORY','test_device_factory:make')
    source=FeatureDeviceOutput(tensor=torch.ones(1,2),plan=())
    def no_copy(*a):raise AssertionError('D2H was forced')
    assert source.copy_to_host(no_copy)=='device-output' and seen==[source]
    monkeypatch.setenv('SGLANG_FEATURE_OUTPUT_FACTORY','invalid')
    with pytest.raises(ValueError):source.copy_to_host(no_copy)


def test_device_bridge_uses_tokenizer_record_list_contract():
    from inscope_server.detection.sglang_worker import device_payload
    record=dict(schema='inscope.device_features.v1',chunks=[])
    assert device_payload({'meta_info':{'features':[record]}})==record
    for bad in (['schema','chunks'],[],[{},{}],{'schema':'wrong'}):
        with pytest.raises(ValueError):device_payload({'meta_info':{'features':bad}})


def test_fragmented_feature_pool_preserves_every_position():
    import torch
    from inscope_server.detection.sglang_ipc_output import wrap_feature_rows
    class Pool:
        def wrap_tensor(self,tensor,**kwargs):
            return tensor.clone() if len(tensor)<=3 else None
        def cancel_proxy(self,proxy):raise AssertionError('Valid fragmented allocation was cancelled')
    x=torch.arange(10).reshape(10,1,1)
    chunks=wrap_feature_rows(Pool(),x,tuple(range(100,110)))
    assert tuple(p for ps,_ in chunks for p in ps)==tuple(range(100,110))
    assert torch.equal(torch.cat([t for _,t in chunks]),x)


def test_partial_feature_allocation_failure_releases_only_new_chunks():
    import pytest,torch
    from inscope_server.detection.sglang_ipc_output import wrap_feature_rows
    cancelled=[]
    class Pool:
        calls=0
        def wrap_tensor(self,tensor,**kwargs):
            self.calls+=1
            return 'lease' if self.calls==2 else None
        def cancel_proxy(self,proxy):cancelled.append(proxy)
    with pytest.raises(RuntimeError,match='one complete feature row'):
        wrap_feature_rows(Pool(),torch.zeros(10,1,1),tuple(range(10)))
    assert cancelled==['lease']
