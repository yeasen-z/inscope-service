import asyncio
from types import SimpleNamespace
import torch
import pytest
from fastapi.testclient import TestClient
from inscope_server.detection.composer import CompiledComposer
from inscope_server.detection.planner import AllPositionPlanner
from inscope_server.detection.postprocess import CalibratedPostProcessor
from inscope_server.detection.pipeline import DetectionPipeline
from inscope_server.detection.scheduler import InferenceScheduler,Overloaded
from inscope_server.detection.app import create_pipeline_app


def request():
    return dict(composer_id='frozen-v1',input_ids=list(range(8)),pre_position=1,
        observation_tail_position=3,post_position=5,suffix_position=6)


class Bank:
    layers=[0,3]; dim=4
    def __init__(self): self.calls=[]
    def predict(self,batch):
        self.calls.append(batch)
        sizes=[len(r['region_positions']) for _,r,_ in batch]
        tokens=torch.cat([v[0][:,:2] for _,_,v in batch])
        dps=torch.stack([v[0][0,:2] for _,_,v in batch])
        return dict(token_logits=tokens,region_logits=dps,detector_logits=dps,logits=dps)


class Backend:
    errors={}
    async def verify(self): pass
    async def close(self): self.closed=True
    async def capture(self,prompt,plan):
        return torch.ones(len(plan.region),8,dtype=torch.bfloat16),torch.ones(2,8,dtype=torch.bfloat16)


def pipeline():
    return DetectionPipeline(CompiledComposer('frozen-v1'),AllPositionPlanner(),Backend(),
        InferenceScheduler(Bank(),wait_ms=2),CalibratedPostProcessor([
            dict(name='a',layer=0,logit_threshold=0.),dict(name='b',layer=3,logit_threshold=2.)]),
        max_capture_bytes=100000)


def test_composer_preserves_ids_planner_union_and_no_labels():
    r=request(); prompt=CompiledComposer('frozen-v1').compose(r)
    plan=AllPositionPlanner().plan(prompt,(0,3))
    assert list(prompt.input_ids)==r['input_ids']
    assert plan.region==(1,2,3) and plan.anchors==(5,6)
    assert plan.positions==(0,1,2,3,5,6)
    for change in ({'composer_id':'wrong'},{'input_ids':[True,2]}, {'post_position':2}):
        with pytest.raises(ValueError): CompiledComposer('frozen-v1').compose({**r,**change})


def test_scheduler_batches_concurrent_requests_and_restores_order():
    async def run():
        bank=Bank(); scheduler=InferenceScheduler(bank,max_positions=6,wait_ms=10)
        def item(i,n):return i,dict(region_positions=list(range(n))),(torch.full((n,8),i,dtype=torch.bfloat16),torch.ones(2,8,dtype=torch.bfloat16))
        try:
            results=await asyncio.gather(*(scheduler.submit(item(i,n)) for i,n in enumerate([2,3,2,2])))
            assert [r['logits'][0,0].item() for r in results]==[0,1,2,3]
            assert len(bank.calls)==2 and all(sum(len(r['region_positions']) for _,r,_ in b)<=6 for b in bank.calls)
            assert scheduler.pending_bytes==0
        finally:await scheduler.close()
    asyncio.run(run())


def test_scheduler_rejection_cancellation_and_shutdown():
    async def run():
        scheduler=InferenceScheduler(Bank(),max_pending_bytes=65,wait_ms=20)
        item=(0,dict(region_positions=[0,1]),(torch.ones(2,8,dtype=torch.bfloat16),torch.ones(2,8,dtype=torch.bfloat16)))
        task=asyncio.create_task(scheduler.submit(item)); await asyncio.sleep(0)
        with pytest.raises(Overloaded):await scheduler.submit(item)
        task.cancel(); await asyncio.gather(task,return_exceptions=True)
        await scheduler.close()
        assert scheduler.pending_bytes==0 and not scheduler.queue
        with pytest.raises(RuntimeError):await scheduler.submit(item)
    asyncio.run(run())


def test_http_pipeline_auth_validation_scores_limits_and_cleanup():
    p=pipeline(); app=create_pipeline_app(api_key='k',pipeline=p,max_body_bytes=1000)
    headers={'authorization':'Bearer k'}
    with TestClient(app) as client:
        assert client.get('/health/live').status_code==200
        assert client.get('/health/ready').status_code==401
        assert client.get('/health/ready',headers=headers).status_code==200
        assert client.post('/detect',json=request()).status_code==401
        response=client.post('/detect',json=request(),headers=headers)
        assert response.status_code==200,response.text
        result=response.json(); assert result['detectors']['a']['flagged'] and not result['detectors']['b']['flagged']
        assert result['detectors']['a']['token_positions']==[1,2,3]
        assert 'total' in result['latency_ms']
        assert client.post('/detect',json={**request(),'label':1},headers=headers).status_code==422
        assert client.post('/detect',json={**request(),'composer_id':'wrong'},headers=headers).status_code==422
        assert client.post('/detect',content=b'x'*1001,headers=headers).status_code==413
        assert client.get('/status',headers=headers).json()['completed']==1
    assert p.backend.closed


def test_timeout_cancels_capture_and_releases_host_budget():
    async def delayed(*a):await asyncio.Event().wait()
    p=pipeline(); p.backend.capture=delayed
    app=create_pipeline_app(api_key='k',pipeline=p,timeout=.02)
    with TestClient(app) as client:
        r=client.post('/detect',json=request(),headers={'authorization':'Bearer k'})
        assert r.status_code==504
        assert p.capture_bytes==0 and p.active==0


def test_scheduler_long_singleton_is_not_truncated_to_batch_target():
    async def run():
        scheduler=InferenceScheduler(Bank(),max_positions=4,max_request_positions=10,wait_ms=0)
        item=(0,dict(region_positions=list(range(9))),(torch.ones(9,8,dtype=torch.bfloat16),torch.ones(2,8,dtype=torch.bfloat16)))
        try:
            result=await scheduler.submit(item)
            assert len(result['token_logits'])==9
        finally:await scheduler.close()
    asyncio.run(run())


def test_named_detector_filter_scores_only_and_backend_contract_failures():
    p=pipeline();headers={'authorization':'Bearer k'}
    with TestClient(create_pipeline_app(api_key='k',pipeline=p)) as client:
        r=client.post('/detect',json={**request(),'output':'scores','enabled_detectors':['b']},headers=headers)
        assert r.status_code==200,r.text
        assert list(r.json()['detectors'])==['b'] and 'token_logits' not in r.json()['detectors']['b']
        assert client.post('/detect',json={**request(),'enabled_detectors':['missing']},headers=headers).status_code==422
        async def corrupt(*a):raise ValueError('bad native features')
        p.backend.capture=corrupt
        assert client.post('/detect',json=request(),headers=headers).status_code==503
