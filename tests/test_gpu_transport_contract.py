import asyncio
import threading
import torch
from inscope_server.detection.scheduler import InferenceScheduler


def test_cancelled_inflight_request_releases_device_lease_after_gpu_worker():
    async def run():
        started=threading.Event();finish=threading.Event();released=[]
        class Lease:
            scheduler_owned=False
            def __iter__(self):return iter((torch.zeros(2,8),torch.zeros(2,8)))
            def __getitem__(self,i):return list(self)[i]
            def release(self):released.append(True)
        class Bank:
            def predict(self,batch):
                started.set()
                assert finish.wait(2)
                return dict(token_logits=torch.zeros(2,1),logits=torch.zeros(1,1))
        scheduler=InferenceScheduler(Bank(),wait_ms=0);lease=Lease()
        task=asyncio.create_task(scheduler.submit((0,dict(region_positions=[0,1]),lease)))
        for _ in range(200):
            if started.is_set():break
            await asyncio.sleep(.001)
        assert started.is_set() and lease.scheduler_owned
        task.cancel();await asyncio.gather(task,return_exceptions=True)
        assert not released
        finish.set();await scheduler.close()
        assert released==[True] and scheduler.pending_bytes==0
    asyncio.run(run())


def test_host_copy_does_not_block_loop_and_cancellation_waits_for_read(monkeypatch):
    from inscope_server.detection.sglang_worker import send_host_frames
    from inscope_server.detection import host_transfer
    started=threading.Event();finish=threading.Event();closed=[]
    def frames(*args):
        try:
            started.set()
            assert finish.wait(2)
            yield {'frame':1}
        finally:closed.append(True)
    monkeypatch.setattr(host_transfer,'host_frames',frames)
    async def run():
        async def send(frame):raise AssertionError('Cancelled frame was dispatched')
        task=asyncio.create_task(send_host_frames({},'rid',None,send))
        for _ in range(200):
            if started.is_set():break
            await asyncio.sleep(.001)
        assert started.is_set()
        task.cancel();await asyncio.sleep(.01)
        assert not task.done() and not closed
        finish.set();await asyncio.gather(task,return_exceptions=True)
        assert closed==[True]
    asyncio.run(run())
