"""Byte-accounted microbatching, with one GPU worker and request isolation."""
import asyncio
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class Overloaded(RuntimeError): pass


class InferenceScheduler:
    def __init__(self,bank,*,max_positions=32768,max_dps=128,max_pending_bytes=16*1024**3,wait_ms=2,max_request_positions=131072):
        if min(max_positions,max_dps,max_pending_bytes) < 1 or not 0 <= wait_ms <= 1000:
            raise ValueError('Invalid inference admission/batching limits')
        if max_request_positions<max_positions: raise ValueError("Request position limit must cover batch target")
        self.max_request_positions=max_request_positions
        self.bank=bank; self.max_positions=max_positions; self.max_dps=max_dps
        self.max_pending_bytes=max_pending_bytes; self.wait_ms=wait_ms
        self.pending_bytes=0; self.queue=deque(); self.wakeup=asyncio.Event()
        self.worker=None; self.closed=False
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='inscope-probe')
        self.metrics=dict(batches=0,dps=0,positions=0,queue_seconds=0.,compute_seconds=0.,rejected=0)

    async def start(self):
        if self.worker is None: self.worker=asyncio.create_task(self._run())

    async def submit(self,item):
        if self.closed: raise RuntimeError('Detector scheduler closed')
        if self.worker is not None and self.worker.done():raise RuntimeError('Detector worker stopped')
        n=len(item[1]['region_positions'])
        size=sum(t.numel()*t.element_size() for t in item[2])
        if n>self.max_request_positions: raise ValueError('DP exceeds configured request-position limit')
        if self.pending_bytes+size>self.max_pending_bytes:
            self.metrics['rejected']+=1; raise Overloaded('Detector queue byte budget exhausted')
        await self.start()
        future=asyncio.get_running_loop().create_future()
        if hasattr(item[2],'scheduler_owned'):item[2].scheduler_owned=True
        self.pending_bytes+=size; self.queue.append((item,future,size,time.monotonic())); self.wakeup.set()
        return await future

    async def _run(self):
        try:
            while True:
                if self.closed and not self.queue: return
                await self.wakeup.wait()
                if self.closed and not self.queue: return
                if self.wait_ms: await asyncio.sleep(self.wait_ms/1000)
                batch=[]; count=0
                while self.queue and len(batch)<self.max_dps:
                    entry=self.queue[0]; n=len(entry[0][1]['region_positions'])
                    if batch and count+n>self.max_positions: break
                    self.queue.popleft()
                    if entry[1].cancelled():
                        self.pending_bytes-=entry[2]
                        if hasattr(entry[0][2],'release'):entry[0][2].release()
                        continue
                    batch.append(entry); count+=n
                if not self.queue: self.wakeup.clear()
                if not batch: continue
                began=time.monotonic()
                self.metrics['queue_seconds']+=sum(began-e[3] for e in batch)
                try:
                    values=await asyncio.get_running_loop().run_in_executor(self.pool,self.bank.predict,[e[0] for e in batch])
                    offset=0
                    for i,(item,future,size,_) in enumerate(batch):
                        n=len(item[1]['region_positions'])
                        row = {
                            k: None if v is None else v[offset:offset+n] if k=='token_logits' else v[i:i+1]
                            for k, v in values.items()
                        }
                        if not future.done(): future.set_result(row)
                        offset+=n
                except Exception as error:
                    for _,future,_,_ in batch:
                        if not future.done(): future.set_exception(error)
                finally:
                    for item,_,_,_ in batch:
                        if hasattr(item[2],'release'):item[2].release()
                    self.pending_bytes-=sum(e[2] for e in batch)
                    self.metrics['batches']+=1; self.metrics['dps']+=len(batch); self.metrics['positions']+=count
                    self.metrics['compute_seconds']+=time.monotonic()-began
        finally:
            for item,future,_,_ in self.queue:
                if hasattr(item[2],'release'):item[2].release()
                if not future.done(): future.set_exception(RuntimeError('Detector scheduler stopped'))
            self.queue.clear(); self.pending_bytes=0

    async def close(self):
        self.closed=True; self.wakeup.set()
        if self.worker: await self.worker
        await asyncio.to_thread(self.pool.shutdown,wait=True,cancel_futures=True)
