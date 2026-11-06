"""Compose -> plan -> capture -> queued detector bank -> deployment policy.

No labels, calibration search or evaluation metrics belong to this pipeline.
"""
import asyncio
import time
from .scheduler import Overloaded


class DetectionPipeline:
    def __init__(self,composer,planner,backend,scheduler,postprocessor,*,max_capture_bytes):
        if max_capture_bytes<1: raise ValueError('Positive host capture budget required')
        self.composer=composer; self.planner=planner; self.backend=backend
        self.scheduler=scheduler; self.postprocessor=postprocessor
        self.max_capture_bytes=max_capture_bytes; self.capture_bytes=0
        self.active=0; self.completed=0; self.failed=0

    async def detect(self,request):
        times={}; began=time.monotonic()
        prompt=await asyncio.to_thread(self.composer.compose,request)
        enabled=prompt.metadata.get('enabled_detectors')
        names={e['name'] for e in self.postprocessor.entries}
        if enabled is not None and (not enabled or len(set(enabled))!=len(enabled) or not set(enabled)<=names):
            raise ValueError('Requested detector names are not loaded or are duplicated')
        times['compose']=time.monotonic()-began; stage=time.monotonic()
        plan=self.planner.plan(prompt,tuple(self.scheduler.bank.layers))
        if len(plan.region)>self.scheduler.max_request_positions:raise ValueError('DP exceeds configured request-position limit')
        times['plan']=time.monotonic()-stage
        # Destination plus queued BF16 wire memory; reserve before requesting GPU
        # work. The caller configures available-RAM safety, not a hidden DP cap.
        size=len(plan.positions)*len(plan.layers)*self.scheduler.bank.dim*4
        if self.capture_bytes+size>self.max_capture_bytes: raise Overloaded('Host capture byte budget exhausted')
        self.capture_bytes+=size; self.active+=1
        features=None
        try:
            stage=time.monotonic(); features=await self.backend.capture(prompt,plan)
            times['capture']=time.monotonic()-stage; stage=time.monotonic()
            row=dict(region_positions=list(plan.region))
            scores=await self.scheduler.submit((0,row,features))
            times['queue_and_infer']=time.monotonic()-stage; stage=time.monotonic()
            result=await asyncio.to_thread(self.postprocessor.process,scores,prompt,plan)
            times['postprocess']=time.monotonic()-stage; times['total']=time.monotonic()-began
            result['transfer']=dict(placement=getattr(features,'placement','host'))
            result['latency_ms']={k:v*1000 for k,v in times.items()}
            self.completed+=1; return result
        except ValueError as error:
            self.failed+=1
            raise RuntimeError('Native features or detector outputs violate the declared contract') from error
        except BaseException:
            self.failed+=1; raise
        finally:
            if features is not None and hasattr(features,'release') and not features.scheduler_owned:features.release()
            self.capture_bytes-=size; self.active-=1

    def status(self):
        return dict(active=self.active,completed=self.completed,failed=self.failed,
            capture_bytes=self.capture_bytes,max_capture_bytes=self.max_capture_bytes,
            inference_pending_bytes=self.scheduler.pending_bytes,
            inference_queue_depth=len(self.scheduler.queue),**self.scheduler.metrics)

    async def close(self):
        await self.scheduler.close()
        await self.backend.close()
