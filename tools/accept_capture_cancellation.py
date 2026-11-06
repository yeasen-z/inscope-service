"""Finite native cancellation/recovery canary; no evaluation labels are loaded."""
import argparse
import asyncio
import json
from pathlib import Path
from inscope_server.detection.composer import CompiledComposer
from inscope_server.detection.planner import AllPositionPlanner
from inscope_server.detection.native_backend import NativeCaptureBackend


async def run(args):
    config=json.loads(args.config.read_text())
    backend=NativeCaptureBackend(hidden_dim=5120,layers=config['bank']['layers'],gpu_device='cuda:0',**config['capture'])
    case=json.loads(args.manifest.open().readline());case['composer_id']=config['composer_id']
    prompt=CompiledComposer(config['composer_id'],262144).compose(case)
    plan=AllPositionPlanner().plan(prompt,tuple(config['bank']['layers']))
    result=dict(cancelled=0,recovered=0,input_tokens=len(prompt.input_ids),positions=len(plan.positions))
    try:
        await backend.verify()
        for delay in (.05,.2,.5):
            task=asyncio.create_task(backend.capture(prompt,plan))
            await asyncio.sleep(delay)
            if not task.done():
                task.cancel();await asyncio.gather(task,return_exceptions=True);result['cancelled']+=1
            else:task.result().release()
            # Let aborted engine results drain before testing fresh full-size allocation.
            for _ in range(60):
                profiles=await backend.verify()
                if all(p.get('reserved_bytes',0)==0 for p in profiles):break
                await asyncio.sleep(.1)
            lease=await backend.capture(prompt,plan)
            lease.release();result['recovered']+=1
        await asyncio.sleep(.1)
        profiles=await backend.verify()
        result['reserved_bytes_after']=[p.get('reserved_bytes',0) for p in profiles]
        result['client_requests_after']=backend.requests
        result['client_tokens_after']=backend.tokens
        result['passed']=result['cancelled']>0 and result['recovered']==3 and not any(result['reserved_bytes_after'])
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        if not result['passed']:raise RuntimeError('Cancellation recovery failed')
    finally:await backend.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    asyncio.run(run(p.parse_args()))


if __name__=='__main__':main()
