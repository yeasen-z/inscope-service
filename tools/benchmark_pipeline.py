"""HTTP/native capture load canary; inference timing only, no detector metrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import httpx


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8303')
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--requests',type=int,default=64)
    p.add_argument('--concurrency',type=int,default=16);p.add_argument('--max-input-tokens',type=int);p.add_argument('--scores-only',action='store_true');args=p.parse_args()
    headers={'authorization':'Bearer '+args.key_file.read_text().strip()}
    cases=[]
    with args.manifest.open() as f:
        for line in f:
            r=json.loads(line)
            if args.max_input_tokens and len(r['input_ids'])>args.max_input_tokens:continue
            cases.append({k:r[k] for k in ('input_ids','pre_position','observation_tail_position','post_position','suffix_position')})
            if len(cases)>=args.requests:break
    for c in cases:
        c['composer_id']='super-simplified-v1'
        if args.scores_only:c['output']='scores'
    with httpx.Client(base_url=args.url,headers=headers,timeout=180,limits=httpx.Limits(max_connections=args.concurrency)) as client:
        ready=client.get('/health/ready');ready.raise_for_status()
        before=client.get('/status').json()
        def one(case):
            began=time.perf_counter();r=client.post('/detect',json=case);r.raise_for_status()
            value=r.json()
            if len(value['detectors'])!=10:raise ValueError('Missing detector output')
            for d in value['detectors'].values():
                if not args.scores_only and len(d['token_logits'])!=case['observation_tail_position']-case['pre_position']+1:
                    raise ValueError('Missing token scores')
            return time.perf_counter()-began,value['latency_ms']
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:results=list(pool.map(one,cases))
        seconds=time.perf_counter()-start;after=client.get('/status').json()
    times=sorted(v[0] for v in results)
    result=dict(requests=len(cases),concurrency=args.concurrency,seconds=seconds,dps_per_second=len(cases)/seconds,
        latency_ms=dict(p50=statistics.median(times)*1000,p95=times[int((len(times)-1)*.95)]*1000),
        mean_stage_ms={k:statistics.mean(v[1][k] for v in results) for k in results[0][1]},
        batches=after['batches']-before['batches'],capture_bytes_after=after['capture_bytes'],
        inference_bytes_after=after['inference_pending_bytes'],ready=ready.json())
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
