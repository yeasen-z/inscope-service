"""Import pinned checkpoints and EXISTING calibration; never fit thresholds.

Run from a checkout: PYTHONPATH=. python tools/import_token_probe_plan.py ...
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--composer-id',required=True)
    p.add_argument('--endpoint',action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-capture-gib',type=float,help='Explicit override; default is measured available RAM minus safety')
    p.add_argument('--safety-gib',type=float,default=256)
    args=p.parse_args()
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    budget=(int(args.max_capture_gib*1024**3) if args.max_capture_gib is not None else memory['MemAvailable']-int(args.safety_gib*1024**3))
    if budget<=0:raise ValueError('Insufficient host RAM after safety reserve')
    source=json.loads(args.plan.read_text())
    calibration={r['model']:r for r in json.loads(args.calibration.read_text())}
    detectors=[]
    for job in source['checkpoints']:
        name=f"small-L{job['layer']}"; row=calibration[name]
        if row['step']!=source['step']:raise ValueError('Calibration step mismatch')
        detectors.append(dict(name=name,layer=job['layer'],logit_threshold=row['logit_threshold'],
            threshold_source=str(args.calibration.resolve())))
    config=dict(schema='inscope.detection_pipeline.v1',composer_id=args.composer_id,
        bank=dict(layers=source['layers'],checkpoints=source['checkpoints']),detectors=detectors,
        max_capture_bytes=budget,max_tokens=131072,
        capture=dict(endpoints=args.endpoint,model_path=source['model'],early_exit=source['early_exit'],
            token_low=3145728,token_high=5242880,max_requests=4096),
        inference=dict(max_positions=32768,max_dps=128,max_pending_bytes=16*1024**3,wait_ms=2))
    from inscope_server.detection.config import load_config
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Refusing to overwrite an existing deployment config')
    args.output.write_text(json.dumps(config,indent=2)+'\n');load_config(args.output)
    print(args.output)


if __name__=='__main__':main()
