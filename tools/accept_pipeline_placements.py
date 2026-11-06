"""Finite HTTP placement parity; no labels or evaluation metrics are loaded."""
import argparse
import json
from pathlib import Path
import httpx
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--requests',type=int,default=16)
    p.add_argument('--max-input-tokens',type=int,default=8192)
    args=p.parse_args()
    clients={name:httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=180,
        headers={'authorization':'Bearer '+(args.runtime_root/f'inscope-placement-{name}'/'api.key').read_text().strip()})
        for name,port in [('host',8305),('same',8303),('peer',8304)]}
    result=dict(requests=0,token_scores=0,max_token_error=0.,max_dp_error=0.,threshold_disagreements=0)
    try:
        with args.manifest.open() as f:
            for line in f:
                r=json.loads(line)
                if len(r['input_ids'])>args.max_input_tokens:continue
                case={k:r[k] for k in ('input_ids','pre_position','observation_tail_position','post_position','suffix_position')}
                case['composer_id']='super-simplified-v1'
                values={}
                for name,client in clients.items():
                    response=client.post('/detect',json=case);response.raise_for_status();values[name]=response.json()['detectors']
                reference=values['host']
                for name in ('same','peer'):
                    if set(values[name])!=set(reference):raise ValueError('Detector identity mismatch')
                    for detector,base in reference.items():
                        actual=values[name][detector]
                        a=np.asarray(base['token_logits']);b=np.asarray(actual['token_logits'])
                        if a.shape!=b.shape:raise ValueError('Token geometry mismatch')
                        result['token_scores']+=a.size
                        result['max_token_error']=max(result['max_token_error'],float(np.max(np.abs(a-b))))
                        for key in ('logit','region_logit','detector_logit'):
                            result['max_dp_error']=max(result['max_dp_error'],abs(base[key]-actual[key]))
                        if base['flagged']!=actual['flagged']:result['threshold_disagreements']+=1
                result['requests']+=1
                if result['requests']>=args.requests:break
        result['passed']=result['requests']==args.requests and result['threshold_disagreements']==0 and result['max_token_error']<=1e-5 and result['max_dp_error']<=1e-5
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        if not result['passed']:raise RuntimeError('Placement parity failed')
    finally:
        for client in clients.values():client.close()


if __name__=='__main__':main()
