"""Owned local pipeline lifecycle. No model/feature engine is started or stopped."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status','stop'])
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--port',type=int,default=8303);p.add_argument('--gpu',default='0');args=p.parse_args()
    root=args.runtime.resolve();root.mkdir(parents=True,exist_ok=True)
    owner=root/'owner.json'
    value=json.loads(owner.read_text()) if owner.exists() else None
    def owned():
        if not value:return False
        cmd=Path(f"/proc/{value['pid']}/cmdline")
        return cmd.exists() and b'inscope_server.cli' in cmd.read_bytes() and str(value['config']).encode() in cmd.read_bytes()
    if args.action=='status':print(json.dumps(dict(owner=value,alive=owned()),indent=2));return
    if args.action=='stop':
        if owned():os.kill(value['pid'],signal.SIGTERM)
        print('Owned API stop requested; capture services untouched');return
    if owned():raise RuntimeError('Owned pipeline already running')
    if not args.config:raise ValueError('--config is required for start')
    key=root/'api.key'
    if not key.exists():
        fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32))
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',args.port))
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]),
        'CUDA_VISIBLE_DEVICES':args.gpu,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
        'INSCOPE_API_KEY':key.read_text().strip()}
    with (root/'server.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-m','inscope_server.cli','--pipeline-config',str(args.config.resolve()),
            '--host','127.0.0.1','--port',str(args.port)],env=env,stdout=log,stderr=log,start_new_session=True)
    owner.write_text(json.dumps(dict(pid=child.pid,config=str(args.config.resolve()),port=args.port,gpu=args.gpu),indent=2)+'\n')
    print(child.pid)


if __name__=='__main__':main()
