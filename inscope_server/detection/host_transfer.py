"""Explicit GPU-to-host compatibility adapter for placement A/B acceptance."""
import torch
from .gpu_transport import import_proxy


def discard_device_payload(payload,device=0):
    """Acknowledge a result proven never dispatched to an external consumer."""
    with torch.cuda.device(device):
        for chunk in payload.get('chunks',[]):
            if 'descriptor' in chunk:import_proxy(chunk['descriptor']).acknowledge_consumption()


def host_frames(payload,rid,receiver,frame_bytes=4194304):
    layers=payload['layers'];hidden=payload['hidden_dim'];seq=0
    total=sum(len(c['positions']) for c in payload['chunks']);offset=0
    pending=list(payload['chunks'])
    try:
        for chunk in list(pending):
            positions=chunk['positions']
            if 'descriptor' in chunk:
                tensor,proxy,_=receiver.receive(chunk['descriptor'])
                try:values=tensor.cpu()
                finally:
                    if proxy:
                        with torch.cuda.device(receiver.index):proxy.release_without_reconstruction()
            else:values=torch.frombuffer(bytearray(chunk['host_bytes']),dtype=torch.bfloat16).reshape(len(positions),len(layers),hidden)
            pending.pop(0)
            rows=max(1,frame_bytes//(len(layers)*hidden*2))
            for start in range(0,len(positions),rows):
                selected=positions[start:start+rows];n=len(selected);offset+=n
                yield dict(schema='fyh.feature_batch.v1',batch_seq=seq,rids=[rid],
                    records=[dict(rid=rid,row_start=0,row_count=n,positions=selected,complete=offset==total)],
                    layer_ids=layers,shape=[n,len(layers),hidden],dtype='bfloat16',
                    hidden_bytes=values[start:start+n].contiguous().view(torch.uint8).numpy().tobytes())
                seq+=1
    finally:
        receiver.activate_context()
        discard_device_payload(dict(chunks=pending),receiver.index)
