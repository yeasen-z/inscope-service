"""Explicit, versioned runtime wiring. Checkpoint identities are immutable."""
import hashlib
import json
import math
from pathlib import Path


def load_config(path):
    config=json.loads(Path(path).read_text())
    if config.get('schema')!='inscope.detection_pipeline.v1':
        raise ValueError('Unsupported detection pipeline config schema')
    if not isinstance(config.get('composer_id'),str) or not config['composer_id']:
        raise ValueError('Explicit composer identity required')
    bank=config['bank']; entries=config['detectors']
    kind=bank.get('kind','token_probe')
    if kind not in ('token_probe','asymmetric_conv'):
        raise ValueError('Unsupported detector bank kind')
    if not entries or len(entries)!=len(bank['checkpoints']):
        raise ValueError('One detector entry per checkpoint required')
    layers=bank['layers']
    if not layers or layers!=sorted(set(layers)) or any(type(x) is not int or x<0 for x in layers):
        raise ValueError('Native boundary layers must be unique and increasing')
    if (sorted({j['layer'] for j in bank['checkpoints']})!=layers or
        [e['layer'] for e in entries]!=[j['layer'] for j in bank['checkpoints']]):
        raise ValueError('Detector entries/checkpoints must match capture layer order')
    if len({e['name'] for e in entries})!=len(entries): raise ValueError('Duplicate detector names')
    for entry,job in zip(entries,bank['checkpoints']):
        if not entry['name'] or not math.isfinite(entry['logit_threshold']):
            raise ValueError('Named detectors with frozen finite logit thresholds required')
        if kind=='asymmetric_conv':
            if not job.get('training_config') or not math.isfinite(entry.get('token_logit_threshold',float('nan'))):
                raise ValueError('Convolution requires training YAML and a frozen token threshold')
        with Path(job['checkpoint']).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=job['checkpoint_sha256']:
                raise ValueError('Checkpoint SHA256 mismatch')
    native=config['capture']
    if not native['endpoints'] or not native['model_path']:
        raise ValueError('Explicit native endpoints/model identity required')
    return config


def build_pipeline(config,device='cuda'):
    from ..token_probe import AllPositionProbeBank
    from .composer import CompiledComposer
    from .native_backend import NativeCaptureBackend
    from .planner import AllPositionPlanner
    from .postprocess import CalibratedPostProcessor
    from .scheduler import InferenceScheduler
    from .pipeline import DetectionPipeline
    if config['bank'].get('kind')=='asymmetric_conv':
        from .conv_bank import DenseConvProbeBank
        bank=DenseConvProbeBank(config['bank'],device)
    else:
        bank=AllPositionProbeBank(config['bank'],device)
    native=dict(config['capture'])
    if config.get('transport')=='cuda_ipc':native['gpu_device']=bank.device
    backend=NativeCaptureBackend(hidden_dim=bank.dim,layers=bank.layers,**native)
    scheduler=InferenceScheduler(bank,**config.get('inference',{}))
    budget=config['max_capture_bytes']
    if config.get('transport')=='cuda_ipc':
        import torch
        available,_=torch.cuda.mem_get_info(bank.device)
        budget=min(budget,max(1,(available-4*1024**3)//3))
    return DetectionPipeline(CompiledComposer(config['composer_id'],config.get('max_tokens',40959)),
        AllPositionPlanner(),backend,scheduler,CalibratedPostProcessor(config['detectors']),
        max_capture_bytes=budget)
