"""Resident all-position Transformer probes, independent of the training repo.

The plan pins checkpoint SHA256s and native residual-boundary layers. Inputs are
BF16 [positions, layers, hidden]. FP16-scale E4M3 roundtrip and train-only
normalization preserve the training codec, including each rounding boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class TokenLocator(nn.Module):
    def __init__(self, dim, width, heads, depth):
        super().__init__()
        self.project = nn.Linear(dim, width)
        # Retained checkpoint parameter; coordinates are explicitly disabled.
        self.coordinates = nn.Linear(1, width, bias=False)
        self.context = nn.ModuleList([nn.TransformerEncoderLayer(
            width, heads, width*4, dropout=0., activation='gelu',
            batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.decoder = nn.Sequential(nn.Linear(width, 1))

    def forward(self, values, cu, maximum):
        from torch.nn.attention.varlen import varlen_attn
        h = self.project(values)
        for layer in self.context:
            norm = layer.norm1(h)
            heads = layer.self_attn.num_heads
            qkv = F.linear(norm, layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias).reshape(-1, 3, heads, norm.shape[-1]//heads)
            out = varlen_attn(qkv[:,0], qkv[:,1], qkv[:,2], cu, cu,
                maximum, maximum, window_size=(-1,-1))
            h = h + F.linear(out.flatten(1), layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias)
            norm = layer.norm2(h)
            h = h + layer.linear2(layer.activation(layer.linear1(norm)))
        return self.decoder(self.norm(h)).squeeze(-1).float()


class TokenDetector(nn.Module):
    def __init__(self, dim, width, heads, depth):
        super().__init__()
        self.locator = TokenLocator(dim, width, heads, depth)
        self.detector = nn.Sequential(nn.Linear(2*dim, 1))


def validate_positions(positions):
    if not positions:
        raise ValueError('A nonempty batch is required')
    for p in positions:
        if len(p) < 2 or any(type(x) is not int or x < 0 for x in p):
            raise ValueError('Each DP requires pre plus at least one body position')
        if any(b != a+1 for a,b in zip(p,p[1:])):
            raise ValueError('All-position requests must be contiguous and increasing')
    return [len(p) for p in positions]


class AllPositionProbeBank:
    """One resident bank and one stream per GPU worker; callers may share it.

    A lock covers staging through result transfer: shared pinned buffers cannot
    be overwritten by concurrent HTTP requests while an H2D copy is in flight.
    """
    def __init__(self, plan, device='cuda'):
        if isinstance(plan, (str, Path)):
            plan = json.loads(Path(plan).read_text())
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('All-position production inference requires CUDA')
        self.layers = list(plan['layers'])
        if not self.layers or any(type(x) is not int or x < 0 for x in self.layers) or sorted(set(self.layers)) != self.layers:
            raise ValueError('Layers must be unique increasing native boundary IDs')
        if sorted({j['layer'] for j in plan['checkpoints']}) != self.layers:
            raise ValueError('Checkpoint layers differ from capture layer union')
        self.layer_slots = torch.tensor([self.layers.index(j['layer']) for j in plan['checkpoints']],
            device=self.device,dtype=torch.int32)
        self.models = []; self.identities = []; means = []; stds = []
        for job in plan['checkpoints']:
            path = Path(job['checkpoint'])
            with path.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            if digest != job['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint changed')
            state = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
            c = state['contract']; v = c['variant']; r = c['recipe']
            required = dict(feature_fusion='probe_mlp', input_mapping='linear',
                readout_mapping='linear', global_mapping='linear', readout_mode='self',
                encoder_position_mode='none', decoder_position_mode='none')
            if v['feature_layers'] != [job['layer']] or any(v.get(k) != x for k,x in required.items()):
                raise ValueError('Unsupported all-position probe contract')
            if v.get('architecture') or v.get('observation_only') or v.get('region_pooling','logmeanexp') != 'logmeanexp':
                raise ValueError('Unsupported probe architecture/pooling')
            stats = state['normalization']; mean = stats['mean']; std = stats['std']
            if (mean.ndim != 1 or std.shape != mean.shape or not torch.isfinite(mean).all()
                    or not torch.isfinite(std).all() or not (std > 0).all()):
                raise ValueError('Invalid train-only normalization')
            dim = mean.numel()
            if means and dim != means[0].numel():
                raise ValueError('Probe dimensions differ')
            model = TokenDetector(dim, r['width'], r['heads'], v.get('depth',1))
            model.load_state_dict(state['model'], strict=True)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Non-finite checkpoint parameter')
            self.models.append(model.to(self.device).eval().requires_grad_(False))
            means.append(mean); stds.append(std); self.identities.append(digest)
        self.mean = torch.stack(means).to(self.device).float().contiguous()
        self.std = torch.stack(stds).to(self.device).float().contiguous()
        self.dim = self.mean.shape[-1]
        self.host = None; self.lock = threading.Lock()

    def _stage(self, batch):
        sizes = validate_positions([r['region_positions'] for _,r,_ in batch])
        count = sum(sizes); shape = (count, len(self.layers), self.dim)
        for _,_,features in batch:
            if getattr(features,'ready_event',None) is not None:
                torch.cuda.current_stream(self.device).wait_event(features.ready_event)
        if all(v[0].is_cuda for _,_,v in batch):
            tensors=[v[0] for _,_,v in batch]
            if any(t.device!=self.mean.device or t.dtype!=torch.bfloat16 or t.shape!=(n,len(self.layers)*self.dim)
                   for t,n in zip(tensors,sizes)):
                raise ValueError('Device feature shape/placement mismatch')
            joined=tensors[0] if len(tensors)==1 else torch.cat(tensors)
            return joined.reshape(shape),sizes
        if self.host is None or len(self.host) < count:
            self.host = torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)
        host = self.host[:count]; offset = 0
        for (_,row,values),n in zip(batch,sizes):
            if values[0].shape != (n, shape[1]*shape[2]) or values[0].dtype != torch.bfloat16 or values[0].device.type != 'cpu':
                raise ValueError('Expected CPU BF16 region features in plan layer order')
            host[offset:offset+n].copy_(values[0].reshape(n,shape[1],shape[2])); offset += n
        return host.to(self.device, non_blocking=True), sizes

    def _codec(self, region, normalize=True):
        from .token_probe_kernel import codec_normalize
        import triton
        n,m,d = region.shape
        if (m,d) != (len(self.layers),self.dim) or region.dtype != torch.bfloat16 or not region.is_contiguous():
            raise ValueError('Expected contiguous BF16 [positions, layers, hidden]')
        if region.device != self.mean.device:
            raise ValueError('Feature device mismatch')
        output = torch.empty((len(self.models),n,d), device=self.device, dtype=torch.bfloat16)
        valid = torch.empty((n,len(self.models)), device=self.device, dtype=torch.bool)
        codec_normalize[(n,len(self.models))](region,self.mean,self.std,self.layer_slots,output,valid,d,m,
            BLOCK=triton.next_power_of_2(d),NORMALIZE=normalize,enable_fp_fusion=False)
        return output, valid.all()

    @torch.inference_mode()
    def _forward(self, region, sizes, anchors=None):
        normalized,valid = self._codec(region)
        offsets = [0]
        for n in sizes: offsets.append(offsets[-1]+n)
        cu = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        outputs = []; global_scores = []
        if anchors is not None:
            if anchors.shape != (len(sizes),2,len(self.layers),self.dim):
                raise ValueError('Expected anchors [DPs, post/suffix, layers, hidden]')
            decoded,anchor_valid = self._codec(anchors.flatten(0,1).contiguous(),False)
            valid = valid & anchor_valid
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for slot,model in enumerate(self.models):
                outputs.append(model.locator(normalized[slot],cu,max(sizes)))
                if anchors is not None:
                    av = decoded[slot].reshape(len(sizes),2*self.dim).float()
                    global_scores.append(model.detector(av).squeeze(-1).float())
        tokens = torch.stack(outputs,1)
        if anchors is None: return tokens, valid
        # Same FP32 stable scatter LME as training, excluding the pre position.
        segment = torch.repeat_interleave(torch.arange(len(sizes),device=self.device),torch.tensor(sizes,device=self.device),output_size=sum(sizes))
        scores = tokens.clone(); scores[cu[:-1].long()] = -torch.inf
        maxima = scores.new_full((len(sizes),len(self.models)),-torch.inf)
        index = segment[:,None].expand_as(scores)
        maxima.scatter_reduce_(0,index,scores,reduce='amax',include_self=True)
        sums = scores.new_zeros(maxima.shape).scatter_add_(0,index,(scores-maxima[segment]).exp())
        region_scores = maxima+sums.log()-torch.tensor([n-1 for n in sizes],device=self.device,dtype=torch.float32).log()[:,None]
        detector = torch.stack(global_scores,1)
        return (tokens,region_scores,detector,(region_scores+detector)*.5),valid

    def warmup(self):
        """Compile both codec paths and warm varlen kernels before readiness."""
        for n in (16,256):
            region=torch.zeros(n,len(self.layers)*self.dim,dtype=torch.bfloat16)
            anchors=torch.zeros(2,len(self.layers)*self.dim,dtype=torch.bfloat16)
            self.predict([(0,dict(region_positions=list(range(n))),(region,anchors))])

    def compute(self,batch):
        """Locator-only tensor output, compatible with the collection consumer."""
        with self.lock, torch.inference_mode(), torch.cuda.device(self.device):
            region,sizes = self._stage(batch)
            logits,valid = self._forward(region,sizes)
            packed = torch.cat((logits.flatten(),valid.float().reshape(1))).cpu()
            if packed[-1] != 1 or not torch.isfinite(packed[:-1]).all():
                raise ValueError('Non-finite feature, FP16-scale overflow or output')
            return packed[:-1].reshape(-1,len(self.models))

    def predict(self,batch):
        """Token logits and DP region/global/ensemble logits; one D2H transfer."""
        with self.lock, torch.inference_mode(), torch.cuda.device(self.device):
            region,sizes = self._stage(batch)
            anchors = torch.stack([v[1].reshape(2,len(self.layers),self.dim) for _,_,v in batch]).to(self.device)
            values,valid = self._forward(region,sizes,anchors)
            present = [v for v in values if v is not None]
            packed = torch.cat([*(v.flatten() for v in present),valid.float().reshape(1)]).cpu()
            if packed[-1] != 1 or not torch.isfinite(packed[:-1]).all():
                raise ValueError('Non-finite feature, FP16-scale overflow or output')
            chunks = iter(packed[:-1].split([v.numel() for v in present]))
            return {key: None if value is None else next(chunks).reshape(value.shape)
                    for key, value in zip(('token_logits','region_logits','detector_logits','logits'), values)}
