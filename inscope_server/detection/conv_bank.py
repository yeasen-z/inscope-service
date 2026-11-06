"""Serving for dense, observation-only asymmetric convolution checkpoints."""
import hashlib
from pathlib import Path
import threading

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from ..token_probe import AllPositionProbeBank


class DenseConvLocator(nn.Module):
    def __init__(self, dim, model):
        super().__init__()
        self.left = model.get('left_radius', 4)
        self.right = model.get('right_radius', 8)
        slots = self.left + self.right + 1
        width = model.get('projection_width', 256)
        channels = model.get('channels', 128)
        keep_anchor = model.get('keep_anchor', True)
        residual = model.get('residual_second', False)
        self.project = nn.Identity() if width is None else nn.Sequential(nn.Linear(dim, width), nn.GELU())
        width = dim if width is None else width
        # Bound the gathered temporary even for raw-width or wide-window recipes.
        self.chunk = max(1, min(2048, (8 * 1024**2) // (slots * width)))
        self.anchor = nn.Linear(width, channels) if keep_anchor else None
        self.window = nn.Linear(slots * width, channels)
        hidden = channels if residual else 128
        self.decoder = nn.Sequential(nn.Linear((3 if keep_anchor else 1) * channels + slots, hidden),
                                     nn.GELU(), nn.Linear(hidden, 1))
        self.second_window = nn.Linear(slots * channels, channels) if residual else None

    def forward(self, values):
        # One contiguous pre/body sequence. Index zero is the pre position.
        projected = self.project(values)
        indices = torch.arange(len(values), device=values.device)
        offsets = torch.arange(-self.left, self.right + 1, device=values.device)

        def context(source, targets, window, pre_center=False):
            neighbors = targets[:, None] + offsets
            valid = (neighbors > 0) & (neighbors < len(values)) & (targets[:, None] > 0)
            if pre_center:
                valid = valid | ((targets[:, None] == 0) & (offsets == 0))
            slots = source[neighbors.clamp(0, len(values) - 1)].masked_fill(~valid[..., None], 0)
            features = F.gelu(window(slots.flatten(1))).masked_fill(~valid.any(1, keepdim=True), 0)
            return features, valid.to(features.dtype)

        chunks = []
        for targets in indices.split(self.chunk):
            local, mask = context(projected, targets, self.window, self.anchor is None)
            if self.anchor is None:
                fused = torch.cat((local, mask), dim=1)
            else:
                anchor = self.anchor(projected[targets])
                fused = torch.cat((anchor, local, anchor * local, mask), dim=1)
            chunks.append(self.decoder(fused).squeeze(-1).float() if self.second_window is None
                          else self.decoder[1](self.decoder[0](fused)))
        if self.second_window is None:
            return torch.cat(chunks)

        hidden = torch.cat(chunks)
        logits = []
        for targets in indices.split(2048):
            update, _ = context(hidden, targets, self.second_window)
            logits.append(self.decoder[2](hidden[targets] + update).squeeze(-1).float())
        return torch.cat(logits)


def load_locator(job):
    checkpoint = Path(job['checkpoint'])
    with checkpoint.open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != job['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    config_bytes = Path(job['training_config']).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != state.get('config_sha256'):
        raise ValueError('Training YAML does not match checkpoint')
    config = yaml.safe_load(config_bytes)
    model = config['model']
    if (config['schema'] != 'inscope.detector-training.v1' or config['data']['stride'] != 1
            or model['type'] != 'asymmetric_conv' or model['feature_layer'] != job['layer']
            or model['centering'] != 'none' or model['readout'] != 'mlp'
            or model['pooling']['type'] != 'max'):
        raise ValueError('Expected a dense, uncentered asymmetric convolution with body-max pooling')
    data = config['data']
    statistics = Path(job.get('statistics', data['statistics']))
    with statistics.open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != data['statistics_sha256']:
            raise ValueError('Frozen normalization statistics changed')
    stats = torch.load(statistics, weights_only=True, map_location='cpu')
    layers = list(stats.get('layer_ids', [job['layer']]))
    mean, std = stats['mean'].float(), stats['std'].float()
    if (mean.ndim != 1 or std.shape != mean.shape or not layers or mean.numel() % len(layers)
            or layers.count(job['layer']) != 1):
        raise ValueError('Invalid normalization layer layout')
    slot = layers.index(job['layer'])
    mean, std = (tensor.chunk(len(layers))[slot] for tensor in (mean, std))
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError('Invalid frozen normalization statistics')
    dim = mean.numel()
    locator = nn.Module()
    locator.locator = DenseConvLocator(dim, model)
    locator.load_state_dict(state['model'], strict=True)
    if any(not torch.isfinite(p).all() for p in locator.parameters()):
        raise ValueError('Non-finite checkpoint parameter')
    return locator.locator.eval().requires_grad_(False), mean, std


class DenseConvProbeBank(AllPositionProbeBank):
    """Reuse the service's staging, FP8 round trip, and transfer boundaries."""
    def __init__(self, plan, device):
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('All-position production inference requires CUDA')
        self.layers = list(plan['layers'])
        loaded = [load_locator(job) for job in plan['checkpoints']]
        self.dim = loaded[0][1].numel()
        if any(mean.numel() != self.dim for _, mean, _ in loaded):
            raise ValueError('Probe dimensions differ')
        self.models = [model.to(self.device) for model, _, _ in loaded]
        self.identities = [job['checkpoint_sha256'] for job in plan['checkpoints']]
        self.layer_slots = torch.tensor([self.layers.index(job['layer']) for job in plan['checkpoints']],
                                        device=self.device, dtype=torch.int32)
        self.mean = torch.stack([mean for _, mean, _ in loaded]).to(self.device)
        self.std = torch.stack([std for _, _, std in loaded]).to(self.device)
        self.host = None
        self.lock = threading.Lock()

    @torch.inference_mode()
    def _forward(self, region, sizes, anchors=None):
        decoded, valid = self._codec(region)
        outputs = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for model, features in zip(self.models, decoded):
                outputs.append(torch.cat([model(dp) for dp in features.split(sizes)]))
        tokens = torch.stack(outputs, dim=1)
        if anchors is None:
            return tokens, valid
        scores = torch.stack([dp[1:].amax(dim=0) for dp in tokens.split(sizes)])
        return (tokens, scores, None, scores), valid
