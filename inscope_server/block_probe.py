"""1-head attention block 探针：ipi-doom 训练侧实现的忠实移植。

移植来源（对拍基准）:
- ``ipi-doom/src/featurize/feature_quantization.py``（E4M3 + FP16 scale 编解码）
- ``ipi-doom/src/probe_training/models.py``（AttentionPooling / BlockProbe attention 路径）
- ``ipi-doom/src/probe_training/debug/__init__.py``（ProbeScorer._score_features /
  score_features 的量化和打分顺序）

训练侧语义（不可偏离）:
- 空 anchor 参与 attention：``terms`` 只按 padding mask 屏蔽，不看 empty，
  因此 empty 行既进 softmax 也出现在 explain 通道里（训练侧不置零）。
- 服务端路径 = 引擎 BF16 特征 → 按行量化到 E4M3+scale → 解码 → mean/std → attention。
  量化往返刻意保留，复现训练存储损失。
- 阈值在 logit 空间（checkpoint 自带），对外换算 sigmoid。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

# 探针输出维度须与 SGLang 隐藏维一致。
LAYER_HIDDEN = 4096

E4_QUANTIZATION_MAX = 448.0
E4_MINIMUM_SCALE = 2. ** -24


def encode_scaled_e4m3(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ipi-doom feature_quantization.encode_scaled_e4m3 逐行移植。"""
    x = features.float()
    if not torch.isfinite(x).all():
        raise ValueError('Cannot quantize non-finite feature values')
    scales = (x.abs().amax(dim=-1) / 448.).clamp_min(2. ** -24).half()
    if not torch.isfinite(scales).all():
        raise ValueError('Feature scale exceeds the float16 storage range')
    values = (x / scales.float().unsqueeze(-1)).clamp(-448., 448.).to(torch.float8_e4m3fn).contiguous()
    return values, scales.contiguous()


def decode_scaled_e4m3(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """ipi-doom feature_quantization.decode_scaled_e4m3 逐行移植。"""
    if values.shape[:-1] != scales.shape:
        raise ValueError('Feature scale shape does not match vectors')
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError('Feature scales must be finite and positive')
    return values.float() * scales.float().unsqueeze(-1)


class AttentionPooling(nn.Module):
    """ipi-doom probe_training.models.AttentionPooling 逐行移植。

    Each head has one learned query, value projection and position slope. The
    scalar head outputs are combined after masked attention pooling.
    """

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim, heads))
        self.value = nn.Parameter(torch.empty(dim, heads))
        self.position = nn.Parameter(torch.zeros(heads))
        self.head_output = nn.Parameter(torch.empty(heads))
        self.bias = nn.Parameter(torch.zeros(()))

    def terms(self, values: torch.Tensor, mask: torch.Tensor):
        steps = torch.arange(values.shape[1], device=values.device, dtype=values.dtype)
        logits = values @ self.query + steps[None, :, None] * self.position
        logits = logits.masked_fill(~mask[:, :, None], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        projected = values @ self.value
        return weights, projected

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        weights, projected = self.terms(values, mask)
        pooled = (weights * projected).sum(1)
        return pooled @ self.head_output + self.bias


@dataclass
class BlockProbeHandle:
    """observation_blocks 布局的 attention 探针（对应 manifest schema v3）。

    与 Probe（fixed_positions 线性探针）不同的部分全部显式放在这里；
    scoring 通过 isinstance 分支，绝不复用线性路径的 flatten/score。
    """

    path: Path
    spec: Any  # ProbeSpec（selected_positions=()，threshold 为概率空间）
    module: AttentionPooling
    mean: torch.Tensor
    std: torch.Tensor
    threshold_logit: float
    sha256: str = ""

    feature_layout: str = "observation_blocks"

    @property
    def threshold(self) -> float:
        return float(torch.sigmoid(torch.tensor(self.threshold_logit)).item())

    @property
    def input_dim(self) -> int:
        """与线性探针同名：供 /info、自检与日志统一读取。"""
        return int(self.spec.input_dim)

    @property
    def capture_layers(self) -> Sequence[int]:
        return self.spec.capture_layers

    @property
    def selected_positions(self) -> Sequence[int]:
        return ()

    def score_blocks(self, rows: torch.Tensor, blocks: Sequence[dict], *,
                     explain: bool = False) -> dict[str, Any]:
        """rows: [n_slots, len(capture_layers)*hidden]，按 capture_layers 顺序拼接。

        逐行复刻 ProbeScorer._score_features（量化往返）+ score_features
        （mean/std、attention forward、local_scores 通道）。blocks 为编译产物。
        """
        if rows.dim() != 2 or rows.shape[-1] != len(self.capture_layers) * LAYER_HIDDEN:
            raise ValueError("Block feature rows have shape %s" % (tuple(rows.shape),))
        if not torch.isfinite(rows).all():
            raise ValueError("Feature tensor contains non-finite values")
        # 训练侧按 [layer, position, hidden] 逐 (layer, position) 求 absmax
        # （plan.json: per-layer/per-position float16 scale）。这里必须先还原层
        # 维度再量化，否则会退化成「每个 capture slot 一个 scale」，数值对不上。
        layered = rows.reshape(rows.shape[0], len(self.capture_layers), LAYER_HIDDEN)
        packed, scales = encode_scaled_e4m3(layered)
        decoded = (packed.float() * scales.float().unsqueeze(-1)).reshape(rows.shape[0], -1)
        slots = [int(block['capture_slot']) for block in blocks]
        if any(slot >= decoded.shape[0] for slot in slots):
            raise ValueError("Block capture_slot exceeds captured rows")
        # ipi-doom: values.append(decoded.transpose(0, 1).flatten(1)[slots])
        # 其 decoded 是 [positions, layers, hidden]，flatten(1) 后按层拼接；
        # 这里 rows 已经按 capture_layers 顺序拼接，直接取行即等价。
        values = decoded[slots]
        with torch.inference_mode():
            normalized = (values - self.mean) / self.std
            if not torch.isfinite(normalized).all():
                raise ValueError("Normalized features contain non-finite values")
            mask = torch.ones(1, values.shape[0], dtype=torch.bool)
            weights, evidence = self.module.terms(normalized.unsqueeze(0), mask)
            pooled = (weights * evidence).sum(1)
            output = pooled @ self.module.head_output + self.module.bias
            if not torch.isfinite(output).all():
                raise ValueError("Probe output contains non-finite values")
            logit = float(output.reshape(-1)[0].item())
            probability = float(torch.sigmoid(torch.tensor(logit)).item())
            if not (math.isfinite(logit) and math.isfinite(probability)):
                raise ValueError("Probe score contains non-finite values")
            result: dict[str, Any] = {"logit": logit, "probability": probability}
            if explain:
                # 复刻 BlockProbe.local_scores 的 attention 分支：
                #   attention_weight      = weights.mean(2)
                #   attention_contribution = (weights * evidence * head_output).sum(2)
                # empty anchor 既参与 softmax 也出现在通道里，训练侧不置零；
                # 任何置零都会让 explain 与训练期数值对不上。
                attention_weight = weights.mean(2)
                attention_contribution = (weights * evidence * self.module.head_output).sum(2)
                result["blocks"] = [dict(block,
                                         attention_weight=float(attention_weight[0, i]),
                                         attention_contribution=float(attention_contribution[0, i]))
                                    for i, block in enumerate(blocks)]
        return result


def load_block_probe(path: Path, manifest: dict, probe_id: str) -> BlockProbeHandle:
    """从 best.pt 风格的 block checkpoint + v3 manifest 构造探针。"""
    from .probe import ProbeSpec

    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    declared = manifest.get("sha256")
    if declared and declared != digest:
        raise ValueError("Checkpoint sha256 mismatch: manifest=%s actual=%s" % (declared, digest))

    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict):
        raise ValueError("Unsupported checkpoint layout: not a dict")
    if ck.get("method") != "attention":
        raise ValueError("Block checkpoint declares unsupported method: %r" % (ck.get("method"),))

    dim = int(ck.get("dim") or 0)
    if dim <= 0 or dim % LAYER_HIDDEN != 0:
        raise ValueError("Block checkpoint declares no usable dim: %r" % (ck.get("dim"),))
    heads = int(ck.get("heads") or 0)
    if heads < 1:
        raise ValueError("Block checkpoint declares no usable heads")

    layers = tuple(int(layer) for layer in manifest["inscope_layers"])
    if dim != len(layers) * LAYER_HIDDEN:
        raise ValueError(
            "dim %d != layers(%d) x hidden(%d)" % (dim, len(layers), LAYER_HIDDEN))

    mean = ck["mean"].float().reshape(-1)
    std = ck["std"].float().reshape(-1)
    if mean.numel() != dim or std.numel() != dim:
        raise ValueError("mean/std must be [%d]" % dim)
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all()) or not (std > 0).all():
        raise ValueError("Invalid normalization tensors")

    state = ck.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Block checkpoint has no state_dict")
    module = AttentionPooling(dim, heads)
    expected = {
        "aggregate.query": (dim, heads),
        "aggregate.value": (dim, heads),
        "aggregate.position": (heads,),
        "aggregate.head_output": (heads,),
        "aggregate.bias": (),
    }
    for name, shape in expected.items():
        if name not in state:
            raise ValueError("Checkpoint missing tensor %r" % name)
        value = state[name]
        if tuple(value.shape) != shape:
            raise ValueError("Tensor %r shape %s != %s" % (name, tuple(value.shape), shape))
        if not torch.isfinite(value.float()).all():
            raise ValueError("Tensor %r has non-finite values" % name)
    module.query.data = state["aggregate.query"].float()
    module.value.data = state["aggregate.value"].float()
    module.position.data = state["aggregate.position"].float()
    module.head_output.data = state["aggregate.head_output"].float()
    module.bias.data = state["aggregate.bias"].float()
    module = module.eval()

    threshold_logit = ck.get("threshold")
    try:
        threshold_logit = float(threshold_logit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Block checkpoint threshold must be a finite logit") from exc
    if not math.isfinite(threshold_logit):
        raise ValueError("Block checkpoint threshold must be a finite logit")
    if manifest.get("threshold_space", "logit") != "logit":
        raise ValueError("Unsupported threshold_space: %r" % (manifest.get("threshold_space"),))
    if manifest.get("empty_anchor") != "attend":
        raise ValueError("Unsupported empty_anchor policy: %r (only 'attend' is valid)" % (manifest.get("empty_anchor"),))

    spec = ProbeSpec(
        path=str(path),
        schema=manifest["schema"],
        probe_id=probe_id,
        probe_architecture="attention_block",
        input_dim=dim,
        block_indices=(),
        capture_layers=layers,
        selected_positions=(),
        feature_normalization="standard",
        flatten_order=tuple(manifest.get("flatten_order") or ("layer", "hidden")),
        threshold=float(torch.sigmoid(torch.tensor(threshold_logit)).item()),
        labeling_protocol=ck.get("labeling_protocol"),
        probe_prefix="",
        layer_semantics=manifest.get("layer_semantics", "native_residual_stream_boundary"),
        feature_name=manifest.get("feature_name"),
    )
    return BlockProbeHandle(
        path=path,
        spec=spec,
        module=module,
        mean=mean,
        std=std,
        threshold_logit=threshold_logit,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
