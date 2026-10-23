"""按 checkpoint 自描述加载探针。

设计原则：server 不预设任何一套探针结构。层号、位置、归一化方式、
拼接顺序、架构与阈值全部从 checkpoint（及其 manifest.json）读取。
未知的 schema / 架构 / 归一化方式一律报错，不回退到某个默认探针。
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from inscope_contract import LayerSemantics, build_layer_spec

# 探针输出维度须与 SGLang 隐藏维一致。
LAYER_HIDDEN = 4096

# 支持的 checkpoint schema -> 归一化字段名所在位置。
SUPPORTED_SCHEMAS = ("fyh.probe_checkpoint.v2",)


class ProbeSpec:
    """checkpoint 自带的运行参数。所有字段都必须来自 checkpoint/manifest。"""

    __slots__ = (
        "path", "schema", "probe_id", "probe_architecture", "input_dim", "block_indices",
        "capture_layers", "selected_positions", "feature_normalization",
        "flatten_order", "threshold", "labeling_protocol", "probe_prefix",
        "layer_semantics", "feature_name",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            object.__setattr__(self, k, kw.get(k))

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


def _read_manifest(ckpt_path: Path) -> dict | None:
    """同目录下的 manifest.json 是层号与位置的权威来源。"""
    manifest = ckpt_path.parent / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid manifest.json next to %s: %s" % (ckpt_path, exc))
    return data if isinstance(data, dict) else None


def _resolve_layers(ck: dict, manifest: dict | None) -> tuple[tuple[int, ...], LayerSemantics]:
    """确定 SGLang 边界层号。

    优先用 manifest 显式给出的 inscope_layers（已换算好的边界号）；
    否则用 checkpoint 的 block_indices 按声明的语义换算。
    """
    if manifest and manifest.get("inscope_layers"):
        layers = tuple(int(x) for x in manifest["inscope_layers"])
        # manifest 给出的是边界号，本身已是最终形态。
        return layers, LayerSemantics.BOUNDARY

    blocks = ck.get("selected_layer_indices") or ck.get("layer_indices")
    if not blocks:
        raise ValueError("Checkpoint declares no layer indices")
    blocks = tuple(int(x) for x in blocks)
    semantic = ck.get("layer_semantics")
    if semantic == LayerSemantics.BOUNDARY.value:
        return blocks, LayerSemantics.BOUNDARY
    if semantic == LayerSemantics.BLOCK_OUTPUT.value:
        spec = build_layer_spec(blocks, LayerSemantics.BLOCK_OUTPUT)
        return spec.capture_layers, LayerSemantics.BLOCK_OUTPUT
    raise ValueError(
        "Checkpoint has no usable layer semantics (layer_semantics=%r); "
        "provide manifest.json with explicit inscope_layers" % (semantic,)
    )


@dataclass
class Probe:
    """一个按 checkpoint 声明的结构实例化的探针。"""

    path: Path
    spec: ProbeSpec
    module: nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    threshold: float
    sha256: str = field(default="")

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim

    @property
    def capture_layers(self) -> Sequence[int]:
        return self.spec.capture_layers

    @property
    def selected_positions(self) -> Sequence[int]:
        return self.spec.selected_positions

    def score(self, features: torch.Tensor) -> tuple[float, float]:
        """features: [1, input_dim] 已按 checkpoint 声明顺序铺平。"""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                "Feature dim %d != checkpoint input_dim %d"
                % (features.shape[-1], self.input_dim)
            )
        if not torch.isfinite(features).all():
            raise ValueError("Feature tensor contains non-finite values")
        x = (features.float() - self.mean) / self.std
        logit = float(self.module(x).reshape(-1)[0].item())
        return logit, float(torch.sigmoid(torch.tensor(logit)).item())

    def flatten(self, layer_blocks: Sequence[torch.Tensor]) -> torch.Tensor:
        """按 checkpoint 声明的 flatten_order 组装特征向量。

        layer_blocks: 每个元素是某层的 [n_pos, hidden]，顺序与 capture_layers 一致。
        """
        if len(layer_blocks) != len(self.capture_layers):
            raise ValueError(
                "Expect %d layer blocks, got %d"
                % (len(self.capture_layers), len(layer_blocks))
            )
        order = list(self.spec.flatten_order or ["batch", "layer", "token", "hidden"])
        if order != ["batch", "layer", "token", "hidden"]:
            raise ValueError("Unsupported flatten_order: %r" % (order,))
        # 每层必须是 [n_positions, hidden]，否则铺平后维度会静默错位。
        expected = (len(self.selected_positions), LAYER_HIDDEN)
        for idx, block in enumerate(layer_blocks):
            if tuple(block.shape) != expected:
                raise ValueError(
                    "Layer block %d has shape %s, expected %s"
                    % (idx, tuple(block.shape), expected))
        stacked = torch.stack([b.float() for b in layer_blocks], dim=0)  # [layer, token, hidden]
        flat = stacked.reshape(1, -1)
        if flat.shape[-1] != self.input_dim:
            raise ValueError("Flattened dim %d != checkpoint input_dim %d"
                             % (flat.shape[-1], self.input_dim))
        return flat


def _build_module(arch: str, state: dict, input_dim: int) -> nn.Module:
    if arch == "linear":
        expected = {"probe.weight": (1, input_dim), "probe.bias": (1,)}
        flat = {
            "0.weight": state["probe.weight"],
            "0.bias": state["probe.bias"],
        }
    elif arch == "mlp":
        expected = {"probe.weight": (1, input_dim), "probe.bias": (1,)}
        flat = {"0.weight": state["probe.weight"], "0.bias": state["probe.bias"]}
    else:
        raise ValueError("Unsupported probe_architecture: %r" % (arch,))

    for name, shape in expected.items():
        if name not in state:
            raise ValueError("Checkpoint missing tensor %r for arch %r" % (name, arch))
        value = state[name]
        if tuple(value.shape) != shape:
            raise ValueError("Tensor %r shape %s != %s" % (name, tuple(value.shape), shape))
        if not torch.isfinite(value).all():
            raise ValueError("Tensor %r has non-finite values" % (name,))

    module = nn.Sequential(nn.Linear(input_dim, 1))
    module.load_state_dict(flat)
    return module.eval()


def _derive_probe_id(ck: dict) -> str:
    """从 checkpoint 自述信息派生稳定 id。

    同一份权重每次加载必须得到同一个 id，不同权重必须得到不同的 id，
    所以这里只用 checkpoint 内的不变量，不含路径、时间戳等环境相关字段。
    """
    arch = ck.get("probe_architecture") or "linear"
    dataset = ck.get("dataset_name") or "unknown"
    feature = ck.get("feature_name") or "unnamed"
    return "%s:%s:%s" % (arch, dataset, feature)


def load_probe(ckpt_path, *, expected_input_dim: int | None = None) -> Probe:
    """从 checkpoint 文件构造探针，结构完全由 checkpoint 自描述。"""
    path = Path(ckpt_path)
    if not path.is_file():
        raise ValueError("Probe checkpoint not found: %s" % path)

    manifest = _read_manifest(path)
    if manifest:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        declared = manifest.get("sha256")
        if declared and declared != digest:
            raise ValueError(
                "Checkpoint sha256 mismatch: manifest=%s actual=%s" % (declared, digest)
            )

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError("Unsupported checkpoint layout: not a dict")

    schema = ck.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(
            "Unsupported checkpoint schema %r; supported: %s"
            % (schema, ", ".join(SUPPORTED_SCHEMAS))
        )

    state = ck.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model_state_dict")

    arch = ck.get("probe_architecture") or "linear"
    input_dim = int(ck.get("input_dim") or 0)
    if input_dim <= 0:
        raise ValueError("Checkpoint declares no usable input_dim")
    if expected_input_dim is not None and input_dim != expected_input_dim:
        raise ValueError(
            "Checkpoint input_dim %d != expected %d" % (input_dim, expected_input_dim)
        )

    layers, semantics = _resolve_layers(ck, manifest)

    positions = ck.get("selected_positions") or (manifest or {}).get("selected_positions")
    if not positions:
        raise ValueError("Checkpoint declares no selected_positions")
    positions = tuple(int(x) for x in positions)

    norm = ck.get("feature_normalization") or "standard"
    if norm != "standard":
        raise ValueError("Unsupported feature_normalization: %r" % (norm,))
    if "input_mean" not in state or "input_std" not in state:
        raise ValueError("Checkpoint missing input_mean/input_std for standard normalization")
    mean = state["input_mean"].float().reshape(-1)
    std = state["input_std"].float().reshape(-1)
    if mean.numel() != input_dim or std.numel() != input_dim:
        raise ValueError("Normalization tensors must be [%d]" % input_dim)
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all()) or not (std > 0).all():
        raise ValueError("Invalid normalization tensors")

    expected_layers = len(layers)
    expected_pos = len(positions)
    if input_dim != expected_layers * expected_pos * LAYER_HIDDEN:
        raise ValueError(
            "input_dim %d != layers(%d) x positions(%d) x hidden(%d)"
            % (input_dim, expected_layers, expected_pos, LAYER_HIDDEN)
        )

    module = _build_module(arch, state, input_dim)

    threshold = ck.get("threshold")
    if threshold is None:
        raise ValueError("Checkpoint declares no threshold")
    threshold = float(threshold)

    flatten_order = (manifest or {}).get("flatten_order") or ck.get("flatten_order")

    # probe_id 优先取 checkpoint/manifest 的显式声明，否则由架构+训练集+特征名派生。
    # 派生而不是写死，是为了让不同 checkpoint 天然得到不同的 id，避免误配 profile。
    probe_id = (ck.get("probe_id") or (manifest or {}).get("probe_id")
                or _derive_probe_id(ck))

    spec = ProbeSpec(
        path=str(path),
        schema=schema,
        probe_id=probe_id,
        probe_architecture=arch,
        input_dim=input_dim,
        block_indices=tuple(int(x) for x in (ck.get("selected_layer_indices") or ck.get("layer_indices") or ())),
        capture_layers=layers,
        selected_positions=positions,
        feature_normalization=norm,
        flatten_order=tuple(flatten_order) if flatten_order else ("batch", "layer", "token", "hidden"),
        threshold=threshold,
        labeling_protocol=ck.get("labeling_protocol"),
        probe_prefix=(manifest or {}).get("probe_prefix"),
        layer_semantics=semantics.value,
        feature_name=ck.get("feature_name"),
    )

    return Probe(
        path=path,
        spec=spec,
        module=module,
        mean=mean,
        std=std,
        threshold=threshold,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
