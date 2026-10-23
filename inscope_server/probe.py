import hashlib
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from inscope_contract import (
    BLEND_ALPHA,
    BLOCK_INDICES,
    CAPTURE_LAYERS,
    DEFAULT_SEED,
    FEATURE_SHAPE,
    INPUT_DIM,
    LAYER_HIDDEN,
    SELECTED_POSITIONS,
    THRESHOLD_LOGIT,
    LayerSemantics,
    blend_arms,
    build_layer_spec,
)


class Probe:
    """A5 blend detector：12288 维 last1 特征，Linear 与 N1024 的 logit 混合。

    两支各自 .eval()，在 logit 空间按 alpha=0.5 混合；不混合 sigmoid 分数，
    也不把三个 seed 再平均成一个未校准模型。
    """

    # 全部来自契约包，不在实现里重复字面量。
    block_indices = BLOCK_INDICES
    # 层号换算只有一个出口：契约包的 build_layer_spec。
    layer_spec = build_layer_spec(BLOCK_INDICES, LayerSemantics.BLOCK_OUTPUT)
    capture_layers = layer_spec.capture_layers
    assert capture_layers == CAPTURE_LAYERS, "契约常量与层号换算不一致"
    selected_positions = SELECTED_POSITIONS
    input_dim = INPUT_DIM
    feature_shape = FEATURE_SHAPE
    blend_alpha = BLEND_ALPHA
    hidden = LAYER_HIDDEN

    def __init__(self, models_dir, seed=DEFAULT_SEED, threshold_logit=None):
        self.models_dir = Path(models_dir)
        self.seed = int(seed)
        arms = blend_arms(self.seed)
        self.linear_path = self.models_dir / arms["linear"]
        self.mlp_path = self.models_dir / arms["mlp"]

        linear_ck = self._load(self.linear_path, "linear")
        mlp_ck = self._load(self.mlp_path, "mlp")
        self.linear_sha256 = hashlib.sha256(self.linear_path.read_bytes()).hexdigest()
        self.mlp_sha256 = hashlib.sha256(self.mlp_path.read_bytes()).hexdigest()

        self.linear = self._build_linear(linear_ck["model_state_dict"])
        self.mlp = self._build_mlp(mlp_ck["model_state_dict"])

        # 同一 cell 两支训练归一化相同；不一致就是拿错了权重。
        mean, std = linear_ck["mean"].float(), linear_ck["std"].float()
        if not torch.equal(mean, mlp_ck["mean"].float()) or not torch.equal(std, mlp_ck["std"].float()):
            raise ValueError("Blend arms disagree on training mean/std")
        if mean.shape != (self.input_dim,) or std.shape != (self.input_dim,):
            raise ValueError("Training mean/std must be [%d]" % self.input_dim)
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all()) or not (std > 0).all():
            raise ValueError("Invalid training mean/std")
        self.mean, self.std = mean, std

        default_threshold = THRESHOLD_LOGIT.get(self.seed)
        if default_threshold is None and threshold_logit is None:
            raise ValueError("No calibrated threshold for seed %d; pass threshold_logit" % self.seed)
        self.threshold_logit = float(threshold_logit if threshold_logit is not None else default_threshold)
        self.threshold = float(torch.sigmoid(torch.tensor(self.threshold_logit)).item())

    @staticmethod
    def _load(path, arm):
        if not Path(path).is_file():
            raise ValueError("Missing %s checkpoint: %s" % (arm, path))
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ck, dict) or "model_state_dict" not in ck:
            raise ValueError("Unsupported %s checkpoint layout" % arm)
        if ck.get("input_dim") != Probe.input_dim:
            raise ValueError("%s checkpoint input_dim mismatch" % arm)
        return ck

    @staticmethod
    def _check(state, expected, arm):
        if set(state) != set(expected):
            raise ValueError("Unexpected %s parameters: %s" % (arm, sorted(state)))
        for name, shape in expected.items():
            value = state[name]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError("Invalid %s tensor: %s" % (arm, name))

    def _build_linear(self, state):
        self._check(state, {"0.weight": (1, self.input_dim), "0.bias": (1,)}, "Linear")
        module = nn.Sequential(nn.Linear(self.input_dim, 1))
        module.load_state_dict(state)
        return module.eval()

    def _build_mlp(self, state):
        self._check(state, {"0.weight": (1024, self.input_dim), "0.bias": (1024,),
                            "1.weight": (1024,), "1.bias": (1024,),
                            "4.weight": (1, 1024), "4.bias": (1,)}, "N1024")
        module = nn.Sequential(nn.Linear(self.input_dim, 1024), nn.LayerNorm(1024),
                               nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 1))
        module.load_state_dict(state)
        return module.eval()

    @torch.inference_mode()
    def prepare_features(self, hidden):
        """原始在线特征 → 训练缓存口径；不是缓存特征入口。"""
        h = torch.as_tensor(hidden, dtype=torch.float32)
        if tuple(h.shape) != self.feature_shape or not torch.isfinite(h).all():
            raise ValueError("Expected finite %s raw features" % (self.feature_shape,))
        # 原始采集是 bfloat16，缓存存储用 FP8 E5M2；不要用权重的 E4M3。
        x = h.to(torch.bfloat16).to(torch.float8_e5m2).float().reshape(1, self.input_dim)
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite features after E5M2 quantization")
        return x

    @torch.inference_mode()
    def score(self, hidden):
        x = self.prepare_features(hidden)
        normalized = (x - self.mean) / self.std
        logit = (self.blend_alpha * F.linear(normalized, self.linear[0].weight, self.linear[0].bias)
                 + (1.0 - self.blend_alpha) * self.mlp(normalized)).item()
        if not math.isfinite(logit):
            raise ValueError("Non-finite blend logit")
        return {"score": float(torch.sigmoid(torch.tensor(logit)).item()), "logit": logit,
                "threshold": self.threshold, "threshold_logit": self.threshold_logit,
                "flagged": logit >= self.threshold_logit}
