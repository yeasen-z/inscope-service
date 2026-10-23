from dataclasses import dataclass
from pathlib import Path
import math
import os

from inscope_contract import (
    BLEND_ARMS,
    DEFAULT_ENGINE_PORT,
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_INFLIGHT,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_PROBE_PREFIX,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SEED,
    INSCOPE_COMMIT,
    MODEL_ARCHITECTURE,
    MODEL_QUANTIZATION,
    THRESHOLD_LOGIT,
    blend_arms,
)

from .composer import ComposerConfig
from .paths import DEFAULT_MODELS_DIR, ROOT


def _flag(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    model_path: str
    api_key: str
    # 上下文编排 profile。必须显式给出；不存在「未配置就用某套规则」的隐式默认。
    composer_profile: str = "a5_simplified_v1"
    composer: ComposerConfig = ComposerConfig(probe_prefix=DEFAULT_PROBE_PREFIX)
    models_dir: str = DEFAULT_MODELS_DIR
    detector_seed: int = DEFAULT_SEED
    threshold_logit: float | None = None
    backend_url: str = "http://127.0.0.1:%d" % DEFAULT_ENGINE_PORT
    backend_api_key: str = ""
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES

    def __post_init__(self):
        if not self.model_path or not self.api_key:
            raise ValueError("INSCOPE_MODEL_PATH and INSCOPE_API_KEY are required")
        if not 1 <= self.max_tokens <= DEFAULT_MAX_INPUT_TOKENS or self.max_inflight < 1:
            raise ValueError("max_tokens must be 1..%d; max_inflight must be positive"
                             % DEFAULT_MAX_INPUT_TOKENS)
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.max_body_bytes < 1:
            raise ValueError("Invalid timeout/body limit")
        if self.threshold_logit is not None and not math.isfinite(self.threshold_logit):
            raise ValueError("threshold_logit must be finite")
        # 阈值只对「不合并」composer 配置标定过；换 seed 或开 merge 必须重评。
        if self.threshold_logit is None and self.detector_seed not in THRESHOLD_LOGIT:
            raise ValueError("No calibrated threshold for seed %d" % self.detector_seed)
        if not self.backend_url.startswith(("http://", "https://")):
            raise ValueError("INSCOPE_ENGINE_URL must be an HTTP URL")

    @classmethod
    def from_env(cls):
        threshold = os.getenv("INSCOPE_THRESHOLD_LOGIT", "")
        return cls(
            model_path=os.environ.get("INSCOPE_MODEL_PATH", ""),
            api_key=os.environ.get("INSCOPE_API_KEY", ""),
            composer_profile=os.getenv("INSCOPE_COMPOSER_PROFILE", "a5_simplified_v1"),
            composer=ComposerConfig(
                discard_cots=_flag("INSCOPE_DISCARD_COTS", True),
                retain_n_assistant_turns=int(os.getenv("INSCOPE_RETAIN_ASSISTANT_TURNS", "1")),
                merge_consecutive_assistants=_flag("INSCOPE_MERGE_CONSECUTIVE_ASSISTANTS", False),
                drop_final_tool_result=_flag("INSCOPE_DROP_FINAL_TOOL_RESULT", False),
                probe_prefix=os.getenv("INSCOPE_PROBE_PREFIX", DEFAULT_PROBE_PREFIX),
                enable_thinking=_flag("INSCOPE_ENABLE_THINKING", True),
                selected_positions=(-1,),
            ),
            models_dir=os.getenv("INSCOPE_DETECTOR_DIR", DEFAULT_MODELS_DIR),
            detector_seed=int(os.getenv("INSCOPE_DETECTOR_SEED", str(DEFAULT_SEED))),
            threshold_logit=float(threshold) if threshold else None,
            backend_url=os.getenv("INSCOPE_ENGINE_URL", "http://127.0.0.1:%d" % DEFAULT_ENGINE_PORT).rstrip("/"),
            backend_api_key=os.getenv("SGLANG_API_KEY", ""),
            max_tokens=int(os.getenv("INSCOPE_MAX_INPUT_TOKENS", str(DEFAULT_MAX_INPUT_TOKENS))),
            max_inflight=int(os.getenv("INSCOPE_MAX_INFLIGHT", str(DEFAULT_MAX_INFLIGHT))),
            timeout=float(os.getenv("INSCOPE_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT))),
        )


def validate_model(path):
    """本地预量化 FP8 权重校验；实现在 contract 包里，三端共用。"""
    from inscope_contract.model import validate_model_weights
    return validate_model_weights(path)
