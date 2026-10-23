"""服务配置。

配置只回答「部署在哪、限额多少」，不回答「探针怎么编排」：
探针结构来自 checkpoint，上下文编排来自 composer profile。
两者各有唯一来源，此文件不得再持有第三份副本。
"""
from dataclasses import dataclass
import math
import os

from inscope_contract import (
    DEFAULT_ENGINE_PORT,
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_INFLIGHT,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
)


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
    # 探针 checkpoint 路径。结构、层号、位置、归一化、阈值全部由该文件自描述。
    probe_checkpoint: str = ""
    # 上下文编排 profile id。必须显式给出；未知 id 在启动期即失败。
    composer_profile: str = ""
    backend_url: str = "http://127.0.0.1:%d" % DEFAULT_ENGINE_PORT
    backend_api_key: str = ""
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES

    def __post_init__(self):
        if not self.model_path or not self.api_key:
            raise ValueError("INSCOPE_MODEL_PATH and INSCOPE_API_KEY are required")
        if not self.probe_checkpoint:
            raise ValueError("INSCOPE_PROBE_CHECKPOINT is required")
        if not self.composer_profile:
            raise ValueError("INSCOPE_COMPOSER_PROFILE is required")
        if not 1 <= self.max_tokens <= DEFAULT_MAX_INPUT_TOKENS or self.max_inflight < 1:
            raise ValueError("max_tokens must be 1..%d; max_inflight must be positive"
                             % DEFAULT_MAX_INPUT_TOKENS)
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.max_body_bytes < 1:
            raise ValueError("Invalid timeout/body limit")
        if not self.backend_url.startswith(("http://", "https://")):
            raise ValueError("INSCOPE_ENGINE_URL must be an HTTP URL")

    @classmethod
    def from_env(cls):
        return cls(
            model_path=os.environ.get("INSCOPE_MODEL_PATH", ""),
            api_key=os.environ.get("INSCOPE_API_KEY", ""),
            probe_checkpoint=os.environ.get("INSCOPE_PROBE_CHECKPOINT", ""),
            composer_profile=os.getenv("INSCOPE_COMPOSER_PROFILE", ""),
            backend_url=os.getenv("INSCOPE_ENGINE_URL",
                                  "http://127.0.0.1:%d" % DEFAULT_ENGINE_PORT).rstrip("/"),
            backend_api_key=os.getenv("SGLANG_API_KEY", ""),
            max_tokens=int(os.getenv("INSCOPE_MAX_INPUT_TOKENS", str(DEFAULT_MAX_INPUT_TOKENS))),
            max_inflight=int(os.getenv("INSCOPE_MAX_INFLIGHT", str(DEFAULT_MAX_INFLIGHT))),
            timeout=float(os.getenv("INSCOPE_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT))),
        )


def validate_model(path):
    """本地预量化 FP8 权重校验；实现在 contract 包里，三端共用。"""
    from inscope_contract.model import validate_model_weights
    return validate_model_weights(path)
