"""服务配置。

配置只回答「部署在哪、限额多少」，不回答「探针怎么编排」：
探针结构来自 checkpoint，上下文编排来自 composer profile。
两者各有唯一来源，此文件不得再持有第三份副本。
"""
from dataclasses import dataclass, field
from collections.abc import Mapping
import math
import os

from inscope_contract import (
    DEFAULT_ANALYZE_BATCH,
    DEFAULT_DETECT_BATCH,
    DEFAULT_AUDIT_TIMEOUT,
    DEFAULT_ANALYZE_TIMEOUT,
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


# 与 coolwatch Probe Bank 的风险代号一致；只允许已知风险，避免拼写错误静默生效。
KNOWN_RISKS = ("harmful", "prompt_leakage", "ipi")


def parse_probe_spec(value):
    """解析 ``risk=checkpoint[,risk=checkpoint]``；每项风险一份独立 checkpoint。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("INSCOPE_PROBES must be 'risk=/path[,risk=/path]'")
    probes = {}
    for item in value.split(","):
        risk, sep, path = item.partition("=")
        risk, path = risk.strip(), path.strip()
        if not sep or not risk or not path:
            raise ValueError("Each probe entry must be 'risk=/path/checkpoint.pt'")
        if risk not in KNOWN_RISKS:
            raise ValueError("Unknown probe risk %r; allowed: %s" % (risk, ", ".join(KNOWN_RISKS)))
        if risk in probes:
            raise ValueError("Duplicate probe risk %r" % risk)
        probes[risk] = path
    return probes


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
    # 分析总 deadline 与单次引擎请求 timeout 分开配置。
    analyze_timeout: float = DEFAULT_ANALYZE_TIMEOUT
    # 每桶 prompt 数上限，不等于分析请求数或 GPU 并发上限。
    analyze_batch: int = DEFAULT_ANALYZE_BATCH
    # /detect 批量：单请求最多合并多少条 prompt
    detect_batch: int = DEFAULT_DETECT_BATCH
    # 审计总超时。检查点串行推进以命中前缀缓存，因此不存在并发参数。
    audit_timeout: float = DEFAULT_AUDIT_TIMEOUT
    # 整个 HTTP 批量请求的条数上限，与每桶规模分开。
    max_batch_items: int = 128
    # 实际引擎 HTTP 请求上限，所有业务入口共享。
    backend_max_inflight: int = DEFAULT_MAX_INFLIGHT
    # 可选多风险注册表 {风险: checkpoint}。为空时只加载 probe_checkpoint。
    probes: Mapping[str, str] = field(default_factory=dict)
    # Probe Bank 协议里 detector_model 的值；默认沿用模型标签。
    detector_model: str = ""
    # 主探针在 per_risk 里的风险代号。
    primary_risk: str = "ipi"

    def __post_init__(self):
        if not self.model_path or not self.api_key:
            raise ValueError("INSCOPE_MODEL_PATH and INSCOPE_API_KEY are required")
        if not self.probe_checkpoint:
            raise ValueError("INSCOPE_PROBE_CHECKPOINT is required")
        if not self.composer_profile:
            raise ValueError("INSCOPE_COMPOSER_PROFILE is required")
        unknown = set(self.probes) - set(KNOWN_RISKS)
        if unknown:
            raise ValueError("Unknown probe risks: %s" % ", ".join(sorted(unknown)))
        if any(not isinstance(path, str) or not path for path in self.probes.values()):
            raise ValueError("Every probe risk needs a non-empty checkpoint path")
        if self.probe_checkpoint and self.primary_risk in self.probes:
            raise ValueError("duplicate risk %r: primary must not also appear in probes"
                             % self.primary_risk)
        if not 1 <= self.max_tokens <= DEFAULT_MAX_INPUT_TOKENS or self.max_inflight < 1:
            raise ValueError("max_tokens must be 1..%d; max_inflight must be positive"
                             % DEFAULT_MAX_INPUT_TOKENS)
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.max_body_bytes < 1:
            raise ValueError("Invalid timeout/body limit")
        if not math.isfinite(self.analyze_timeout) or self.analyze_timeout < self.timeout:
            raise ValueError("analyze_timeout must be finite and >= request timeout")
        if not math.isfinite(self.audit_timeout) or self.audit_timeout < self.timeout:
            raise ValueError("audit_timeout must be finite and >= request timeout")
        for name in ("analyze_batch", "detect_batch", "max_batch_items", "backend_max_inflight"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("%s must be a positive integer" % name)
        if self.max_batch_items > 128:
            raise ValueError("max_batch_items cannot exceed public contract limit 128")
        from urllib.parse import urlsplit
        url = urlsplit(self.backend_url)
        if (url.scheme not in ("http", "https") or not url.hostname or url.username
                or url.password or url.fragment):
            raise ValueError("INSCOPE_ENGINE_URL must be an HTTP URL without credentials or fragment")
        try:
            port = url.port
        except ValueError as exc:
            raise ValueError("INSCOPE_ENGINE_URL has an invalid port") from exc
        if port is not None and port < 1:
            raise ValueError("INSCOPE_ENGINE_URL has an invalid port")

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
            max_body_bytes=int(os.getenv("INSCOPE_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))),
            max_batch_items=int(os.getenv("INSCOPE_MAX_BATCH_ITEMS", "128")),
            backend_max_inflight=int(os.getenv("INSCOPE_BACKEND_MAX_INFLIGHT", str(DEFAULT_MAX_INFLIGHT))),
            analyze_batch=int(os.getenv("INSCOPE_ANALYZE_BATCH", str(DEFAULT_ANALYZE_BATCH))),
            detect_batch=int(os.getenv("INSCOPE_DETECT_BATCH", str(DEFAULT_DETECT_BATCH))),
            audit_timeout=float(os.getenv("INSCOPE_AUDIT_TIMEOUT", str(DEFAULT_AUDIT_TIMEOUT))),
            analyze_timeout=float(os.getenv("INSCOPE_ANALYZE_TIMEOUT",
                                            str(DEFAULT_ANALYZE_TIMEOUT))),
            probes=parse_probe_spec(os.environ["INSCOPE_PROBES"])
            if os.getenv("INSCOPE_PROBES") else {},
            detector_model=os.getenv("INSCOPE_DETECTOR_MODEL", ""),
            primary_risk=os.getenv("INSCOPE_PRIMARY_RISK", "ipi"),
        )


def validate_model(path):
    """本地预量化 FP8 权重校验；实现在 contract 包里，三端共用。"""
    from inscope_contract.model import validate_model_weights
    return validate_model_weights(path)
