"""HTTP 输入输出契约。服务端与调用方共用，避免两边各写一份校验。"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import SUPPORTED_ROLES


class DetectionRequest(BaseModel):
    """/detect 的输入：只有 messages 和 tools，setting 由服务端配置。"""

    model_config = ConfigDict(extra="forbid")
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=128)
    tools: list[dict[str, Any]] | None = None
    # 可选：覆盖启动期默认编排策略；必须与已加载探针兼容。
    composer_profile: str | None = Field(default=None, min_length=1)
    # 可选：只要这些风险的结果（须是服务已加载的子集）；None 表示全部已加载风险。
    enabled_risks: list[str] | None = Field(default=None, min_length=1)
    # 可选：observation_blocks 探针返回 block 级 attention 通道（同一次前向的
    # 副产品）；其他架构的探针忽略该字段。
    explain: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages):
        for message in messages:
            if message.get("role") not in SUPPORTED_ROLES:
                raise ValueError("Unsupported message role")
            if not isinstance(message.get("content"), str):
                # 只有「带 tool_calls 的 assistant」允许 content=None。
                if not (message.get("role") == "assistant"
                        and message.get("content") is None
                        and message.get("tool_calls")):
                    raise ValueError("Text content is required (multimodal input is unsupported)")
            if "tool_calls" in message:
                if not isinstance(message["tool_calls"], list):
                    raise ValueError("tool_calls must be a list")
                for call in message["tool_calls"]:
                    if not isinstance(call, dict):
                        raise ValueError("Invalid tool call")
                    function = call.get("function", call)
                    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                        raise ValueError("Tool function name is required")
        return messages


class BankRiskVerdict(BaseModel):
    """单个风险的判定：阈值来自该风险自己的 checkpoint。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    flagged: bool


class DetectionResult(BaseModel):
    """Successful /detect response; numeric values are finite and in probability units."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False,
                              populate_by_name=True)
    probability: float = Field(ge=0, le=1)
    logit: float
    threshold: float = Field(ge=0, le=1)
    label: Literal["attack", "benign"]
    risk: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    request_id: str = Field(min_length=1)
    # 数字（历史形态）或 {阶段: 毫秒}（Probe Bank 兼容形态，含 total）。
    latency_ms: float | dict[str, float]
    composer_profile: str = Field(min_length=1)
    checkpoint_sha256: str = Field(min_length=1)
    composer: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None
    # 多风险（Probe Bank 兼容）字段：只在注册了额外探针时出现。
    enabled_risks: list[str] | None = None
    disabled_risks: list[str] | None = None
    overall: bool | None = None
    profile_name: str | None = Field(default=None, alias="profile")
    per_risk: dict[str, BankRiskVerdict] | None = None
    per_entry: dict[str, float] | None = None
    triggered_entries: list[str] | None = None
    detector_model: str | None = None
    bank_version: str | None = None
    scores: dict[str, tuple[float, float]] | None = None
    # observation_blocks 探针的 explain 输出（仅 explain=true 时出现）。
    # blocks[].message_index 属于 compiled_messages 坐标系，见 coordinate_system；
    # blocks[].source_message_indices 指回原始请求里的消息下标。
    blocks: list[dict[str, Any]] | None = None
    coordinate_system: Literal["compiled_messages"] | None = None
    compiled_context: str | None = None
    capture_positions: list[int] | None = None
    importance_semantics: str | None = None

    @model_validator(mode="after")
    def validate_label(self):
        expected = "attack" if self.probability >= self.threshold else "benign"
        if self.label != expected:
            raise ValueError("label is inconsistent with probability and threshold")
        return self

    @property
    def flagged(self) -> bool:
        """SDK compatibility convenience; not a separate wire field."""
        return self.label == "attack"


class DetectionItemError(BaseModel):
    """A failed batch item has no usable score and must never be treated as benign."""

    model_config = ConfigDict(extra="forbid", strict=True)
    error: str = Field(min_length=1)
    detail: str
    status_code: int = Field(ge=400, le=599)
    risk: str | None = None
    model: str | None = None
    request_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    composer_profile: str | None = None
    checkpoint_sha256: str | None = None
    composer: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None


class DetectionBatchResult(BaseModel):
    """Ordered batch response; successes and failures retain their original slots."""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    schema_name: Literal["inscope.detect_batch.v1"] = Field(alias="schema")
    count: int = Field(ge=1)
    results: list[DetectionResult | DetectionItemError] = Field(min_length=1)
    risk: str = Field(min_length=1)
    model: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    latency_ms: float | dict[str, float]
    composer_profile: str = Field(min_length=1)
    checkpoint_sha256: str = Field(min_length=1)
    composer: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None
    enabled_risks: list[str] | None = None
    disabled_risks: list[str] | None = None
    overall: bool | None = None
    profile_name: str | None = Field(default=None, alias="profile")
    per_risk: dict[str, BankRiskVerdict] | None = None
    per_entry: dict[str, float] | None = None
    triggered_entries: list[str] | None = None
    detector_model: str | None = None
    bank_version: str | None = None

    @model_validator(mode="after")
    def validate_count(self):
        if self.count != len(self.results):
            raise ValueError("count does not match results length")
        return self


class DetectionBatchRequest(BaseModel):
    """Batch wire shape; the service validates each conversation independently."""

    model_config = ConfigDict(extra="forbid")
    messages: list[list[dict[str, Any]]] = Field(min_length=1, max_length=128)
    tools: list[dict[str, Any]] | None = None
    composer_profile: str | None = Field(default=None, min_length=1)


DetectionResponse = DetectionResult | DetectionBatchResult


class AnalyzeRequest(DetectionRequest):
    """/analyze 的输入：在完整请求之上指定要对哪条 observation 做定位。

    粒度/预算都是显式参数，服务端不自动跑多路；缺省 paragraph + auto，
    auto 会按段落数在 exact / hierarchical 之间自行降级。可选
    ``composer_profile`` 覆盖本次请求的上下文编排策略，必须与探针兼容。
    """

    observation_index: int = Field(ge=0)
    granularity: Literal["paragraph", "line", "sentence"] = "paragraph"
    # auto：单元数 <= 8 用 exact（2^8=256，正好一个预算），超过则分级展开。
    mode: Literal["auto", "exact", "hierarchical"] = "auto"
    focus: Literal["high", "low"] = "high"
    # 分级展开时每层组数上限；auto 会按单元数自适应，显式传入则原样使用。
    group_size: int = Field(default=6, ge=2, le=8)
    max_coalitions: int = Field(default=256, ge=4, le=4096)

    @model_validator(mode="after")
    def validate_observation(self):
        if self.observation_index >= len(self.messages):
            raise ValueError(
                "observation_index %d is outside the %d-message request"
                % (self.observation_index, len(self.messages)))
        return self


class AuditRequest(DetectionRequest):
    """/audit 的输入：一段完整执行轨迹，按检查点逐步评分。

    检查点默认是"每条 tool/user 观察"——审计关心的是新进入上下文的
    外部内容，而不是模型自己说了什么。调用方可显式收窄起止位置。
    """

    checkpoint_roles: list[Literal["tool", "user", "assistant", "system", "developer"]] = Field(
        default_factory=lambda: ["tool", "user"])
    start_index: int = Field(default=0, ge=0)

    @field_validator("checkpoint_roles")
    @classmethod
    def validate_checkpoint_roles(cls, roles):
        if not roles:
            raise ValueError("checkpoint_roles must not be empty")
        return list(dict.fromkeys(roles))

    @model_validator(mode="after")
    def validate_window(self):
        if self.start_index >= len(self.messages):
            raise ValueError(
                "start_index %d is outside the %d-message request"
                % (self.start_index, len(self.messages)))
        return self
