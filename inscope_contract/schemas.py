"""HTTP 输入输出契约。服务端与调用方共用，避免两边各写一份校验。"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import SUPPORTED_ROLES


class DetectionRequest(BaseModel):
    """/detect 的输入：只有 messages 和 tools，setting 由服务端配置。"""

    model_config = ConfigDict(extra="forbid")
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=128)
    tools: list[dict[str, Any]] | None = None

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


class DetectionResult(BaseModel):
    """检测结果；字段名是对外契约的一部分。"""

    score: float
    logit: float
    threshold: float
    threshold_logit: float
    flagged: bool
    risk: str
    model: str


class AnalyzeRequest(DetectionRequest):
    """/analyze 的输入：在完整请求之上指定要对哪条 observation 做定位。

    粒度/预算都是显式参数，服务端不自动跑多路；缺省 paragraph + auto，
    auto 会按段落数在 exact / hierarchical 之间自行降级。
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
    thresholds: list[float] | None = None

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
