"""HTTP 输入输出契约。服务端与调用方共用，避免两边各写一份校验。"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
