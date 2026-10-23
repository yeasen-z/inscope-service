"""simplified context composer.

自包含移植自 FYH `follow_your_heart/runtime/context_composer.py`
（`build_probe_prompt` + `_select_prompt_messages` + `_normalize_prompt_messages`）
与 `message_content.normalize_chat_message`。不 import 训练树，热路径只依赖 tokenizer。

与在线 `rendering.render_ids` 的差别只有四处，其余消息处理规则一致：
  1. 选窗：开场上下文 + 所有 system/developer + 最近 N 个完整 assistant 动作块及其后全部消息。
  2. discard_cots：删独立 reasoning 字段和 assistant ` thinking…</think>`。
  3. merge_consecutive_assistants：连续 assistant 合并为一个动作块。
  4. probe_prefix：模板文本与问题**分别编码**后拼接，问题落在 generation header 之后。
"""
from __future__ import annotations

import ast
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComposerConfig:
    """服务端显式配置；不从消息内容猜测。"""

    discard_cots: bool = True
    retain_n_assistant_turns: int = 1
    merge_consecutive_assistants: bool = False
    drop_final_tool_result: bool = False
    probe_prefix: str = ""
    enable_thinking: bool = True
    selected_positions: tuple[int, ...] = (-1,)

    def __post_init__(self):
        if self.retain_n_assistant_turns != -1 and self.retain_n_assistant_turns < 1:
            raise ValueError("retain_n_assistant_turns must be -1 or positive")
        if not self.selected_positions or any(type(p) is not int for p in self.selected_positions):
            raise ValueError("selected_positions must be a non-empty list of ints")

    def describe(self) -> dict[str, Any]:
        return {"discard_cots": self.discard_cots,
                "retain_n_assistant_turns": self.retain_n_assistant_turns,
                "merge_consecutive_assistants": self.merge_consecutive_assistants,
                "drop_final_tool_result": self.drop_final_tool_result,
                "probe_prefix": self.probe_prefix,
                "enable_thinking": self.enable_thinking,
                "selected_positions": list(self.selected_positions)}


@dataclass(frozen=True)
class ComposedPrompt:
    text: str
    token_ids: tuple[int, ...]


def compose_prompt(tokenizer, messages, tools=None, config: ComposerConfig | None = None) -> ComposedPrompt:
    """编译一次检测输入。token_ids 即必须原样送进 engine 的 input_ids。"""
    config = config or ComposerConfig()
    if not messages:
        raise ValueError("messages must be non-empty")

    selected = select_prompt_messages(messages, config.retain_n_assistant_turns)
    if config.drop_final_tool_result:
        selected = drop_final_tool_result(selected)
    prompt_messages = normalize_prompt_messages(selected, config=config)
    if config.retain_n_assistant_turns != -1:
        # Qwen3 模板没有 developer 分支，会静默丢掉该消息；只在送模板的副本上改成 system。
        for message in prompt_messages:
            if message["role"] == "developer":
                message["role"] = "system"
    if not prompt_messages:
        raise ValueError("composer produced no prompt messages")

    template_text = _apply_chat_template(tokenizer, prompt_messages,
                                        tools=list(tools or ()), enable_thinking=config.enable_thinking)
    token_ids = list(_encode_text(tokenizer, template_text))
    if not token_ids:
        raise ValueError("chat template produced an empty prompt")
    if config.probe_prefix:
        # 模板边界与问题分别编码，边界 token 不因拼接而改变。
        token_ids += list(_encode_text(tokenizer, config.probe_prefix))
    if any(type(t) is not int or t < 0 for t in token_ids):
        raise ValueError("Tokenizer returned invalid token ids")
    return ComposedPrompt(text=template_text + (config.probe_prefix or ""), token_ids=tuple(token_ids))


def select_prompt_messages(messages: Sequence[Mapping[str, Any]], retain_n_assistant_turns: int):
    """保留开场任务/指令与最后 N 个完整 agent turn。

    连续 assistant（例如 reasoning 后紧跟调用）算一个 turn。选中后缀整体保留：
    并行 tool 结果、原生 user 观察、后续 user 指令都维持原顺序；
    不解析正文、不猜 call ID、不修补来源配对。
    """
    if retain_n_assistant_turns == -1:
        return list(messages)
    if retain_n_assistant_turns < 1:
        raise ValueError("retain_n_assistant_turns must be -1 or positive")
    starts = [i for i, message in enumerate(messages)
              if message.get("role") == "assistant" and
              (i == 0 or messages[i - 1].get("role") != "assistant")]
    if not starts:
        return list(messages)
    cutoff = starts[max(0, len(starts) - retain_n_assistant_turns)]
    opening_end = starts[0]
    return [message for i, message in enumerate(messages)
            if i < opening_end or i >= cutoff or
            message.get("role") in ("system", "developer")]


def drop_final_tool_result(messages: Sequence[Mapping[str, Any]]):
    """去掉一条尾部观察；原生 user 观察跟在动作之后，同样算观察。"""
    last_action = next((i for i in range(len(messages) - 1, -1, -1)
                        if messages[i].get("role") == "assistant"), -1)
    for i in range(len(messages) - 1, last_action, -1):
        role = messages[i].get("role")
        if role == "tool" or (role == "user" and last_action >= 0):
            return list(messages[:i]) + list(messages[i + 1:])
    raise ValueError("no final tool/environment result to drop")


def normalize_prompt_messages(messages: Sequence[Mapping[str, Any]], *, config: ComposerConfig):
    normalized: list[dict[str, Any]] = []
    for message in messages:
        current = _normalize_prompt_message(message, discard_cots=config.discard_cots)
        if (config.merge_consecutive_assistants and current["role"] == "assistant"
                and normalized and normalized[-1]["role"] == "assistant"):
            previous = normalized[-1]
            for key in ("content", "reasoning_content", "reasoning"):
                if current.get(key):
                    previous[key] = previous[key] + "\n" + current[key] if previous.get(key) else current[key]
            if current.get("tool_calls"):
                previous.setdefault("tool_calls", []).extend(current["tool_calls"])
        else:
            normalized.append(current)
    return normalized


def _normalize_prompt_message(message: Mapping[str, Any], *, discard_cots: bool) -> dict[str, Any]:
    source = _discard_reasoning_content(message) if discard_cots else message
    if discard_cots and message.get("role") == "assistant" and isinstance(source.get("content"), str):
        source = dict(source)
        source["content"] = re.sub(r"<think>.*?</think>\s*", "", source["content"], flags=re.DOTALL)
    normalized = normalize_chat_message(source)
    if not discard_cots and "reasoning_content" not in normalized and "reasoning" in normalized:
        normalized["reasoning_content"] = normalized["reasoning"]
    tool_calls = _normalize_tool_calls(source.get("tool_calls"))
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    for key in ("tool_call_id", "name"):
        value = _coerce_optional_text(source.get(key))
        if value is not None:
            normalized[key] = value
    return normalized


def normalize_chat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """content 归一 + 调用归一 + reasoning 提取，规则同 FYH message_content。"""
    if not isinstance(message, Mapping):
        raise ValueError("each message must be a JSON object")
    role = str(message.get("role") or "")
    if not role:
        raise ValueError("message role is required")
    raw_content = message.get("content")
    normalized: dict[str, Any] = {"role": role, "content": _coerce_visible_content(raw_content)}
    tool_calls = _normalize_response_tool_calls(message.get("tool_calls"))
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    if role == "tool":
        tool_call_id = _coerce_optional_text(message.get("tool_call_id"))
        if tool_call_id is not None:
            normalized["tool_call_id"] = tool_call_id
        tool_name = _coerce_optional_text(message.get("name"))
        if tool_name is None:
            tool_call_payload = message.get("tool_call")
            if isinstance(tool_call_payload, Mapping):
                tool_name = _coerce_optional_text(
                    tool_call_payload.get("function") or tool_call_payload.get("name"))
        if tool_name is not None:
            normalized["name"] = tool_name

    reasoning_content = _coerce_optional_text(message.get("reasoning_content"))
    reasoning = _coerce_optional_text(message.get("reasoning"))
    if role == "assistant":
        inferred = _extract_reasoning_from_content(raw_content)
        if reasoning_content is None and reasoning is None and inferred is not None:
            reasoning_content = inferred
        if reasoning_content is None and reasoning is None:
            inline, visible = parse_qwen3_reasoning_output(_coerce_raw_content(raw_content))
            if inline is not None:
                reasoning_content = inline
                normalized["content"] = visible or ""
    if reasoning_content is not None:
        normalized["reasoning_content"] = reasoning_content
    if reasoning is not None:
        normalized["reasoning"] = reasoning
    return normalized


def parse_qwen3_reasoning_output(text: str) -> tuple[str | None, str | None]:
    start_token, end_token = "<think>", "</think>"
    if start_token not in text:
        return None, text.strip() or None
    start_index = text.index(start_token) + len(start_token)
    if end_token not in text[start_index:]:
        return text[start_index:].strip() or None, None
    end_index = text.index(end_token, start_index)
    return (text[start_index:end_index].strip() or None,
            text[end_index + len(end_token):].strip() or None)


def _apply_chat_template(tokenizer, messages, *, tools, enable_thinking: bool) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template")
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True,
                              "enable_thinking": enable_thinking}
    try:
        parameters = inspect.signature(apply_chat_template).parameters
    except Exception:
        parameters = {}
    if tools and "tools" in parameters:
        kwargs["tools"] = list(tools)
    return apply_chat_template(messages, **kwargs)


def _encode_text(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping) and "input_ids" in encoded:
        return list(encoded["input_ids"])
    raise TypeError("tokenizer must provide encode() or __call__(...)->input_ids")


def _discard_reasoning_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _discard_reasoning_content(item) for key, item in value.items()
                if key not in ("reasoning_content", "reasoning")}
    if isinstance(value, list):
        return [_discard_reasoning_content(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_discard_reasoning_content(item) for item in value)
    return value


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)):
        return []
    return [call for call in (_normalize_tool_call(c) for c in tool_calls) if call is not None]


def _normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, Mapping):
        return None
    function_payload = tool_call.get("function")
    function_name: str | None = None
    arguments: Any = None
    if isinstance(function_payload, Mapping):
        function_name = _coerce_optional_text(function_payload.get("name") or function_payload.get("function"))
        arguments = function_payload.get("arguments")
        if arguments is None:
            arguments = function_payload.get("args")
    else:
        function_name = _coerce_optional_text(function_payload)
    if function_name is None:
        function_name = _coerce_optional_text(tool_call.get("name"))
    if arguments is None:
        arguments = tool_call.get("arguments")
    if arguments is None:
        arguments = tool_call.get("args")
    if function_name is None:
        return None
    resolved: Any = arguments
    if isinstance(resolved, str):
        try:
            resolved = json.loads(resolved)
        except (ValueError, TypeError):
            try:
                resolved = ast.literal_eval(resolved)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                pass
    if not isinstance(resolved, Mapping):
        resolved = resolved if resolved is not None else {}
    normalized: dict[str, Any] = {"function": {"name": function_name, "arguments": resolved}}
    tool_call_id = _coerce_optional_text(tool_call.get("id"))
    if tool_call_id is not None:
        normalized["id"] = tool_call_id
    return normalized


def _normalize_response_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)):
        return None
    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        function_payload = tool_call.get("function")
        function_name: str | None = None
        arguments: Any = None
        if isinstance(function_payload, Mapping):
            function_name = _coerce_optional_text(function_payload.get("name") or function_payload.get("function"))
            arguments = function_payload.get("arguments")
            if arguments is None:
                arguments = function_payload.get("args")
        else:
            function_name = _coerce_optional_text(function_payload)
        if function_name is None:
            function_name = _coerce_optional_text(tool_call.get("name"))
        if arguments is None:
            arguments = tool_call.get("arguments")
        if arguments is None:
            arguments = tool_call.get("args")
        if function_name is None:
            continue
        normalized_call: dict[str, Any] = {
            "function": {"name": function_name, "arguments": arguments if arguments is not None else {}}}
        tool_call_id = _coerce_optional_text(tool_call.get("id"))
        if tool_call_id is not None:
            normalized_call["id"] = tool_call_id
        normalized_calls.append(normalized_call)
    return normalized_calls or None


def _extract_reasoning_from_content(content: Any) -> str | None:
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts = [str(item.get("content")) for item in content
                 if isinstance(item, Mapping) and item.get("type") in {"thinking", "redacted_thinking"}
                 and item.get("content")]
        if parts:
            return "\n".join(parts)
    return None


def _coerce_visible_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") in {"thinking", "redacted_thinking"}:
                    continue
                text = item.get("content")
                if text is None:
                    text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _coerce_raw_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("content")
                if text is None:
                    text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value).strip() or None
