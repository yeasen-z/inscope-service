"""Observation-block 编译器：ipi-doom canonical simplified-v3 的忠实移植。

移植来源（对拍基准，逐字节对齐）:
- ``ipi-doom/src/featurize/observation_blocks.py``（build_block_prompt / block_spans）
- ``ipi-doom/follow_your_heart/runtime/message_content.py``（normalize_chat_message）
- ``ipi-doom/follow_your_heart/runtime/context_composer.py``（_normalize_prompt_messages 链）
- ``ipi-doom/src/featurize/position_scheduler.py``（_PositionTokenizer / _render_position_template）
- ``ipi-doom/src/featurize/observation_turn.py``（observation_turn_span）

刻意不复用 composer.py 的 normalize_prompt_messages：那一版在 discard_cots 时有
额外的 reasoning 后处理（未闭合 think 再剥离），与训练侧 FYH 行为不同；探针
训练特征来自 FYH 链，这里必须与其逐字节一致，包括"未闭合 think 残留进模板"
这类边界行为。黄金样本对拍测试（tests/test_observation_blocks_parity.py）钉住。
"""
from __future__ import annotations

import inspect
import json
import re
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .composer import (_apply_chat_template, _encode_text, retained_message_indices,
                       select_prompt_messages)

# 与 ipi-doom featurize/observation_blocks.py SETTING 完全一致。
SETTING = dict(retain_n_assistant_turns=1, discard_cots=True,
               merge_consecutive_assistants=True, enable_thinking=True, probe_prefix='')


@dataclass(frozen=True)
class ObservationBlocks:
    """一次编译的完整产物。blocks 坐标属于编译后 messages（compiled_messages）。"""

    input_ids: tuple[int, ...]
    text: str
    positions: tuple[int, ...]
    blocks: tuple[dict, ...]
    # 编译后 messages 下标 -> 原始请求消息下标（合并 assistant 时一对多）。
    message_sources: tuple[tuple[int, ...], ...] = ()


def block_spans(content: str):
    """Literal double-newline boundaries; offsets refer to unmodified content."""
    start = 0
    for part in content.split('\n\n'):
        if part.strip():
            yield start, start + len(part)
        start += len(part) + 2


def observation_turn_span(messages: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """最后一条 assistant 动作之后的 tool/user 观察区间（含两端下标）。"""
    action = next((i for i in range(len(messages) - 1, -1, -1)
                   if messages[i].get('role') == 'assistant'), None)
    if action is None:
        raise ValueError('No preceding assistant action for an observation turn')
    observations = [i for i in range(action + 1, len(messages))
                    if messages[i].get('role') in ('tool', 'user')]
    if not observations:
        raise ValueError('No observation after the final assistant action')
    return observations[0], observations[-1]


class _PositionTokenizer:
    """Reuse canonical tokenization for offsets within one prompt compilation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.encodings = {}

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        key = (text, add_special_tokens)
        if key not in self.encodings:
            self.encodings[key] = self.tokenizer(
                text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
        return self.encodings[key]

    def encode(self, text, *, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)['input_ids']


@dataclass(frozen=True)
class _Prompt:
    text: str
    token_ids: tuple[int, ...]


def _render_position_template(prompt: _Prompt, tokenizer, messages, *, tools, enable_thinking):
    """ipi-doom _render_position_template 在 add_omit_reminder=False 时的等价路径。

    v3 编译不产生 omit_reminder 边界（prompt.omit_reminder_before_messages 恒为空），
    因此渲染就是一次 chat template 调用。
    """
    return _apply_chat_template(tokenizer, messages, tools=tools, enable_thinking=enable_thinking)


# --------------------------------------------------------------- FYH 消息规范化
# 以下 normalize 链逐行移植自 follow_your_heart/runtime/message_content.py 与
# context_composer.py（训练侧实际执行的路径），不做任何"改进"。

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


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, Mapping):
        for key in ("content", "text", "reasoning_content", "reasoning"):
            if key in value:
                return _coerce_optional_text(value.get(key))
        return None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        parts = [_coerce_optional_text(item) for item in value]
        filtered = [part for part in parts if part]
        return "\n".join(filtered) or None
    return str(value).strip() or None


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
            "function": {
                "name": function_name,
                "arguments": arguments if arguments is not None else {},
            }
        }
        tool_call_id = _coerce_optional_text(tool_call.get("id"))
        if tool_call_id is not None:
            normalized_call["id"] = tool_call_id
        normalized_calls.append(normalized_call)
    return normalized_calls or None


def _extract_reasoning_from_content(content: Any) -> str | None:
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") not in {"thinking", "redacted_thinking"}:
                continue
            text = item.get("content")
            if text:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return None


def normalize_chat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """follow_your_heart/runtime/message_content.py 逐行移植。"""
    if not isinstance(message, Mapping):
        raise ValueError("each message must be a JSON object")
    role = str(message.get("role") or "")
    if not role:
        raise ValueError("message role is required")

    raw_content = message.get("content")
    normalized: dict[str, Any] = {
        "role": role,
        "content": _coerce_visible_content(raw_content),
    }
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
                    tool_call_payload.get("function") or tool_call_payload.get("name")
                )
        if tool_name is not None:
            normalized["name"] = tool_name

    reasoning_content = _coerce_optional_text(message.get("reasoning_content"))
    reasoning = _coerce_optional_text(message.get("reasoning"))

    if role == "assistant":
        inferred_reasoning = _extract_reasoning_from_content(raw_content)
        if reasoning_content is None and reasoning is None and inferred_reasoning is not None:
            reasoning_content = inferred_reasoning

        if reasoning_content is None and reasoning is None:
            inline_reasoning, visible_content = parse_qwen3_reasoning_output(_coerce_raw_content(raw_content))
            if inline_reasoning is not None:
                reasoning_content = inline_reasoning
                normalized["content"] = visible_content or ""

    if reasoning_content is not None:
        normalized["reasoning_content"] = reasoning_content
    if reasoning is not None:
        normalized["reasoning"] = reasoning
    return normalized


def _discard_reasoning_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _discard_reasoning_content(item)
            for key, item in value.items()
            if key not in ("reasoning_content", "reasoning")
        }
    if isinstance(value, list):
        return [_discard_reasoning_content(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_discard_reasoning_content(item) for item in value)
    return value


def _normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, Mapping):
        return None

    function_payload = tool_call.get("function")
    function_name: str | None = None
    arguments: Any = None

    if isinstance(function_payload, Mapping):
        function_name = _coerce_optional_text(
            function_payload.get("name") or function_payload.get("function")
        )
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

    resolved_arguments: Any = arguments
    if isinstance(resolved_arguments, str):
        try:
            resolved_arguments = json.loads(resolved_arguments)
        except (ValueError, TypeError):
            try:
                import ast
                resolved_arguments = ast.literal_eval(resolved_arguments)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                pass
    if not isinstance(resolved_arguments, Mapping):
        resolved_arguments = resolved_arguments if resolved_arguments is not None else {}

    normalized_call: dict[str, Any] = {
        "function": {
            "name": function_name,
            "arguments": resolved_arguments,
        }
    }
    tool_call_id = _coerce_optional_text(tool_call.get("id"))
    if tool_call_id is not None:
        normalized_call["id"] = tool_call_id
    return normalized_call


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)):
        return []
    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        normalized_call = _normalize_tool_call(tool_call)
        if normalized_call is not None:
            normalized_calls.append(normalized_call)
    return normalized_calls


def _normalize_prompt_message(message: Mapping[str, Any], *, discard_cots: bool = False) -> dict[str, Any]:
    """FYH _normalize_prompt_message 逐行移植（不含 inscope composer 的额外后处理）。"""
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


def _normalize_prompt_messages(messages: Sequence[Mapping[str, Any]], *, discard_cots: bool = False,
                               merge_consecutive_assistants: bool = False,
                               origins: Sequence[int] | None = None,
                               sources: list[list[int]] | None = None) -> list[dict[str, Any]]:
    """``sources`` 非 None 时按输出消息逐个记录其来源（origins 为原始请求下标）。

    合并连续 assistant 时一个输出消息对应多个原始下标，所以是列表而非单个整数。
    """
    normalized = []
    for position, message in enumerate(messages):
        current = _normalize_prompt_message(message, discard_cots=discard_cots)
        origin = origins[position] if origins is not None else position
        if (merge_consecutive_assistants and current['role'] == 'assistant' and
                normalized and normalized[-1]['role'] == 'assistant'):
            previous = normalized[-1]
            for key in ('content', 'reasoning_content', 'reasoning'):
                if current.get(key):
                    previous[key] = previous[key] + '\n' + current[key] if previous.get(key) else current[key]
            if current.get('tool_calls'):
                previous.setdefault('tool_calls', []).extend(current['tool_calls'])
            if sources is not None:
                sources[-1].append(origin)
        else:
            normalized.append(current)
            if sources is not None:
                sources.append([origin])
    return normalized


# --------------------------------------------------------------- 编译入口

def compile_observation_blocks(tokenizer, messages, tools=None, *,
                               max_prompt_tokens: int = 40959) -> ObservationBlocks:
    """canonical simplified-v3 observation-block 编译（ipi-doom build_block_prompt 移植）。

    产物与 ipi-doom featurize.observation_blocks.build_block_prompt 逐项一致：
    同一 compiled text/token_ids、同一 capture positions、同一 block 元数据。
    """
    selected_indices = retained_message_indices(messages, SETTING['retain_n_assistant_turns'])
    selected = [messages[i] for i in selected_indices]
    sources: list[list[int]] = []
    prompt_messages = _normalize_prompt_messages(
        selected, discard_cots=SETTING['discard_cots'],
        merge_consecutive_assistants=SETTING['merge_consecutive_assistants'],
        origins=selected_indices, sources=sources)
    for message in prompt_messages:
        if message['role'] == 'developer':
            message['role'] = 'system'
    first, last = observation_turn_span(prompt_messages)
    position_tokenizer = _PositionTokenizer(tokenizer)
    canonical_text = _apply_chat_template(
        position_tokenizer, prompt_messages, tools=list(tools or ()),
        enable_thinking=SETTING['enable_thinking'])
    token_ids = tuple(_encode_text(position_tokenizer, canonical_text))
    if not token_ids:
        raise ValueError("chat template produced an empty prompt")
    prompt = _Prompt(text=canonical_text, token_ids=token_ids)
    if len(prompt.token_ids) > max_prompt_tokens:
        raise ValueError('Observation-block prompt exceeds token limit; no truncation')
    encoded = position_tokenizer(prompt.text, add_special_tokens=False, return_offsets_mapping=True)
    if tuple(encoded['input_ids']) != prompt.token_ids:
        raise ValueError('Offset tokenizer differs from the captured prompt')
    offsets = encoded['offset_mapping']
    valid = [(i, a, b) for i, (a, b) in enumerate(offsets) if a < b]
    starts = [a for _, a, _ in valid]
    ends = [b for _, _, b in valid]
    positions: list[int] = []
    blocks: list[dict] = []
    tools = tools or []
    marked = deepcopy(prompt_messages)
    observations = []
    nonce = 'IPI_BLOCK_' + uuid4().hex
    while nonce in prompt.text:
        nonce = 'IPI_BLOCK_' + uuid4().hex
    for index in range(first, last + 1):
        if prompt_messages[index]['role'] not in ('tool', 'user'):
            continue
        content = prompt_messages[index].get('content')
        if not isinstance(content, str):
            raise ValueError('Observation-block capture requires text content')
        marker = f'{nonce}_{index}_END'
        # Preserve the template branch for native user-carried tool responses.
        wrapped = (prompt_messages[index]['role'] == 'user' and content.startswith('<tool_response>')
                   and content.endswith('</tool_response>'))
        marked[index]['content'] = '<tool_response>' + marker + '</tool_response>' if wrapped else marker
        observations.append((index, content, marker, wrapped))
    rendered = _render_position_template(prompt, position_tokenizer, marked, tools=tools,
                                         enable_thinking=SETTING['enable_thinking'])
    pieces = []
    locations = []
    cursor = actual = 0
    for index, content, marker, wrapped in observations:
        at = rendered.index(marker, cursor)
        prefix = rendered[cursor:at]
        pieces.append(prefix)
        actual += len(prefix)
        locations.append((index, content, actual - len('<tool_response>') if wrapped else actual))
        payload = content[len('<tool_response>'):-len('</tool_response>')] if wrapped else content
        pieces.append(payload)
        actual += len(payload)
        cursor = at + len(marker)
    pieces.append(rendered[cursor:])
    if ''.join(pieces) != prompt.text:
        raise ValueError('Observation boundary changed chat-template rendering')
    for index, content, begin in locations:
        before = bisect_right(ends, begin) - 1
        if before < 0:
            raise ValueError('Observation has no complete preceding token')
        units = [('empty', 0, 0, valid[before][0])]
        for a, b in block_spans(content):
            at = bisect_right(starts, begin + b - 1) - 1
            if at < 0 or not valid[at][1] <= begin + b - 1 < valid[at][2]:
                raise ValueError('Block end cannot be mapped to a model token')
            token = valid[at][0]
            # BPE often merges punctuation with the following line separator.
            # Whitespace is allowed; following content must never leak in.
            if prompt.text[begin + b:offsets[token][1]].strip():
                raise ValueError('Block boundary crosses a token; no future-content readout')
            units.append(('block', a, b, token))
        for kind, a, b, token in units:
            positions.append(token)
            blocks.append(dict(message_index=index, role=prompt_messages[index]['role'], kind=kind,
                               start=a, end=b, token_position=token))
    # Same BPE token may end more than one unit. Capture once, retain row aliases.
    unique = tuple(sorted(set(positions)))
    slots = {token: i for i, token in enumerate(unique)}
    for block in blocks:
        block['capture_slot'] = slots[block['token_position']]
    return ObservationBlocks(input_ids=prompt.token_ids, text=prompt.text,
                             positions=unique, blocks=tuple(blocks),
                             message_sources=tuple(tuple(source) for source in sources))
