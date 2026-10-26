"""Conservative shape normalization: never infer history or invent tool-call IDs."""
import copy
import json

ROLE_ALIASES = {"human": "user", "ai": "assistant", "function": "tool",
                "tool_result": "tool", "tool-result": "tool"}


def _reject_nonfinite(value):
    raise ValueError("Non-finite tool argument JSON: " + value)


def normalize_payload(payload, *, extra_fields=()):
    """规范化 messages/tools 的形状。

    ``extra_fields`` 供 /analyze 这类接口使用：它们在同一份 chat 请求上
    追加自己的参数，不在这里被当成非法字段。传进来的字段原样保留。
    """
    changes = []
    allowed = {"messages", "tools"} | set(extra_fields)
    if isinstance(payload, list):
        payload = {"messages": payload}
        changes.append("request.list_to_object")
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Expected only %s" % ", ".join(sorted(allowed)))
    result = copy.deepcopy(payload)
    messages = result.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise ValueError("messages must contain 1..128 objects")
    for i, m in enumerate(messages):
        prefix = f"messages[{i}]"
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise ValueError("Every message needs a role")
        role = m["role"].strip().lower()
        role = ROLE_ALIASES.get(role, role)
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError("Unknown role; cannot infer message source")
        if role != m["role"]:
            changes.append(prefix + ".role")
        m["role"] = role
        if "content" not in m and "text" in m:
            m["content"] = m.pop("text")
            changes.append(prefix + ".text_to_content")
        if isinstance(m.get("content"), list):
            blocks = m["content"]
            if any(not isinstance(b, dict) or b.get("type") not in ("text", "input_text", "output_text")
                   or not isinstance(b.get("text"), str) for b in blocks):
                raise ValueError("Only plain text blocks can be normalized")
            m["content"] = "".join(b["text"] for b in blocks)
            changes.append(prefix + ".text_blocks")
        if "reasoning" in m and "reasoning_content" not in m:
            m["reasoning_content"] = m["reasoning"]
            changes.append(prefix + ".reasoning_alias")
        if m.get("function_call") is not None:
            if role != "assistant" or m.get("tool_calls"):
                raise ValueError("Ambiguous legacy function_call")
            m["tool_calls"] = [m.pop("function_call")]
            changes.append(prefix + ".function_call")
        if "tool_calls" in m:
            if role != "assistant":
                raise ValueError("tool_calls require assistant role")
            calls = m["tool_calls"]
            if isinstance(calls, dict):
                calls = [calls]
                changes.append(prefix + ".tool_calls_list")
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be objects")
            normalized = []
            for j, call in enumerate(calls):
                if not isinstance(call, dict) or call.get("type", "function") != "function":
                    raise ValueError("Only function tool calls are supported")
                function = call.get("function", call)
                if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
                    raise ValueError("Tool function name is required")
                arguments = function.get("arguments", function.get("args"))
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
                    changes.append(prefix + f".tool_calls[{j}].arguments_object")
                if not isinstance(arguments, str):
                    raise ValueError("Tool arguments must be a JSON string or object")
                try:
                    parsed = json.loads(arguments, parse_constant=_reject_nonfinite)
                except (ValueError, TypeError) as exc:
                    raise ValueError("Malformed tool argument JSON") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("Tool arguments must encode an object")
                canonical = {"type": "function", "function": {"name": function["name"], "arguments": arguments}}
                if "id" in call:
                    if not isinstance(call["id"], str):
                        raise ValueError("Tool call ID must be a string")
                    canonical["id"] = call["id"]
                if canonical != call:
                    changes.append(prefix + f".tool_calls[{j}].shape")
                normalized.append(canonical)
            m["tool_calls"] = normalized
        if m.get("content") is None and role == "assistant":
            if m.get("tool_calls"):
                if "content" not in m:
                    changes.append(prefix + ".empty_call_content")
                m["content"] = None
            elif isinstance(m.get("reasoning_content"), str):
                m["content"] = ""
                changes.append(prefix + ".reasoning_only_content")
            else:
                raise ValueError("Assistant without text, reasoning, or calls")
        elif not isinstance(m.get("content"), str):
            raise ValueError("Message text is required; cannot infer missing content")
    tools = result.get("tools")
    if tools is None:
        tools = []
        changes.append("tools.default_empty")
    elif isinstance(tools, dict):
        tools = [tools]
        changes.append("tools.object_to_list")
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    canonical_tools = []
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise ValueError("Only function tools are supported")
        function = tool.get("function", tool)
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
            raise ValueError("Tool definition name is required")
        if "parameters" in function and not isinstance(function["parameters"], dict):
            raise ValueError("Tool parameters must be an object")
        canonical = {"type": "function", "function": function}
        if canonical != tool:
            changes.append(f"tools[{i}].shape")
        canonical_tools.append(canonical)
    result["tools"] = canonical_tools
    return result, changes


def split_detect_payload(payload):
    """把 /detect 请求体拆成 (请求列表, tools, 是否批量)。

    默认批量：``messages`` 为二维列表即多条；一维列表视为单条，
    保持旧调用方不受影响。``tools`` 由整批共享（同一套工具定义）。
    """
    if not isinstance(payload, dict):
        raise ValueError("body must be an object")
    if "messages" not in payload:
        raise ValueError("missing messages")
    messages = payload["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    tools = payload.get("tools")
    extra = set(payload) - {"messages", "tools"}
    if extra:
        raise ValueError("unexpected fields: %s" % ", ".join(sorted(extra)))
    # 二维（元素是列表）→ 批量；否则单条
    if all(isinstance(item, list) for item in messages):
        requests = [{"messages": item, "tools": tools} for item in messages]
        return requests, tools, True
    if any(isinstance(item, list) for item in messages):
        raise ValueError("messages is neither a single message list nor a batch")
    return [{"messages": messages, "tools": tools}], tools, False
