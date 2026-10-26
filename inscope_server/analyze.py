"""定位算法的服务适配入口。

职责有三件，且只做这三件：
1. 校验 observation 下标，把越界索引统一成 ValueError（HTTP 422）。
2. 把 ``mode="auto"`` 按单元数解析成 exact / hierarchical，让调用方不必
   自己算组合预算，也就不会撞上 "exceed coalition budget"。
3. 透传给上游 ``analyze_observation``；切分、枚举与归因都留在 localization 包。

模式解析规则
------------
``mode="auto"`` 时，服务按**切出来的单元数**自行决定模式与每层组数，
调用方不必理解组合预算，也就不会撞上 "exceed coalition budget"。

exact 需要枚举 2^n 个子集。默认预算 256 意味着 n>=9 必然超支。因此：

    n <= 8   → exact，一次全枚举（2^8 = 256，正好用满预算，最精确）
    n >= 9   → hierarchical，分级展开

分级展开时每层组数自适应，而不是固定值：

    group_size 上限 6（2^6 = 64 组合/层），保证 256 预算至少能展开 4 层；
    若固定 8，单层就吃掉全部 256，第二层无预算可用，退化成"只做一次粗分"。

显式 mode 保留；仅 mode="auto" 时，group_size（即使显式传入）也由服务
按单元数与预算重新规划。显式 exact / hierarchical 时保留调用方的 group_size。
"""
from .batching import bucket_by_length, make_batched_scorer, make_score_batch
from .localization.api import analyze_observation
from .localization.segments import Span, split_spans

__all__ = [
    "Span", "split_spans", "analyze", "analyze_observation", "resolve_mode",
    "plan_analysis", "EXACT_UNIT_LIMIT", "MAX_GROUP_SIZE",
    "make_score_batch", "bucket_by_length", "make_batched_scorer",
]

# 2^8 = 256，与默认 max_coalitions 对齐；超过就必须分级展开。
EXACT_UNIT_LIMIT = 8
# 每层组数上限：2^6 = 64，保证 256 预算至少能走 4 层。
MAX_GROUP_SIZE = 6


def plan_analysis(unit_count, max_coalitions=256):
    """按单元数规划分析方式，返回 ``(mode, group_size, reason)``。

    1. 段数少到能一次枚举完 → exact，最精确。
    2. 否则分级展开，组数取能留出 >=2 层预算的最大值（上限 6）。
    """
    if max_coalitions < 4:
        raise ValueError("max_coalitions must be >= 4")
    affordable = max_coalitions.bit_length() - 1
    if unit_count <= min(EXACT_UNIT_LIMIT, affordable):
        # exact 不使用组数切分，但上游统一要求 group_size >= 2。
        size = max(2, min(unit_count, MAX_GROUP_SIZE))
        return ("exact", size,
                "%d units, 2^%d <= %d, full enumeration"
                % (unit_count, unit_count, max_coalitions))
    for size in range(MAX_GROUP_SIZE, 1, -1):
        layers = max_coalitions // (2 ** size)
        if layers >= 2:
            return ("hierarchical", size,
                    "%d units > %d, %d groups/level gives %d layers within %d coalitions"
                    % (unit_count, min(EXACT_UNIT_LIMIT, affordable), size,
                       layers, max_coalitions))
    return ("hierarchical", 2,
            "%d units, budget too tight, smallest grouping" % unit_count)


def resolve_mode(mode, unit_count, max_coalitions=256):
    """把 auto 解析成具体模式；显式模式原样返回。

    返回 ``(mode, reason)``，reason 用于在响应里说明"为什么这样选"，
    避免调用方看到一个自己没传过的 mode 时困惑。
    """
    if mode != "auto":
        return mode, "explicit"
    resolved, _size, reason = plan_analysis(unit_count, max_coalitions)
    return resolved, reason


async def analyze(replay_request, observation_index, score_batch, **kwargs):
    """包装上游 ``analyze_observation``，补上越界校验与 auto 模式解析。

    上游直接 ``snapshot['messages'][index]``，越界会抛 IndexError；HTTP 层
    只把 ValueError 映射成 422，不转换就会变成 500，调用方看不出是自己
    传错了 observation_index。
    """
    messages = replay_request.get("messages") if isinstance(replay_request, dict) else None
    if not isinstance(messages, list):
        raise ValueError("request must carry a messages list")
    if not 0 <= observation_index < len(messages):
        raise ValueError(
            "observation_index %d is outside the %d-message request"
            % (observation_index, len(messages)))

    # 只在 auto 时数一次单元；显式模式保持上游行为，不做额外切分。
    if kwargs.get("mode") == "auto":
        text = messages[observation_index].get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("observation content must be nonempty text")
        granularity = kwargs.get("granularity", "paragraph")
        # spans 由调用方显式给出时，单元数就是 spans 的长度。
        spans = kwargs.get("spans")
        unit_count = len(spans) if spans is not None else len(split_spans(text, granularity))
        resolved, size, reason = plan_analysis(
            unit_count, kwargs.get("max_coalitions", 256))
        kwargs = {**kwargs, "mode": resolved, "group_size": size}
        result = await analyze_observation(
            replay_request, observation_index, score_batch, **kwargs)
        result["mode_requested"] = "auto"
        result["mode_resolved"] = resolved
        result["group_size_resolved"] = size
        result["mode_reason"] = reason
        return result

    return await analyze_observation(replay_request, observation_index, score_batch, **kwargs)
