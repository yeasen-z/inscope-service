"""轨迹审计：沿执行过程组织多次评分，回答"最早在哪一步检出"。

对外入口只有一个
----------------
调用方把**整条轨迹**一次交给 ``audit_trajectory``，不必自己切窗口。
服务内部按检查点逐个回放 ``messages[:i+1]`` 并评分——切点、跳步、
未来消息的边界都由这里保证，调用方无法切错。

为什么内部是逐条而不是一次整条
------------------------------
* 逐步窗口是**严格递增的前缀链**：第 i 步是第 i+1 步的前缀。
* 引擎的 radix cache 会命中这条链上已经算过的前缀，每步只真正计算
  新追加的那几条消息（实测：公共前缀 100 token 时 cached=100）。
  因此逐条回放的累计算力≈一次全长 forward，并不比整条输入吃亏。
* 换成"整条输入 + 多点取值"会牺牲语义：探针训练时的位置都落在
  对话末尾，从序列中途取到的 hidden state 不保证有同等判别力。

三条不可违反的语义
------------------
1. **不泄未来**：第 i 步的评分只使用 messages[:i+1]，绝不包含之后的内容。
2. **错误不是阴性**：某步评分失败就记 failed 并终止，不把错误吞成"未检出"。
3. **检出不是因果**：首次检出指"截至该步的累计上下文已越过阈值"，
   不等于已证明攻击在该步发生。

历史截取按当前已加载探针的 composer 规则执行：探针只认"开场 + 最近 N 个
动作块及其后消息"，因此检查点必须落在被保留的窗口里，否则该步的观察
根本不会进入模型输入，评分就失去意义。
"""
from __future__ import annotations

import asyncio

__all__ = ["audit_trajectory", "checkpoint_indices"]


def checkpoint_indices(messages, roles=("tool", "user"), start_index=0):
    """选出要评分的消息下标：只挑指定角色，且从 start_index 起算。

    返回升序下标列表。空轨迹返回空列表，而不是抛错——调用方据此返回
    "没有可审计的检查点"，这本身就是有意义的结论。
    """
    wanted = set(roles)
    return [i for i in range(start_index, len(messages))
            if messages[i].get("role") in wanted]


async def audit_trajectory(messages, score_at, *, tools=None, roles=("tool", "user"),
                           start_index=0, case_id="audit"):
    """整条轨迹进，审计时间线出。

    ``score_at(payload) -> {"logit": float, "probability": float, "label": str}``
    由服务注入：它负责编排 + 采集 + 打分，serializer 与探针细节都不在本模块。

    **串行**推进检查点，不并发：相邻窗口是前缀链，串行才能让引擎的
    radix cache 复用上一步算过的前缀。并发会同时打出 N 个互不相同的
    前缀，缓存互相打不中，反而更慢。

    返回值刻意区分三种"没检出"：
      * 全部检查点都低于阈值 → ``status="clear"``
      * 某些检查点没跑或跑失败 → ``status="incomplete"``（不是 clear）
      * 至少一个检查点越过阈值 → ``status="flagged"``
    """
    checkpoints = checkpoint_indices(messages, roles, start_index)
    steps = []
    first_flagged = None
    failed_at = None

    for index in checkpoints:
        # 关键：只截到该检查点为止，后面的消息一步都不能带进去。
        window = messages[:index + 1]
        try:
            scored = await score_at({"messages": window, "tools": tools})
        except Exception as exc:  # 记录失败位置并终止，绝不当作阴性
            failed_at = {"observation_index": index, "error": type(exc).__name__,
                         "detail": str(exc)[:300]}
            break
        steps.append({"observation_index": index,
                      "role": messages[index].get("role"),
                      "logit": scored["logit"],
                      "probability": scored["probability"],
                      "label": scored["label"]})
        if first_flagged is None and scored["label"] == "attack":
            first_flagged = index

    total = len(checkpoints)
    if failed_at is not None:
        status = "incomplete"
    elif first_flagged is not None:
        status = "flagged"
    elif steps and len(steps) == total:
        status = "clear"
    else:
        status = "incomplete"

    flagging = [s["observation_index"] for s in steps if s["label"] == "attack"]
    return {
        "schema": "inscope.audit.v1",
        "case_id": case_id,
        "status": status,
        "checkpoint_count": total,
        "scored_count": len(steps),
        "steps": steps,
        "first_flagged_index": first_flagged,
        "flagged_indices": flagging,
        "failed_at": failed_at,
        # 把边界写进响应，避免下游把"检出"读成"攻击发生"。
        "semantics": {
            "flagged_means": "cumulative context up to this index crossed the probe threshold",
            "not_claimed": "attack occurred here, or that this step caused later behavior",
            "future_messages_excluded": True,
        },
    }
