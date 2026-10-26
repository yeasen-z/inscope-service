"""轨迹审计：沿执行过程组织多次评分，回答"最早在哪一步检出"。

对外入口只有一个
----------------
调用方把**整条轨迹**一次交给 ``audit_trajectory``，不必自己切窗口。
服务内部按检查点逐个回放 ``messages[:i+1]`` 并评分——切点、跳步、
未来消息的边界都由这里保证，调用方无法切错。

为什么内部是逐条而不是一次整条
------------------------------
* 原始消息窗口是逐步增长的前缀；实际 token 输入还要经过 profile 编排，
  模板末尾与选窗规则可能改变可共享的 token 前缀。
* 串行推进为引擎复用已计算前缀提供机会，但缓存命中取决于实际输入、
  缓存驻留和引擎状态；不保证每步只计算新增 token，也不保证累计算力
  等同一次全长 forward。
* 换成"整条输入 + 多点取值"会牺牲语义：探针训练时的位置都落在
  对话末尾，从序列中途取到的 hidden state 不保证有同等判别力。

三条不可违反的语义
------------------
1. **不泄未来**：第 i 步的评分只使用 messages[:i+1]，绝不包含之后的内容。
2. **错误不是阴性**：某步评分失败就记 failed 并终止，不把错误吞成"未检出"。
3. **检出不是因果**：首次检出指"截至该步的累计上下文已越过阈值"，
   不等于已证明攻击在该步发生。

历史截取按当前已加载探针的 composer profile 执行：A6 profile 使用
retain_n_assistant_turns=-1，保留截至当前检查点的全部历史与 CoT；
A5 简化 profile 才使用"开场 + 最近 N 个动作块及其后消息"的选窗规则。
不能把某个 profile 的窗口规则推广到全部探针；检查点是否进入最终模型
输入，应依据实际 profile 和编排结果判断。
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

    **串行**推进检查点，不并发，保持确定的评分与失败终止顺序，并为
    引擎复用相邻输入的公共前缀提供机会。实际缓存命中由 profile 编排、
    模板与引擎状态决定；这里不承诺并发一定更慢或固定的缓存收益。

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
