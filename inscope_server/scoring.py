"""服务侧打分流水线：编排 → 特征捕获 → 探针评分。

持有 lifespan 初始化的依赖字典，不加载第二份模型或探针。
本模块保留现有 HTTP 错误映射；传输细节属于 backend，归因属于 localization。
"""
import asyncio
import logging

from fastapi import HTTPException

from .backend import BackendError
from .batching import bucket_by_length
from .context_profile import compose

log = logging.getLogger(__name__)


class ScoringPipeline:
    def __init__(self, state, settings, profile):
        self.state = state
        self.settings = settings
        self.profile = profile

    async def prepare(self, messages, tools=None):
        """messages -> (input_ids, profile 描述)。只做编排，不碰 GPU。

        Composer 与探针在启动期已互相认领，这里的 profile 就是当前探针唯一
        可用的编排规则；detect 与 analyze 共享同一份；audit 尚未实现。
        """
        try:
            # Context Composer 按 profile 产出唯一可用的 input_ids；
            # 后端只允许原样转发，不得重新渲染或重新 tokenize。
            ctx = await asyncio.to_thread(compose, self.state["tokenizer"], messages, tools, self.profile)
        except NotImplementedError as exc:
            raise HTTPException(501, "Composer profile is not implemented") from exc
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise HTTPException(422, "Invalid conversation or tool template input") from exc
        active = self.state["probe"]
        ids = list(ctx.input_ids)
        if len(ids) > self.settings.max_tokens:
            raise HTTPException(413, "Input exceeds %d tokens; no truncation applied"
                                % self.settings.max_tokens)
        # 探针声明的负向位置要求输入至少有这么长；提前给出明确原因。
        need = max((-p for p in active.selected_positions if p < 0), default=0)
        if len(ids) < need:
            raise HTTPException(422, "Input has %d tokens but the probe needs at least %d"
                                % (len(ids), need))
        return ids, self.profile.describe()

    async def score_ids(self, ids):
        """单条特征捕获与打分，供 detect 和 readiness 使用。"""
        active = self.state["probe"]
        try:
            hidden = await self.state["backend"].capture(
                ids, active.capture_layers, active.selected_positions)
        except TimeoutError as exc:
            raise HTTPException(504, "SGLang request timed out") from exc
        except BackendError as exc:
            log.warning("Shadow backend failed: %s", type(exc).__name__)
            raise HTTPException(503, "Shadow feature backend unavailable") from exc
        try:
            return self.score_features(hidden)
        except ValueError as exc:
            raise HTTPException(503, "Invalid InScope features or score") from exc

    async def capture(self, messages, tools=None):
        ids, described = await self.prepare(messages, tools)
        logit, prob = await self.score_ids(ids)
        active = self.state["probe"]
        score = {"logit": logit, "probability": prob,
                 "threshold": active.threshold,
                 "label": "attack" if prob >= active.threshold else "benign"}
        return score, len(ids), described

    def score_features(self, hidden):
        """单条和批量复用同一探针的铺平与评分规则。"""
        active = self.state["probe"]
        return active.score(active.flatten(hidden))

    async def score_prompts(self, bucket, *, isolate_errors=False):
        """一桶对话逐条编排，再交给批量后端，最后恢复调用方顺序。

        ``isolate_errors=False``（analyze 使用）：任何一条失败都抛出——
        反事实组合缺一条就无法计算 Shapley，必须整体失败。
        ``isolate_errors=True``（detect 批量使用）：逐条记 error，不拖垮整批。
        """
        prepared, errors = [], {}
        for slot, request_payload in enumerate(bucket):
            try:
                ids, _described = await self.prepare(
                    request_payload["messages"], request_payload.get("tools"))
                prepared.append((slot, ids))
            except Exception as exc:
                if not isolate_errors:
                    raise
                errors[slot] = {"error": type(exc).__name__, "detail": str(exc)[:200]}
        if not prepared:
            rows = [None] * len(bucket)
            for slot, row in errors.items():
                rows[slot] = row
            return rows
        # 保留现有桶内长度排序；通过原下标恢复结果，不作加速保证。
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i][1]), reverse=True)
        ordered_ids = [prepared[i][1] for i in order]
        active = self.state["probe"]
        try:
            hidden_batches = await self.state["backend"].capture_batch(
                ordered_ids, active.capture_layers, active.selected_positions)
        except Exception as exc:
            if not isolate_errors:
                raise
            for slot, _ids in prepared:
                errors[slot] = {"error": type(exc).__name__, "detail": str(exc)[:200]}
            rows = [None] * len(bucket)
            for slot, row in errors.items():
                rows[slot] = row
            return rows
        rows = [None] * len(bucket)
        for slot, row in errors.items():
            rows[slot] = row
        for position, ids_features in zip(order, hidden_batches):
            slot = prepared[position][0]
            try:
                logit, prob = self.score_features(ids_features)
            except ValueError as exc:
                if not isolate_errors:
                    raise HTTPException(503, "Invalid InScope features or score") from exc
                rows[slot] = {"error": "InvalidFeatures", "detail": str(exc)[:200]}
                continue
            # token 数随行返回：/detect 的单条与批量路径都要给出同一组字段。
            rows[slot] = {"logit": logit, "probability": prob,
                          "input_tokens": len(prepared[position][1])}
        return rows

    async def detect_batch(self, requests, *, concurrency=None):
        """批量检测：分桶合并采集，逐条独立评分，失败逐条隔离。

        返回与输入同序的行；每行要么有分数，要么有 error，二者必居其一。
        单条失败不会让整批失败——批量接口的正确语义。
        """
        if not requests:
            return []
        active = self.state["probe"]
        limit = concurrency or self.settings.detect_batch or 8

        def chars(item):
            return sum(len(str(m.get("content") or "")) for m in item.messages)

        buckets = bucket_by_length(requests, length_of=chars, max_batch=limit)

        async def run(bucket):
            payloads = [{"messages": r.messages, "tools": r.tools} for _, r in bucket]
            rows = await self.score_prompts(payloads, isolate_errors=True)
            out = []
            for (slot, _req), row in zip(bucket, rows):
                if "error" in row:
                    out.append((slot, row))
                else:
                    out.append((slot, {"logit": row["logit"],
                                       "probability": row["probability"],
                                       "threshold": active.threshold,
                                       "label": ("attack" if row["probability"] >= active.threshold
                                                 else "benign"),
                                       "input_tokens": row["input_tokens"]}))
            return out

        results = [None] * len(requests)
        for done in await asyncio.gather(*(run(b) for b in buckets)):
            for slot, row in done:
                results[slot] = row
        return results

