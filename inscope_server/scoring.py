"""服务侧打分流水线：编排 → 特征捕获 → 探针评分。

持有 lifespan 初始化的依赖字典，不加载第二份模型或探针。
本模块保留现有 HTTP 错误映射；传输细节属于 backend，归因属于 localization。
"""
import asyncio
import contextvars
import logging

import torch
from fastapi import HTTPException

from .backend import BackendError
from .batching import bucket_by_length
from .context_profile import compose

log = logging.getLogger(__name__)

# 与 ipi-doom ProbeScorer explain 输出同一句口径。
EXPLAIN_SEMANTICS = "cached cumulative-prefix evidence, not IPI probabilities"


class ScoringPipeline:
    def __init__(self, state, settings, profile):
        self.state = state
        self.settings = settings
        self.default_profile = profile
        # 请求级 profile 覆盖：并发请求互不影响。
        self._local = contextvars.ContextVar("inscope_composer_profile", default=None)
        # 对注入后端也生效；真实 HTTP 分组还有 backend 自身限额。
        self._backend_slots = asyncio.Semaphore(settings.max_inflight)
        # 多风险（Probe Bank）时为本轮请求激活的 {risk: probe}；空表示单探针。
        self.request_probes = None

    def use_profile(self, profile):
        """在本请求上下文中切换编排策略；None 保持启动期默认。"""
        return self._local.set(profile)

    def reset_profile(self, token):
        self._local.reset(token)

    @property
    def profile(self):
        return self._local.get() or self.default_profile

    def active_probes(self):
        """本请求要打分的一组探针；单探针模式即启动期那一个。"""
        selected = self.request_probes
        if selected:
            return list(selected.values())
        return [self.state["probe"]]

    @staticmethod
    def _is_block(probe):
        return getattr(probe, "feature_layout", None) == "observation_blocks"

    def capture_plan(self):
        """所有拟用探针的层并集与位置并集：一次前向覆盖全部探针。

        位置取并集后，各探针从同一份采集结果里按需取自己的位置子集。
        observation_blocks 探针没有静态位置声明，返回 ()，位置由编译产物
        在 prepare 时逐请求给出。
        """
        probes = self.active_probes()
        layers = tuple(dict.fromkeys(layer for probe in probes for layer in probe.capture_layers))
        if any(self._is_block(probe) for probe in probes):
            return layers, ()
        positions = tuple(dict.fromkeys(pos for probe in probes for pos in probe.selected_positions))
        return layers, positions

    def probe_blocks(self, features, probe):
        """从共享采集结果中取本探针声明的层块，并恢复其声明顺序。

        features 为 {layer: [abs_len, hidden]}，位置对齐与否由后端保证：
        后端按解析后的绝对位置采集，单探针时位置并集即该探针全部位置。
        """
        if not isinstance(features, dict):
            return features
        return [features[layer] for layer in probe.capture_layers]

    def score_probe(self, features, probe):
        return probe.score(probe.flatten(self.probe_blocks(features, probe)))

    async def prepare(self, messages, tools=None):
        """messages -> (input_ids, profile 描述, 采集计划)。只做编排，不碰 GPU。

        Composer 与探针在启动期已互相认领，这里的 profile 就是当前探针唯一
        可用的编排规则；detect、audit 与 analyze 共享同一份。
        """
        # 放在 try 之外：profile 配置错误是服务端缺陷，不能被包装成 422。
        blocks = self.profile.position_mode == "blocks"
        try:
            if blocks:
                from .observation_blocks import compile_observation_blocks
                plan = await asyncio.to_thread(
                    compile_observation_blocks, self.state["tokenizer"], messages, tools,
                    max_prompt_tokens=self.settings.max_tokens)
                return list(plan.input_ids), self.profile.describe(), plan
            # Context Composer 按 profile 产出唯一可用的 input_ids；
            # 后端只允许原样转发，不得重新渲染或重新 tokenize。
            ctx = await asyncio.to_thread(compose, self.state["tokenizer"], messages, tools, self.profile)
        except NotImplementedError as exc:
            raise HTTPException(501, "Composer profile is not implemented") from exc
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise HTTPException(422, "Invalid conversation or tool template input") from exc
        ids = list(ctx.input_ids)
        if len(ids) > self.settings.max_tokens:
            raise HTTPException(413, "Input exceeds %d tokens; no truncation applied"
                                % self.settings.max_tokens)
        # 任一在用探针声明的负向位置都要求输入足够长；提前给出明确原因。
        _layers, positions = self.capture_plan()
        need = max((-p for p in positions if p < 0), default=0)
        if len(ids) < need:
            raise HTTPException(422, "Input has %d tokens but the probe needs at least %d"
                                % (len(ids), need))
        return ids, self.profile.describe(), None

    async def score_ids(self, ids, plan=None):
        """单条特征捕获与多探针打分，供 detect 和 readiness 使用。"""
        layers, positions = self.capture_plan()
        if plan is not None:
            positions = plan.positions
        try:
            async with self._backend_slots:
                hidden = await self.state["backend"].capture(ids, layers, positions)
        except TimeoutError as exc:
            raise HTTPException(504, "SGLang request timed out") from exc
        except BackendError as exc:
            log.warning("Shadow backend failed: %s", type(exc).__name__)
            raise HTTPException(503, "Shadow feature backend unavailable") from exc
        try:
            scores, _explained = self.score_features_explained(hidden)
            return scores
        except ValueError as exc:
            raise HTTPException(503, "Invalid InScope features or score") from exc

    def probe_layers(self, probe, features):
        """从共享采集结果取出本探针的层块；features 是 {layer: [hidden]} 或层列表。"""
        if not isinstance(features, dict):
            return features
        return [features[layer] for layer in probe.capture_layers]

    def score_primary(self, features):
        """主探针的 (logit, probability)；保留既有 detect 扁平字段。"""
        probe = self.state["probe"]
        return probe.score(probe.flatten(self.probe_layers(probe, features)))

    async def capture(self, messages, tools=None):
        ids, described, plan = await self.prepare(messages, tools)
        scores = await self.score_ids(ids, plan)
        if isinstance(scores, dict):
            logit, prob = scores[self.state["probe"].spec.probe_id]
        else:
            logit, prob = scores
        active = self.state["probe"]
        score = {"logit": logit, "probability": prob,
                 "threshold": active.threshold,
                 "label": "attack" if prob >= active.threshold else "benign"}
        return score, len(ids), described

    def score_features(self, hidden):
        """单条和批量复用同一套铺平与评分规则；hidden 可以是层列表或按层映射。"""
        scores, _explained = self.score_features_explained(hidden)
        return scores

    def score_features_explained(self, hidden, plan=None, explain=False):
        """评分 + 可选的 block 级 explain 通道；返回 (scores, explained)。

        plan 为 observation_blocks 编译产物时走 block 探针路径（动态位置、
        E4M3 量化往返、attention 打分），不复用 fixed_positions 的 flatten/score。
        """
        probe = self.state["probe"]
        if plan is not None:
            if not self._is_block(probe):
                raise ValueError("Block capture plan requires an observation_blocks probe")
            per_layer = self.probe_layers(probe, hidden)
            rows = torch.stack([block.float() for block in per_layer], dim=1)
            rows = rows.reshape(rows.shape[0], -1)
            scored = probe.score_blocks(rows, plan.blocks, explain=explain)
            return (scored["logit"], scored["probability"]), scored
        probes = self.active_probes()
        if isinstance(hidden, dict):
            return ({candidate.spec.probe_id: self.score_probe(hidden, candidate)
                     for candidate in probes}, None)
        probe0 = probes[0]
        return probe0.score(probe0.flatten(hidden)), None

    def score_probes_all(self, hidden):
        """多风险统一按 {probe_id: (logit, probability)} 返回，单风险也一样。"""
        return {probe.spec.probe_id: self.score_probe(hidden, probe)
                for probe in self.active_probes()}

    async def score_prompts(self, bucket, *, isolate_errors=False):
        """一桶对话逐条编排，再交给批量后端，最后恢复调用方顺序。

        ``isolate_errors=False``（analyze 使用）：任何一条失败都抛出——
        反事实组合缺一条就无法计算 Shapley，必须整体失败。
        ``isolate_errors=True``（detect 批量使用）：逐条记 error，不拖垮整批。

        observation_blocks 输入的 capture positions 逐条不同，批量协议无法
        共享 feature_capture spec；这类条目逐条采集（位置集合不同本来也
        无法合并成一次 GPU forward）。
        """
        prepared, errors = [], {}
        for slot, request_payload in enumerate(bucket):
            try:
                ids, _described, plan = await self.prepare(
                    request_payload["messages"], request_payload.get("tools"))
                prepared.append((slot, ids, plan, bool(request_payload.get("explain"))))
            except Exception as exc:
                if not isolate_errors:
                    raise
                errors[slot] = {"error": type(exc).__name__,
                                "detail": str(getattr(exc, "detail", exc))[:200],
                                "status_code": getattr(exc, "status_code", 500)}
        if not prepared:
            rows = [None] * len(bucket)
            for slot, row in errors.items():
                rows[slot] = row
            return rows
        captured: dict[int, tuple] = {}
        block_items = [item for item in prepared if item[2] is not None]
        plain_items = [item for item in prepared if item[2] is None]
        layers, positions = self.capture_plan()
        try:
            async with self._backend_slots:
                if plain_items:
                    order = sorted(range(len(plain_items)), key=lambda i: len(plain_items[i][1]), reverse=True)
                    ordered_ids = [plain_items[i][1] for i in order]
                    hidden_batches = await self.state["backend"].capture_batch(
                        ordered_ids, layers, positions)
                    if len(hidden_batches) != len(ordered_ids):
                        raise BackendError("Batch feature count does not match inputs")
                    for index, features in zip(order, hidden_batches):
                        captured[plain_items[index][0]] = (features, plain_items[index][2],
                                                           plain_items[index][3])
                for slot, ids, plan, explain in block_items:
                    hidden = await self.state["backend"].capture(ids, layers, plan.positions)
                    captured[slot] = (hidden, plan, explain)
        except (BackendError, TimeoutError) as exc:
            if not isolate_errors:
                status = 504 if isinstance(exc, TimeoutError) else 503
                raise HTTPException(status, "SGLang request timed out" if status == 504
                                    else "InScope feature backend unavailable") from exc
            for slot, _ids, _plan, _explain in prepared:
                errors[slot] = {"error": type(exc).__name__, "detail": str(exc)[:200],
                                "status_code": 504 if isinstance(exc, TimeoutError) else 503}
            rows = [None] * len(bucket)
            for slot, row in errors.items():
                rows[slot] = row
            return rows
        rows = [None] * len(bucket)
        for slot, row in errors.items():
            rows[slot] = row
        for slot, ids, item_plan, item_explain in prepared:
            features, plan, explain = captured[slot]
            try:
                scores, explained = self.score_features_explained(features, plan, explain=explain)
            except ValueError as exc:
                if not isolate_errors:
                    raise HTTPException(503, "Invalid InScope features or score") from exc
                rows[slot] = {"error": "InvalidFeatures", "detail": str(exc)[:200], "status_code": 503}
                continue
            logit, prob = scores if not isinstance(scores, dict) else scores[self.state["probe"].spec.probe_id]
            row = {"logit": logit, "probability": prob,
                   # token 数随行返回：/detect 的单条与批量路径都要给出同一组字段。
                   "input_tokens": len(ids)}
            if explained is not None and "blocks" in explained:
                row.update(blocks=explained["blocks"],
                           coordinate_system="compiled_messages",
                           compiled_context=plan.text,
                           capture_positions=list(plan.positions),
                           importance_semantics=EXPLAIN_SEMANTICS)
            if isinstance(scores, dict):
                row["scores"] = scores
            rows[slot] = row
        return rows

    def _primary(self, scores):
        """取主探针分数；多探针时 scores 是 {probe_id: (logit, prob)}。"""
        if not isinstance(scores, dict):
            return scores
        primary = self.state["probe"]
        if primary.spec.probe_id not in scores:
            raise ValueError("Primary probe missing from capture group")
        return scores[primary.spec.probe_id]

    async def detect_batch(self, requests, *, concurrency=None):
        """批量检测：分桶合并采集，逐条独立评分，失败逐条隔离。

        返回与输入同序的行；每行要么有分数，要么有 error，二者必居其一。
        单条失败不会让整批失败——批量接口的正确语义。
        """
        if not requests:
            return []
        active = self.state["probe"]

        def threshold_of(probe_id):
            for probe in self.active_probes():
                if probe.spec.probe_id == probe_id:
                    return probe.threshold
            return active.threshold

        limit = concurrency or self.settings.detect_batch or 8

        def chars(item):
            return sum(len(str(m.get("content") or "")) for m in item.messages)

        buckets = bucket_by_length(requests, length_of=chars, max_batch=limit)

        async def run(bucket):
            payloads = [{"messages": r.messages, "tools": r.tools,
                         "explain": bool(getattr(r, "explain", False))} for _, r in bucket]
            rows = await self.score_prompts(payloads, isolate_errors=True)
            out = []
            for (slot, _req), row in zip(bucket, rows):
                if "error" in row:
                    out.append((slot, row))
                else:
                    primary_id = active.spec.probe_id
                    entry = {"logit": row["logit"],
                             "probability": row["probability"],
                             "threshold": threshold_of(primary_id),
                             "label": ("attack"
                                       if row["probability"] >= threshold_of(primary_id)
                                       else "benign"),
                             "input_tokens": row["input_tokens"]}
                    for key in ("blocks", "coordinate_system", "compiled_context",
                                "capture_positions", "importance_semantics"):
                        if key in row:
                            entry[key] = row[key]
                    if "scores" in row:
                        entry["scores"] = row["scores"]
                    out.append((slot, entry))
            return out

        results = [None] * len(requests)
        # 桶串行消费，避免为整个批量同时创建所有后端任务。
        for bucket in buckets:
            for slot, row in await run(bucket):
                results[slot] = row
        return results

    async def collect(self, messages, tools=None, *, explain=False):
        """单条：一次采集 + 多探针打分，返回 (主探针分数行, 全部探针分数, token 数)。

        observation_blocks 探针（plan 非空）按编译产物打分，explain 时返回
        block 级 attention 通道；多风险 Probe Bank 路径与 block 探针不共存。
        """
        ids, described, plan = await self.prepare(messages, tools)
        layers, positions = self.capture_plan()
        if plan is not None:
            positions = plan.positions
        try:
            async with self._backend_slots:
                hidden = await self.state["backend"].capture(ids, layers, positions)
        except TimeoutError as exc:
            raise HTTPException(504, "SGLang request timed out") from exc
        except BackendError as exc:
            log.warning("Shadow backend failed: %s", type(exc).__name__)
            raise HTTPException(503, "Shadow feature backend unavailable") from exc
        primary = self.state["probe"]
        if plan is not None:
            try:
                (logit, probability), explained = self.score_features_explained(
                    hidden, plan, explain=explain)
            except ValueError as exc:
                raise HTTPException(503, "Invalid InScope features or score") from exc
            row = {"logit": logit, "probability": probability,
                   "threshold": primary.threshold,
                   "label": "attack" if probability >= primary.threshold else "benign",
                   "input_tokens": len(ids)}
            if explain and explained is not None and "blocks" in explained:
                row.update(blocks=explained["blocks"],
                           coordinate_system="compiled_messages",
                           compiled_context=plan.text,
                           capture_positions=list(plan.positions),
                           importance_semantics=EXPLAIN_SEMANTICS)
            return row, None, described
        try:
            # 只要本请求激活了探针组（哪怕只有一个风险），就按组返回全部分数，
            # 以便上层组装 per_risk；单探针且无风险注册时才保持原返回值。
            probed = self.request_probes is not None
            scores = self.score_probes_all(hidden)
        except ValueError as exc:
            raise HTTPException(503, "Invalid InScope features or score") from exc
        logit, probability = self._primary(scores)
        row = {"logit": logit, "probability": probability,
               "threshold": primary.threshold,
               "label": "attack" if probability >= primary.threshold else "benign",
               "input_tokens": len(ids)}
        if probed:
            row["scores"] = scores
        return row, (scores if probed else None), described

