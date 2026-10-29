"""HTTP 入口：依赖生命周期、鉴权、请求校验和结果封装。"""
import asyncio
from contextlib import asynccontextmanager
import hmac
import logging

from functools import wraps
import json
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from inscope_contract import (
    AnalyzeRequest,
    AuditRequest,
    DetectionRequest,
    ENDPOINTS,
    MODEL_LABEL,
    RISK_TYPE,
)

from inscope_contract.schemas import DetectionResponse, DetectionBatchRequest

from .analyze import analyze as localize_observation
from .audit import audit_trajectory
from .batching import make_batched_scorer
from .backend import InScopeBackend
from .config import KNOWN_RISKS, Settings, validate_model
from .context_profile import bind_profile, get_profile
from .normalization import normalize_payload, split_detect_payload
from .probe import load_probe
from .scoring import ScoringPipeline

# 就绪探针必须长到能覆盖探针声明的负向位置；不足时由 capture 报 422。
log = logging.getLogger(__name__)
READY_PROMPT = "Hello, please confirm the detection pipeline is operational."
# observation_blocks 探针要求存在 assistant 动作 + 其后的观察，普通单句不满足。
READY_PROMPT_BLOCKS = [
    {"role": "user", "content": "Please check the server status."},
    {"role": "assistant", "content": None,
     "tool_calls": [{"id": "call_ready", "type": "function",
                     "function": {"name": "check_status", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_ready", "name": "check_status",
     "content": "All services are running normally."},
]


def create_app(settings=None, *, tokenizer=None, probe=None, backend=None, probes=None):
    """Dependencies can be injected for CPU-only protocol tests."""
    settings = settings or Settings.from_env()
    # profile 必须在启动期解析：未知 id 直接失败，而不是首个请求才暴露。
    profile = get_profile(settings.composer_profile)

    state = {"probe": probe, "probes": dict(probes or {}), "tokenizer": tokenizer, "backend": backend}

    @asynccontextmanager
    async def lifespan(app):
        if state["tokenizer"] is None:
            validate_model(settings.model_path)
            from transformers import AutoTokenizer
            state["tokenizer"] = AutoTokenizer.from_pretrained(
                settings.model_path, local_files_only=True, trust_remote_code=False)
        # 探针结构与层号完全由 checkpoint 自描述决定，不在此处写死任何一套。
        if state["probe"] is None:
            state["probe"] = load_probe(settings.probe_checkpoint)
        active = state["probe"]
        # profile 与探针必须互相认领，避免「用 A 的编排喂 B 的探针」。
        bind_profile(profile, active.spec.probe_id)
        # profile 声明的位置必须与探针一致，否则特征会错位且不报错。
        wanted = profile.composer.selected_positions if profile.composer else None
        if wanted is not None and tuple(wanted) != tuple(active.selected_positions):
            raise ValueError(
                "composer profile %r expects positions %s but probe %r declares %s"
                % (profile.profile_id, list(wanted), active.spec.probe_id,
                   list(active.selected_positions)))
        # 多风险：每个风险一份 checkpoint，层可以不同；任一失败即拒绝启动。
        if not state["probes"] and getattr(settings, "probes", None):
            if probe is not None:
                raise ValueError("Injected probe cannot be combined with INSCOPE_PROBES")
            state["probes"] = {risk: load_probe(path) for risk, path in settings.probes.items()}
        if state["probes"]:
            # 位置决定编排出的 input_ids 是否被探针正确读取，必须与 profile 一致。
            for risk, candidate in state["probes"].items():
                if wanted is not None and tuple(wanted) != tuple(candidate.selected_positions):
                    raise ValueError(
                        "probe %r declares positions %s but composer profile %r expects %s"
                        % (risk, list(candidate.selected_positions), profile.profile_id,
                           list(wanted)))
            state.setdefault("primary_risk", settings.primary_risk)
        # observation_blocks 探针的采集/打分路径与 fixed_positions 探针不共享，
        # 多风险共实例会让 block 探针拿到按错误规则编排的输入；单实例单探针。
        loaded_all = [state["probe"], *(state["probes"] or {}).values()]
        if len(loaded_all) > 1 and any(
                getattr(candidate, "feature_layout", None) == "observation_blocks"
                for candidate in loaded_all):
            raise ValueError(
                "observation_blocks probes must run in a single-risk instance; "
                "do not combine with INSCOPE_PROBES")
        state["backend"] = state["backend"] or InScopeBackend(settings)
        try:
            yield
        finally:
            await state["backend"].close()

    app = FastAPI(title="InScope detection API", version="0.6.0", lifespan=lifespan)
    app.state.inscope = state
    inflight = 0

    def authenticate(request):
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth.encode(), ("Bearer " + settings.api_key).encode()):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    pipeline = ScoringPipeline(state, settings, profile)

    def bank_view():
        """本轮请求激活的风险->探针；仅当注册了额外风险时才输出 bank 字段。"""
        return getattr(pipeline, "request_probes", None) or {}

    def all_probes(view):
        """已加载的全部风险->探针（主探针 + 额外注册表）。"""
        if isinstance(view, dict) and "probe" in view and "backend" in view:
            registered = view.get("probes") or {}
            return {getattr(settings, "primary_risk", "ipi"): view["probe"], **registered}
        return dict(view or {})

    def select_risks(view, requested):
        """请求级风险过滤：请求集合必须落在已加载集合内，否则 422。

        服务可以比调用方多加载风险；但调用方不能要求服务未加载的风险，
        否则会得到伪造分数或静默缺失，两者都不可接受。
        """
        available = all_probes(view)
        if isinstance(view, dict) and "probe" in view and "backend" in view:
            # 额外的 disabled_risks 需要知道全部风险全集，因此单风险也输出 bank 字段。
            if not requested or set(requested) >= set(available):
                return available
        if not requested:
            return available
        missing = [risk for risk in requested if risk not in available]
        if missing:
            raise HTTPException(422, "Requested risks are not loaded: %s (available: %s)"
                                % (", ".join(missing), ", ".join(available)))
        return {risk: probe for risk, probe in available.items() if risk in set(requested)}

    def set_bank(view):
        """开启本轮请求的多风险打分：view 为 {risk: probe} 或含注册表的 state。"""
        pipeline.request_probes = view or None

    def bank_payload(scores, input_tokens, probes):
        """Probe Bank 兼容视图：per_risk/per_entry 只来自真实打分过的探针。"""
        probes = dict(probes)
        per_risk, per_entry, triggered = {}, {}, []
        for risk, candidate in probes.items():
            _logit, probability = scores[candidate.spec.probe_id]
            per_entry[risk] = probability
            per_risk[risk] = {"score": probability,
                              "threshold": candidate.threshold,
                              "flagged": probability >= candidate.threshold}
            if per_risk[risk]["flagged"]:
                triggered.append(risk)
        enabled = list(probes)
        return {"enabled_risks": enabled,
                "disabled_risks": [risk for risk in ("harmful", "prompt_leakage", "ipi")
                                   if risk not in enabled],
                "overall": any(row["flagged"] for row in per_risk.values()),
                "profile": "balanced",
                "per_risk": per_risk,
                "per_entry": per_entry,
                "triggered_entries": triggered,
                "detector_model": settings.detector_model or MODEL_LABEL,
                "bank_version": "sha256:" + "".join(
                    candidate.sha256 or "" for candidate in probes.values())[:32],
                "input_tokens": input_tokens}

    def request_profile(name):
        """解析请求级覆盖策略；None 表示沿用启动期默认。"""
        if name is None:
            return None
        try:
            candidate = get_profile(name)
            return bind_profile(candidate, state["probe"].spec.probe_id)
        except ValueError as exc:
            raise HTTPException(422, "Invalid composer_profile: %s" % exc) from exc

    def bounded(deadline):
        """所有评分入口共享请求限额；读取 body 也计入总 deadline。"""
        def decorate(fn):
            @wraps(fn)
            async def wrapped(request: Request):
                nonlocal inflight
                authenticate(request)
                if inflight >= settings.max_inflight:
                    raise HTTPException(429, "Too many in-flight requests", headers={"Retry-After": "1"})
                inflight += 1
                try:
                    async with asyncio.timeout(deadline):
                        return await fn(request)
                except TimeoutError as exc:
                    raise HTTPException(504, "Request deadline exceeded") from exc
                finally:
                    inflight -= 1
            return wrapped
        return decorate

    @app.get(ENDPOINTS["live"])
    async def live():
        return {"status": "alive"}

    @app.get(ENDPOINTS["ready"])
    async def ready(request: Request):
        # 真实 prefill 验证抓取能力与张量形状，而不只是端口存活。
        authenticate(request)
        nonlocal inflight
        if inflight >= settings.max_inflight:
            raise HTTPException(429, "Detector busy")
        inflight += 1
        try:
            async with asyncio.timeout(settings.timeout):
                ready_messages = (READY_PROMPT_BLOCKS
                                  if profile.position_mode == "blocks" else
                                  [{"role": "user", "content": READY_PROMPT}])
                _, _, profile_used = await pipeline.capture(ready_messages)
            return {"status": "ready",
                    "model": MODEL_LABEL,
                    "probe": state["probe"].spec.probe_id,
                    "checkpoint_sha256": state["probe"].sha256,
                    "capture_layers": list(state["probe"].capture_layers),
                    "composer_profile": profile_used}
        except TimeoutError as exc:
            raise HTTPException(504, "Readiness deadline exceeded") from exc
        finally:
            inflight -= 1

    @app.get(ENDPOINTS["info"])
    async def info(request: Request):
        authenticate(request)
        active = state["probe"]
        spec = active.spec
        # 调用方（CoolWatch）不配置模型名，靠这里的自述区分"引擎是谁"。
        model = getattr(settings, "detector_model", "") or MODEL_LABEL
        risks = {getattr(settings, "primary_risk", "ipi"): active, **(state.get("probes") or {})}
        return {
            "model": model,
            "detector_model": model,
            "risks": [{"risk": risk,
                       "probe_id": probe.spec.probe_id,
                       "checkpoint_sha256": probe.sha256,
                       "threshold": probe.threshold,
                       "threshold_unit": "sigmoid",
                       "block_indices": list(probe.spec.block_indices),
                       "capture_layers": list(probe.capture_layers)} for risk, probe in risks.items()],
            "enabled_risks": [risk for risk in KNOWN_RISKS if risk in risks],
            "disabled_risks": [risk for risk in KNOWN_RISKS if risk not in risks],
            "detector": spec.probe_id,
            "probe_schema": spec.schema,
            "probe_architecture": spec.probe_architecture,
            "checkpoint_sha256": active.sha256,
            "threshold": active.threshold,
            "threshold_unit": "sigmoid",
            "layer_semantics": spec.layer_semantics,
            "block_indices": list(spec.block_indices),
            "capture_layers": list(active.capture_layers),
            "selected_positions": list(active.selected_positions),
            "feature_normalization": spec.feature_normalization,
            "flatten_order": list(spec.flatten_order),
            "input_dim": active.input_dim,
            "feature_name": spec.feature_name,
            "labeling_protocol": spec.labeling_protocol,
            "composer": profile.describe(),
            "max_input_tokens": settings.max_tokens,
        }

    @app.post(ENDPOINTS["detect"], response_model=DetectionResponse, response_model_exclude_none=True)
    async def detect(request: Request):
        """检测注入风险。默认接收批量，也兼容单条。

        批量：``{"messages": [[...], [...]]}`` —— messages 为二维列表
        单条：``{"messages": [...]}`` —— 一维列表，返回结果与旧版完全一致

        批量里每条独立评分、独立报错：一条失败不会拖垮整批，
        失败项以 ``error`` 字段返回，绝不伪装成 benign。
        """
        authenticate(request)
        nonlocal inflight
        if inflight >= settings.max_inflight:
            raise HTTPException(429, "Too many in-flight requests", headers={"Retry-After": "1"})
        inflight += 1
        start = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        try:
            async with asyncio.timeout(settings.timeout):
                body = bytearray()
                async for chunk in request.stream():
                    if len(body) + len(chunk) > settings.max_body_bytes:
                        raise HTTPException(413, "Request exceeds body size limit")
                    body.extend(chunk)
                try:
                    payload = json.loads(body)
                    requests, tools, batch = split_detect_payload(payload)
                except (ValueError, UnicodeError, ValidationError) as exc:
                    raise HTTPException(422, "Invalid detect request: %s" % exc) from exc
                if len(requests) > getattr(settings, "max_batch_items", 128):
                    raise HTTPException(413, "Too many items in detect batch")
                parsed, slots, errors, normalizations = [], [], {}, {}
                for slot, item in enumerate(requests):
                    try:
                        normalized, changes = normalize_payload(
                            item, extra_fields=("composer_profile", "enabled_risks", "explain"))
                        parsed.append(DetectionRequest.model_validate(normalized))
                        slots.append(slot)
                        normalizations[slot] = {"applied": bool(changes), "changes": changes}
                    except (ValueError, ValidationError) as exc:
                        if not batch:
                            raise HTTPException(422, "Invalid detect request: %s" % exc) from exc
                        errors[slot] = {"error": "InvalidRequest", "detail": str(exc)[:200], "status_code": 422}
                selected = request_profile(requests[0].get("composer_profile")) if parsed else None
                token = pipeline.use_profile(selected) if selected is not None else None
                # 请求级风险子集：调用方可只取自己关心的风险，服务可多加载。
                set_bank(select_risks(state, requests[0].get("enabled_risks") if parsed else None))
                try:
                    started = time.perf_counter()
                    if batch:
                        scored = await pipeline.detect_batch(parsed)
                        bank_scores = next(({candidate.spec.probe_id: (row.get("scores") or {}).get(
                            candidate.spec.probe_id, (row["logit"], row["probability"]))
                            for candidate in bank_view().values()} for row in scored
                            if "error" not in row and "scores" in row), None)
                        tokens = next((row["input_tokens"] for row in scored
                                       if "error" not in row), 0)
                    else:
                        row, bank_scores, _described = await pipeline.collect(
                            parsed[0].messages, parsed[0].tools,
                            explain=bool(parsed[0].explain))
                        scored = [row]
                        tokens = row["input_tokens"]
                    results = [errors.get(slot) for slot in range(len(requests))]
                    for slot, row in zip(slots, scored):
                        results[slot] = {**row, "normalization": normalizations[slot]}
                    described = (selected or pipeline.profile).describe()
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                    common = {
                        "risk": RISK_TYPE,
                        "model": MODEL_LABEL,
                        "composer": described,
                        "composer_profile": described["profile_id"],
                        "checkpoint_sha256": state["probe"].sha256,
                        "request_id": request_id,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                    if (state.get("probes") or requests[0].get("enabled_risks")) and bank_scores is not None:
                        # Probe Bank 调用方按 mapping 解包 latency_ms，另给 total。
                        common = {**common, "latency_ms": {"total": elapsed_ms},
                                  **bank_payload(bank_scores, tokens, bank_view())}
                    items = [{**row, **common} for row in results]
                finally:
                    if token is not None:
                        pipeline.reset_profile(token)
                    set_bank(None)
                if batch:
                    return {**common, "schema": "inscope.detect_batch.v1",
                            "count": len(items), "results": items}
                return items[0]
        except TimeoutError as exc:
            raise HTTPException(504, "Detection deadline exceeded") from exc
        finally:
            inflight -= 1

    @app.post(ENDPOINTS["audit"])
    @bounded(settings.audit_timeout)
    async def audit(request: Request):
        """沿执行轨迹逐检查点评分，回答"最早在哪一步检出"。

        与 detect 共用编排和探针评分；不构造反事实，只回放各时刻可见的上下文。
        评分失败会记入 failed_at 并标 incomplete，不会被当成阴性。
        """
        authenticate(request)
        start = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > settings.max_body_bytes:
                raise HTTPException(413, "Request exceeds body size limit")
            body.extend(chunk)
        try:
            payload = json.loads(body)
            normalized, changes = normalize_payload(
                payload, extra_fields=("checkpoint_roles", "start_index", "composer_profile"))
            parsed = AuditRequest.model_validate(normalized)
        except (ValueError, UnicodeError, ValidationError) as exc:
            raise HTTPException(422, "Invalid audit request: %s" % exc) from exc

        selected = request_profile(parsed.composer_profile)
        token = pipeline.use_profile(selected) if selected is not None else None
        try:
            return await _audit(parsed, changes, request_id, start, selected)
        finally:
            if token is not None:
                pipeline.reset_profile(token)

    async def _audit(parsed, changes, request_id, start, selected):
        async def score_at(window):
            score, tokens, _described = await pipeline.capture(
                window["messages"], window["tools"])
            return {**score, "input_tokens": tokens}

        try:
            async with asyncio.timeout(settings.audit_timeout):
                result = await audit_trajectory(
                    parsed.messages, score_at, tools=parsed.tools,
                    roles=tuple(parsed.checkpoint_roles),
                    start_index=parsed.start_index,
                    case_id=request_id)
        except TimeoutError as exc:
            raise HTTPException(
                504, "Audit exceeded %ss deadline" % settings.audit_timeout) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {**result,
                "risk": RISK_TYPE,
                "model": MODEL_LABEL,
                "composer": (selected or pipeline.profile).describe(),
                "composer_profile": (selected or pipeline.profile).profile_id,
                "normalization": {"applied": bool(changes), "changes": changes},
                "checkpoint_sha256": state["probe"].sha256,
                "request_id": request_id,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    @app.post(ENDPOINTS["analyze"])
    @bounded(settings.analyze_timeout)
    async def analyze(request: Request):
        """定位一条 observation 内影响探针评分的关键区间。

        与 detect 共享编排和探针评分；特征请求使用批量后端。
        与其他评分入口共享请求限额，读取请求体也计入分析 deadline。
        """
        authenticate(request)
        start = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > settings.max_body_bytes:
                raise HTTPException(413, "Request exceeds body size limit")
            body.extend(chunk)
        try:
            payload = json.loads(body)
            normalized, changes = normalize_payload(
                payload,
                extra_fields=("observation_index", "granularity", "mode",
                              "focus", "group_size", "max_coalitions", "composer_profile"))
            parsed = AnalyzeRequest.model_validate(normalized)
        except (ValueError, UnicodeError, ValidationError) as exc:
            raise HTTPException(422, "Invalid analyze request: %s" % exc) from exc

        score_batch = make_batched_scorer(pipeline.score_prompts, max_batch=settings.analyze_batch)
        selected = request_profile(parsed.composer_profile)
        token = pipeline.use_profile(selected) if selected is not None else None
        try:
            async with asyncio.timeout(settings.analyze_timeout):
                result = await localize_observation(
                    {"messages": parsed.messages, "tools": parsed.tools},
                    observation_index=parsed.observation_index,
                    score_batch=score_batch,
                    case_id=request_id,
                    granularity=parsed.granularity,
                    mode=parsed.mode,
                    focus=parsed.focus,
                    group_size=parsed.group_size,
                    max_coalitions=parsed.max_coalitions)
        except TimeoutError as exc:
            raise HTTPException(
                504, "Analysis exceeded %ss deadline" % settings.analyze_timeout) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            if token is not None:
                pipeline.reset_profile(token)
        return {**result,
                "risk": RISK_TYPE,
                "model": MODEL_LABEL,
                "composer": (selected or pipeline.profile).describe(),
                "composer_profile": (selected or pipeline.profile).profile_id,
                "normalization": {"applied": bool(changes), "changes": changes},
                "checkpoint_sha256": state["probe"].sha256,
                "request_id": request_id,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    # 保留流式读取和大小限制，同时由同一契约生成 OpenAPI 请求说明。
    from fastapi.openapi.utils import get_openapi
    from pydantic.json_schema import models_json_schema

    def openapi():
        if app.openapi_schema is None:
            document = get_openapi(title=app.title, version=app.version, routes=app.routes)
            models = [DetectionRequest, DetectionBatchRequest, AuditRequest, AnalyzeRequest]
            _refs, definitions = models_json_schema(
                [(model, "validation") for model in models],
                ref_template="#/components/schemas/{model}")
            document.setdefault("components", {}).setdefault("schemas", {}).update(definitions.get("$defs", {}))
            for endpoint, names in (("detect", ["DetectionRequest", "DetectionBatchRequest"]),
                                    ("audit", ["AuditRequest"]), ("analyze", ["AnalyzeRequest"])):
                refs = [{"$ref": "#/components/schemas/" + name} for name in names]
                schema = {"anyOf": refs} if len(refs) > 1 else refs[0]
                document["paths"][ENDPOINTS[endpoint]]["post"]["requestBody"] = {
                    "required": True, "content": {"application/json": {"schema": schema}}}
            app.openapi_schema = document
        return app.openapi_schema

    app.openapi = openapi
    return app
