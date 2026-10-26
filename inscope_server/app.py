"""HTTP 入口：依赖生命周期、鉴权、请求校验和结果封装。"""
import asyncio
from contextlib import asynccontextmanager
import hmac
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

from .analyze import analyze as localize_observation
from .audit import audit_trajectory
from .batching import make_batched_scorer
from .backend import InScopeBackend
from .config import Settings, validate_model
from .context_profile import bind_profile, get_profile
from .normalization import normalize_payload, split_detect_payload
from .probe import load_probe
from .scoring import ScoringPipeline

# 就绪探针必须长到能覆盖探针声明的负向位置；不足时由 capture 报 422。
READY_PROMPT = "Hello, please confirm the detection pipeline is operational."


def create_app(settings=None, *, tokenizer=None, probe=None, backend=None):
    """Dependencies can be injected for CPU-only protocol tests."""
    settings = settings or Settings.from_env()
    # profile 必须在启动期解析：未知 id 直接失败，而不是首个请求才暴露。
    profile = get_profile(settings.composer_profile)

    state = {"probe": probe, "tokenizer": tokenizer, "backend": backend}

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
                _, _, profile_used = await pipeline.capture(
                    [{"role": "user", "content": READY_PROMPT}])
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
        return {
            "model": MODEL_LABEL,
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

    @app.post(ENDPOINTS["detect"])
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
                    parsed = [DetectionRequest.model_validate(item) for item in requests]
                except (ValueError, UnicodeError, ValidationError) as exc:
                    raise HTTPException(
                        422, "Invalid detect request: %s" % exc) from exc
                results = await pipeline.detect_batch(parsed)
                described = profile.describe()
                common = {
                    "risk": RISK_TYPE,
                    "model": MODEL_LABEL,
                    "composer": described,
                    "composer_profile": described["profile_id"],
                    "checkpoint_sha256": state["probe"].sha256,
                    "request_id": request_id,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }
                items = [{**row, **common} for row in results]
                if batch:
                    return {**common, "schema": "inscope.detect_batch.v1",
                            "count": len(items), "results": items}
                return items[0]
        except TimeoutError as exc:
            raise HTTPException(504, "Detection deadline exceeded") from exc
        finally:
            inflight -= 1

    @app.post(ENDPOINTS["audit"])
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
                payload, extra_fields=("checkpoint_roles", "start_index"))
            parsed = AuditRequest.model_validate(normalized)
        except (ValueError, UnicodeError, ValidationError) as exc:
            raise HTTPException(422, "Invalid audit request: %s" % exc) from exc

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
                "composer": profile.describe(),
                "composer_profile": profile.profile_id,
                "normalization": {"applied": bool(changes), "changes": changes},
                "checkpoint_sha256": state["probe"].sha256,
                "request_id": request_id,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    @app.post(ENDPOINTS["analyze"])
    async def analyze(request: Request):
        """定位一条 observation 内影响探针评分的关键区间。

        与 detect 共享编排和探针评分；特征请求使用批量后端。
        当前仅限制每批规模和总超时，尚无 analyze 请求级并发隔离。
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
                              "focus", "group_size", "max_coalitions"))
            parsed = AnalyzeRequest.model_validate(normalized)
        except (ValueError, UnicodeError, ValidationError) as exc:
            raise HTTPException(422, "Invalid analyze request: %s" % exc) from exc

        score_batch = make_batched_scorer(pipeline.score_prompts, max_batch=settings.analyze_batch)
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
        return {**result,
                "risk": RISK_TYPE,
                "model": MODEL_LABEL,
                "composer": profile.describe(),
                "composer_profile": profile.profile_id,
                "normalization": {"applied": bool(changes), "changes": changes},
                "checkpoint_sha256": state["probe"].sha256,
                "request_id": request_id,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    return app
