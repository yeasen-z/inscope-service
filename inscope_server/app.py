import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from inscope_contract import DetectionRequest, ENDPOINTS, MODEL_LABEL, RISK_TYPE

from .backend import BackendError, InScopeBackend
from .config import Settings, validate_model
from .context_profile import bind_profile, compose, get_profile
from .normalization import normalize_payload
from .probe import load_probe

log = logging.getLogger(__name__)

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

    async def capture(messages, tools=None):
        active = state["probe"]
        try:
            # Context Composer 按 profile 产出唯一可用的 input_ids；
            # 后端只允许原样转发，不得重新渲染或重新 tokenize。
            ctx = await asyncio.to_thread(compose, state["tokenizer"], messages, tools, profile)
        except NotImplementedError as exc:
            raise HTTPException(501, "Composer profile is not implemented") from exc
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise HTTPException(422, "Invalid conversation or tool template input") from exc
        ids = list(ctx.input_ids)
        if len(ids) > settings.max_tokens:
            raise HTTPException(413, "Input exceeds %d tokens; no truncation applied"
                                % settings.max_tokens)
        # 探针声明的负向位置要求输入至少有这么长；提前给出明确原因。
        need = max((-p for p in active.selected_positions if p < 0), default=0)
        if len(ids) < need:
            raise HTTPException(422, "Input has %d tokens but the probe needs at least %d"
                                % (len(ids), need))
        try:
            hidden = await state["backend"].capture(
                ids, active.capture_layers, active.selected_positions)
        except TimeoutError as exc:
            raise HTTPException(504, "SGLang request timed out") from exc
        except BackendError as exc:
            log.warning("Shadow backend failed: %s", type(exc).__name__)
            raise HTTPException(503, "Shadow feature backend unavailable") from exc
        try:
            logit, prob = active.score(active.flatten(hidden))
        except ValueError as exc:
            raise HTTPException(503, "Invalid InScope features or score") from exc
        score = {"logit": logit, "probability": prob,
                 "threshold": active.threshold,
                 "label": "attack" if prob >= active.threshold else "benign"}
        return score, len(ids), ctx.profile

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
                _, _, profile_used = await capture(
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
                    normalized, changes = normalize_payload(payload)
                    parsed = DetectionRequest.model_validate(normalized)
                except (ValueError, UnicodeError, ValidationError) as exc:
                    raise HTTPException(
                        422, "Expected messages list or {messages: [...], tools?: [...]}") from exc
                result, tokens, profile_used = await capture(parsed.messages, parsed.tools)
                return {**result,
                        "risk": RISK_TYPE,
                        "model": MODEL_LABEL,
                        "composer": profile_used,
                        "composer_profile": profile_used["profile_id"],
                        "normalization": {"applied": bool(changes), "changes": changes},
                        "checkpoint_sha256": state["probe"].sha256,
                        "input_tokens": tokens,
                        "request_id": request_id,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2)}
        except TimeoutError as exc:
            raise HTTPException(504, "Detection deadline exceeded") from exc
        finally:
            inflight -= 1

    return app
