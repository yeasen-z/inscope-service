import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from inscope_contract import DETECTOR_NAME, DetectionRequest, ENDPOINTS, MODEL_LABEL

from .backend import BackendError, InScopeBackend
from .config import INSCOPE_COMMIT, Settings, validate_model
from .context_profile import compose, get_profile
from .probe import Probe, load_probe
from .normalization import normalize_payload

log = logging.getLogger(__name__)


def create_app(settings=None, *, tokenizer=None, probe=None, backend=None):
    """Dependencies can be injected for CPU-only protocol tests."""
    settings = settings or Settings.from_env()
    # profile 必须在启动期解析：未知 id 或不匹配的探针直接失败，而不是首个请求才暴露。
    profile = get_profile(settings.composer_profile) if hasattr(settings, "composer_profile") \
        else get_profile("a5_simplified_v1", probe_prefix=settings.composer.probe_prefix)
    composer = profile.composer

    @asynccontextmanager
    async def lifespan(app):
        nonlocal tokenizer, probe, backend
        if tokenizer is None:
            validate_model(settings.model_path)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(settings.model_path, local_files_only=True,
                                                     trust_remote_code=False)
        # 探针结构与层号完全由 checkpoint 自描述决定，不在此处写死任何一套。
        if probe is None:
            if not getattr(settings, "probe_checkpoint", ""):
                raise ValueError("INSCOPE_PROBE_CHECKPOINT is required")
            probe = load_probe(settings.probe_checkpoint)
        backend = backend or InScopeBackend(settings)
        try:
            yield
        finally:
            await backend.close()

    app = FastAPI(title="CoolWatch-server Qwen3-8B-FP8 detector", version="0.3.0", lifespan=lifespan)
    inflight = 0

    def authenticate(request):
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth.encode(), ("Bearer " + settings.api_key).encode()):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    async def capture(messages, tools=None):
        try:
            # Context Composer 按 profile 产出唯一可用的 input_ids；
            # 后端只允许原样转发，不得重新渲染或重新 tokenize。
            ctx = await asyncio.to_thread(compose, tokenizer, messages, tools, profile)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise HTTPException(422, "Invalid conversation or tool template input") from exc
        except NotImplementedError as exc:
            raise HTTPException(501, "Composer profile is not implemented") from exc
        ids = list(ctx.input_ids)
        if len(ids) > settings.max_tokens:
            raise HTTPException(413, f"Input exceeds {settings.max_tokens} tokens; no truncation applied")
        # 探针声明的负向位置要求输入至少有这么长；提前给出明确原因。
        need = max((-p for p in probe.selected_positions if p < 0), default=0)
        if len(ids) < need:
            raise HTTPException(422, "Input has %d tokens but the probe needs at least %d"
                                % (len(ids), need))
        try:
            hidden = await backend.capture(ids, probe.capture_layers, probe.selected_positions)
        except TimeoutError as exc:
            raise HTTPException(504, "SGLang request timed out") from exc
        except BackendError as exc:
            log.warning("Shadow backend failed: %s", type(exc).__name__)
            raise HTTPException(503, "Shadow feature backend unavailable") from exc
        try:
            logit, prob = probe.score(probe.flatten(hidden))
        except ValueError as exc:
            raise HTTPException(503, "Invalid InScope features or score") from exc
        score = {"logit": logit, "probability": prob,
                 "threshold": probe.threshold,
                 "label": "attack" if prob >= probe.threshold else "benign"}
        return score, len(ids), ctx.profile

    @app.get(ENDPOINTS["live"])
    async def live():
        return {"status": "alive"}

    @app.get(ENDPOINTS["ready"])
    async def ready(request: Request):
        # A real prefill checks capture support and tensor shape, not just a live socket.
        authenticate(request)
        nonlocal inflight
        if inflight >= settings.max_inflight:
            raise HTTPException(429, "Detector busy")
        inflight += 1
        try:
            async with asyncio.timeout(settings.timeout):
                _, _, profile_used = await capture(
                    [{"role": "user", "content": "Hello, please confirm the detection pipeline is operational."}])
            return {"status": "ready", "model": "Qwen3-8B-FP8",
                    "capture_layers": list(probe.capture_layers),
                    "composer_profile": profile_used}
        except TimeoutError as exc:
            raise HTTPException(504, "Readiness deadline exceeded") from exc
        finally:
            inflight -= 1

    @app.get(ENDPOINTS["info"])
    async def info(request: Request):
        authenticate(request)
        spec = probe.spec
        return {"model": "Qwen3-8B-FP8", "quantization": "fp8", "risk": "indirect_prompt_injection",
                "detector": DETECTOR_NAME,
                "probe_schema": spec.schema,
                "probe_architecture": spec.probe_architecture,
                "checkpoint_sha256": probe.sha256,
                "threshold": probe.threshold, "threshold_unit": "sigmoid",
                "setting": "simplified", "composer": profile.describe(),
                "layer_semantics": spec.layer_semantics,
                "block_indices": list(spec.block_indices),
                "capture_layers": list(probe.capture_layers),
                "selected_positions": list(probe.selected_positions),
                "feature_normalization": spec.feature_normalization,
                "flatten_order": list(spec.flatten_order),
                "input_dim": probe.input_dim,
                "feature_name": spec.feature_name,
                "labeling_protocol": spec.labeling_protocol,
                "enable_thinking": composer.enable_thinking if composer else None,
                "max_input_tokens": settings.max_tokens,
                "inscope_commit": INSCOPE_COMMIT}

    @app.post(ENDPOINTS["detect"])
    async def detect(request: Request):
        authenticate(request)
        nonlocal inflight
        if inflight >= settings.max_inflight:
            raise HTTPException(429, "Too many in-flight requests", headers={"Retry-After": "1"})
        inflight += 1
        start = time.perf_counter()
        request_id = str(uuid.uuid4())
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
                    raise HTTPException(422, "Expected messages list or {messages: [...], tools?: [...]}") from exc
                result, tokens, profile_used = await capture(parsed.messages, parsed.tools)
                return {**result, "risk": "indirect_prompt_injection", "model": MODEL_LABEL,
                        "setting": "simplified",
                        "composer": profile_used,
                        "composer_profile": profile_used["profile_id"],
                        "normalization": {"applied": bool(changes), "changes": changes},
                        "checkpoint_sha256": probe.sha256, "input_tokens": tokens,
                        "request_id": request_id, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}
        except TimeoutError as exc:
            raise HTTPException(504, "Detection deadline exceeded") from exc
        finally:
            inflight -= 1
    return app
