"""CPU-only public wire/SDK regressions; no network or credentials."""
import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from inscope import AsyncClient, Client, DetectResult, InScopeError
from inscope_contract import DETECTION_RESPONSE_FIELDS
from inscope_contract.schemas import (
    AuditRequest, DetectionBatchRequest, DetectionBatchResult, DetectionItemError, DetectionResult,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def payload(**updates):
    return {**dict(probability=0.8, logit=1.4, threshold=0.9, label="benign",
                  risk="indirect_prompt_injection", model="test", input_tokens=7,
                  request_id="id", latency_ms=1.0, composer_profile="profile",
                  checkpoint_sha256="abc"), **updates}


def batch_payload():
    common = {key: value for key, value in payload().items()
              if key not in {"probability", "logit", "threshold", "label", "input_tokens"}}
    return {**common, "schema": "inscope.detect_batch.v1", "count": 2,
            "results": [payload(), {"error": "InvalidRequest", "detail": "bad item", "status_code": 422}]}


def install(client, handler):
    transport = httpx.MockTransport(handler)
    if isinstance(client, AsyncClient):
        client._client = httpx.AsyncClient(base_url="http://test", transport=transport)
    else:
        client._session = httpx.Client(base_url="http://test", transport=transport)


def test_detection_fields_are_actual_required_contract():
    required = {name for name, field in DetectionResult.model_fields.items() if field.is_required()}
    assert set(DETECTION_RESPONSE_FIELDS) == required
    result = DetectionResult.model_validate(payload())
    assert result.flagged is False
    assert "flagged" not in result.model_dump()
    assert "score" not in result.model_dump()
    assert result.model == "test"
    assert result.checkpoint_sha256 == "abc"
    with pytest.raises(ValidationError):
        DetectionResult.model_validate({k: v for k, v in payload().items() if k != "model"})


@pytest.mark.parametrize("updates", [
    {"label": "unknown"}, {"label": "attack"}, {"probability": -0.1},
    {"probability": 1.1}, {"probability": float("nan")}, {"logit": float("inf")},
    {"threshold": -1.0}, {"latency_ms": -1.0}, {"input_tokens": True},
    {"input_tokens": -1}, {"probability": "0.8"}, {"label": None},
    {"error": "oops"}, {"flagged": False},
])
def test_sdk_illegal_success_payload_fails_closed(updates):
    with pytest.raises(InScopeError):
        DetectResult.from_payload(payload(**updates))


def test_batch_error_cannot_masquerade_as_success():
    result = DetectionBatchResult.model_validate(batch_payload())
    assert isinstance(result.results[0], DetectionResult)
    assert isinstance(result.results[1], DetectionItemError)
    assert not hasattr(result.results[1], "flagged")
    assert result.model_dump(by_alias=True)["schema"] == "inscope.detect_batch.v1"
    bad = batch_payload()
    bad["results"][1]["label"] = "benign"
    with pytest.raises(ValidationError):
        DetectionBatchResult.model_validate(bad)
    with pytest.raises(ValidationError):
        DetectionBatchResult.model_validate({**batch_payload(), "count": 3})


def test_request_contracts_reject_unimplemented_audit_thresholds():
    with pytest.raises(ValidationError):
        AuditRequest.model_validate({"messages": MESSAGES, "thresholds": [0.5]})
    with pytest.raises(ValidationError):
        DetectionBatchRequest.model_validate({"messages": [MESSAGES] * 129})


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sync_async_batch_and_analysis_parity(asynchronous):
    seen = []
    client = (AsyncClient if asynchronous else Client)(analyze_timeout=1234.0, audit_timeout=2345.0)

    def handler(request):
        seen.append((request.url.path, json.loads(request.content), request.extensions["timeout"], request.headers))
        value = batch_payload() if request.url.path == "/detect" else {"schema": "analysis", "status": "complete"}
        return httpx.Response(200, json=value)

    install(client, handler)

    async def run_async():
        async with client:
            result = await client.detect_batch([MESSAGES, MESSAGES], tools=[], request_id="rid")
            analysis = await client.analyze(MESSAGES, 0, mode="exact", request_id="arid")
            await client.analyze(MESSAGES, 0, timeout=3456.0)
            await client.audit(MESSAGES)
        return result, analysis

    if asynchronous:
        result, analysis = asyncio.run(run_async())
    else:
        with client:
            result = client.detect_batch([MESSAGES, MESSAGES], tools=[], request_id="rid")
            analysis = client.analyze(MESSAGES, 0, mode="exact", request_id="arid")
            client.analyze(MESSAGES, 0, timeout=3456.0)
            client.audit(MESSAGES)
    assert type(result) is DetectionBatchResult
    assert isinstance(result.results[1], DetectionItemError)
    assert analysis["status"] == "complete"
    assert seen[0][1] == {"messages": [MESSAGES, MESSAGES], "tools": []}
    assert seen[0][3]["X-Request-ID"] == "rid"
    assert seen[1][1]["mode"] == "exact"
    assert seen[1][1]["observation_index"] == 0
    assert seen[1][3]["X-Request-ID"] == "arid"
    assert [row[2]["read"] for row in seen[1:]] == [1234.0, 3456.0, 2345.0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", ["detect", "detect_batch", "analyze", "audit", "info", "ready"])
def test_all_transport_errors_become_inscope_error(asynchronous, method):
    client = (AsyncClient if asynchronous else Client)()

    def fail(request):
        raise httpx.ConnectError("unreachable", request=request)

    install(client, fail)
    args = {"detect": [MESSAGES], "detect_batch": [[MESSAGES]], "analyze": [MESSAGES, 0],
            "audit": [MESSAGES], "info": [], "ready": []}[method]
    if asynchronous:
        async def run():
            async with client:
                with pytest.raises(InScopeError, match="transport"):
                    await getattr(client, method)(*args)
        asyncio.run(run())
    else:
        with client, pytest.raises(InScopeError, match="transport"):
            getattr(client, method)(*args)


@pytest.mark.parametrize("value", [[], None, "benign", {"error": "failure"}, {"score": 0}])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_detect_malformed_json_shape_fails_closed(value, asynchronous):
    client = (AsyncClient if asynchronous else Client)()
    install(client, lambda request: httpx.Response(200, json=value))
    if asynchronous:
        async def run():
            async with client:
                with pytest.raises(InScopeError):
                    await client.detect(MESSAGES)
        asyncio.run(run())
    else:
        with client, pytest.raises(InScopeError):
            client.detect(MESSAGES)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_batch_response_cardinality_must_match_request(asynchronous):
    client = (AsyncClient if asynchronous else Client)()
    install(client, lambda request: httpx.Response(200, json=batch_payload()))
    if asynchronous:
        async def run():
            async with client:
                with pytest.raises(InScopeError, match="request length"):
                    await client.detect_batch([MESSAGES])
        asyncio.run(run())
    else:
        with client, pytest.raises(InScopeError, match="request length"):
            client.detect_batch([MESSAGES])


def test_async_client_rejects_synchronous_with_without_opening_session():
    client = AsyncClient()
    with pytest.raises(TypeError, match="async with"):
        with client:
            pass
    assert client._client is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"not-json"),
    httpx.Response(200, json=[]),
    httpx.Response(200, json={"error": "failed"}),
    httpx.Response(503, json=["unavailable"]),
])
def test_analysis_http_and_invalid_response_fail_closed(asynchronous, response):
    client = (AsyncClient if asynchronous else Client)()
    install(client, lambda request: httpx.Response(response.status_code, content=response.content))
    if asynchronous:
        async def run():
            async with client:
                with pytest.raises(InScopeError):
                    await client.analyze(MESSAGES, 0)
        asyncio.run(run())
    else:
        with client, pytest.raises(InScopeError):
            client.analyze(MESSAGES, 0)


@pytest.mark.parametrize("messages", [[], MESSAGES, "not a batch", [MESSAGES, {}]])
def test_batch_method_requires_explicit_batch_shape(messages):
    with Client() as client, pytest.raises(InScopeError, match="conversations"):
        client.detect_batch(messages)
