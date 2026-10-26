"""Synchronous and asynchronous HTTP clients sharing the public wire contract."""
import httpx
from pydantic import Field, ValidationError

from inscope_contract import DEFAULT_ANALYZE_TIMEOUT, DEFAULT_API_PORT, DEFAULT_AUDIT_TIMEOUT, ENDPOINTS
from inscope_contract.schemas import DetectionBatchResult, DetectionItemError, DetectionResult


class InScopeError(Exception):
    """Transport, HTTP or malformed-response failure (never a benign result)."""


class DetectResult(DetectionResult):
    """SDK detection result, including the historical ``flagged`` convenience.

    Older services omitted metadata; defaults preserve those successful callers.
    Score, label and token count are always validated by the shared contract.
    """

    risk: str = ""
    model: str = ""
    request_id: str = ""
    latency_ms: float = Field(default=0.0, ge=0)
    composer_profile: str = ""
    checkpoint_sha256: str = ""

    @classmethod
    def from_payload(cls, payload):
        try:
            return cls.model_validate(payload)
        except (ValidationError, TypeError, ValueError) as exc:
            raise InScopeError("Malformed detect response: %s" % exc) from exc


# Batch methods intentionally return exactly the same shared type in both clients.
DetectBatchResult = DetectionBatchResult
DetectItemError = DetectionItemError


def _json_object(response):
    try:
        value = response.json()
    except (ValueError, UnicodeError) as exc:
        raise InScopeError("Malformed response: invalid JSON") from exc
    if not isinstance(value, dict) or "error" in value:
        raise InScopeError("Malformed response: expected a successful JSON object")
    return value


def _raise(response):
    if response.status_code == 200:
        return
    try:
        payload = response.json()
        detail = payload.get("detail", response.text) if isinstance(payload, dict) else payload
    except (ValueError, UnicodeError):
        detail = response.text
    raise InScopeError("%d: %s" % (response.status_code, str(detail)[:400]))


def _body(messages, tools, composer_profile=None):
    body = {"messages": messages}
    if tools is not None:
        body["tools"] = tools
    if composer_profile is not None:
        body["composer_profile"] = composer_profile
    return body


def _headers(request_id):
    return {"X-Request-ID": request_id} if request_id else {}


def _batch_result(response, expected_count):
    try:
        result = DetectionBatchResult.model_validate(_json_object(response))
        if result.count != expected_count:
            raise ValueError("batch result does not match request length")
        return result
    except (ValidationError, TypeError, ValueError) as exc:
        raise InScopeError("Malformed detect batch response: %s" % exc) from exc


def _batch_body(messages, tools, composer_profile=None):
    # Explicit method: do not silently change a one-dimensional detect call to batch.
    if not isinstance(messages, list) or not messages or any(not isinstance(row, list) for row in messages):
        raise InScopeError("detect_batch requires a nonempty list of conversations")
    return _body(messages, tools, composer_profile)


def _analysis_body(messages, tools, observation_index, granularity, mode, focus, group_size,
                   max_coalitions, composer_profile=None):
    return {**_body(messages, tools, composer_profile), "observation_index": observation_index,
            "granularity": granularity, "mode": mode, "focus": focus,
            "group_size": group_size, "max_coalitions": max_coalitions}


def _audit_body(messages, tools, checkpoint_roles, start_index, composer_profile=None):
    body = {**_body(messages, tools, composer_profile), "start_index": start_index}
    if checkpoint_roles is not None:
        body["checkpoint_roles"] = list(checkpoint_roles)
    return body


class Client:
    """Reusable synchronous client. Analysis/audit have independent long deadlines.

    ``analyze_timeout`` and ``audit_timeout`` accept seconds or httpx.Timeout;
    their respective methods also accept a per-call ``timeout`` override.
    """

    def __init__(self, base_url=None, api_key="", timeout=120.0, *,
                 analyze_timeout=DEFAULT_ANALYZE_TIMEOUT, audit_timeout=DEFAULT_AUDIT_TIMEOUT):
        self.base_url = (base_url or "http://127.0.0.1:%d" % DEFAULT_API_PORT).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.analyze_timeout = analyze_timeout
        self.audit_timeout = audit_timeout
        self._session = None

    def _client(self):
        if self._session is None:
            headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
            self._session = httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)
        return self._session

    def _request(self, method, endpoint, **kwargs):
        try:
            return getattr(self._client(), method)(ENDPOINTS[endpoint], **kwargs)
        except httpx.RequestError as exc:
            raise InScopeError("InScope transport failure: %s" % type(exc).__name__) from exc

    def _post(self, endpoint, body, request_id, **kwargs):
        response = self._request("post", endpoint, json=body, headers=_headers(request_id), **kwargs)
        _raise(response)
        return response

    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def info(self):
        response = self._request("get", "info")
        _raise(response)
        return _json_object(response)

    def detect(self, messages, tools=None, request_id=None, *, composer_profile=None):
        body = _body(messages, tools, composer_profile)
        return DetectResult.from_payload(_json_object(self._post("detect", body, request_id)))

    def detect_batch(self, messages, tools=None, request_id=None, *, composer_profile=None):
        """Return ordered DetectionResult/DetectionItemError rows, never drop failures."""
        body = _batch_body(messages, tools, composer_profile)
        return _batch_result(self._post("detect", body, request_id), len(messages))

    def analyze(self, messages, observation_index, tools=None, *, granularity="paragraph",
                mode="auto", focus="high", group_size=6, max_coalitions=256,
                request_id=None, timeout=None, composer_profile=None):
        """Locate influential observation spans; result is the service JSON object."""
        body = _analysis_body(messages, tools, observation_index, granularity, mode, focus,
                              group_size, max_coalitions, composer_profile)
        return _json_object(self._post("analyze", body, request_id,
                            timeout=self.analyze_timeout if timeout is None else timeout))

    def audit(self, messages, tools=None, *, checkpoint_roles=None, start_index=0,
              request_id=None, timeout=None, composer_profile=None):
        body = _audit_body(messages, tools, checkpoint_roles, start_index, composer_profile)
        return _json_object(self._post("audit", body, request_id,
                            timeout=self.audit_timeout if timeout is None else timeout))

    def ready(self):
        return self._request("get", "ready").status_code == 200

    _raise = staticmethod(_raise)


class AsyncClient:
    """Async equivalent of Client; use ``async with`` or explicit ``aclose()``."""

    def __init__(self, base_url=None, api_key="", timeout=120.0, *,
                 analyze_timeout=DEFAULT_ANALYZE_TIMEOUT, audit_timeout=DEFAULT_AUDIT_TIMEOUT):
        self.base_url = (base_url or "http://127.0.0.1:%d" % DEFAULT_API_PORT).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.analyze_timeout = analyze_timeout
        self.audit_timeout = audit_timeout
        self._client = None

    def _ensure(self):
        if self._client is None:
            headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout)
        return self._client

    async def _request(self, method, endpoint, **kwargs):
        try:
            return await getattr(self._ensure(), method)(ENDPOINTS[endpoint], **kwargs)
        except httpx.RequestError as exc:
            raise InScopeError("InScope transport failure: %s" % type(exc).__name__) from exc

    async def _post(self, endpoint, body, request_id, **kwargs):
        response = await self._request("post", endpoint, json=body, headers=_headers(request_id), **kwargs)
        _raise(response)
        return response

    async def info(self):
        response = await self._request("get", "info")
        _raise(response)
        return _json_object(response)

    async def detect(self, messages, tools=None, request_id=None, *, composer_profile=None):
        body = _body(messages, tools, composer_profile)
        return DetectResult.from_payload(_json_object(await self._post("detect", body, request_id)))

    async def detect_batch(self, messages, tools=None, request_id=None, *, composer_profile=None):
        body = _batch_body(messages, tools, composer_profile)
        return _batch_result(await self._post("detect", body, request_id), len(messages))

    async def analyze(self, messages, observation_index, tools=None, *, granularity="paragraph",
                      mode="auto", focus="high", group_size=6, max_coalitions=256,
                      request_id=None, timeout=None, composer_profile=None):
        body = _analysis_body(messages, tools, observation_index, granularity, mode, focus,
                              group_size, max_coalitions, composer_profile)
        return _json_object(await self._post("analyze", body, request_id,
                            timeout=self.analyze_timeout if timeout is None else timeout))

    async def audit(self, messages, tools=None, *, checkpoint_roles=None, start_index=0,
                    request_id=None, timeout=None, composer_profile=None):
        body = _audit_body(messages, tools, checkpoint_roles, start_index, composer_profile)
        return _json_object(await self._post("audit", body, request_id,
                            timeout=self.audit_timeout if timeout is None else timeout))

    async def ready(self):
        return (await self._request("get", "ready")).status_code == 200

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    def __enter__(self):
        raise TypeError("AsyncClient requires 'async with', not synchronous 'with'")

    def __exit__(self, *exc):
        raise TypeError("AsyncClient requires 'async with'")
