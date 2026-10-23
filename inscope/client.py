"""同步/异步 HTTP 客户端。契约字段名来自 inscope_contract，不在这里重复。"""
from dataclasses import dataclass

import httpx

from inscope_contract import DEFAULT_API_PORT, ENDPOINTS


class InScopeError(Exception):
    """服务不可达、鉴权失败、或返回非法结果。"""


@dataclass(frozen=True)
class DetectResult:
    score: float
    logit: float
    threshold: float
    threshold_logit: float
    flagged: bool
    risk: str
    input_tokens: int
    request_id: str

    @classmethod
    def from_payload(cls, payload):
        return cls(
            score=payload["score"],
            logit=payload["logit"],
            threshold=payload["threshold"],
            threshold_logit=payload["threshold_logit"],
            flagged=payload["flagged"],
            risk=payload["risk"],
            input_tokens=payload["input_tokens"],
            request_id=payload["request_id"],
        )


class Client:
    """指向 InScope API 的薄客户端。

        client = Client("http://127.0.0.1:8302", api_key="...")
        result = client.detect([{"role": "user", "content": "..."}])
    """

    def __init__(self, base_url=None, api_key="", timeout=120.0):
        self.base_url = (base_url or "http://127.0.0.1:%d" % DEFAULT_API_PORT).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _client(self):
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        return httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)

    def info(self):
        with self._client() as client:
            response = client.get(ENDPOINTS["info"])
            self._raise(response)
            return response.json()

    def detect(self, messages, tools=None, request_id=None):
        body = {"messages": messages}
        if tools:
            body["tools"] = tools
        headers = {"X-Request-ID": request_id} if request_id else {}
        with self._client() as client:
            response = client.post(ENDPOINTS["detect"], json=body, headers=headers)
            self._raise(response)
            return DetectResult.from_payload(response.json())

    def ready(self):
        with self._client() as client:
            return client.get(ENDPOINTS["ready"]).status_code == 200

    @staticmethod
    def _raise(response):
        if response.status_code == 200:
            return
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise InScopeError("%d: %s" % (response.status_code, detail[:400]))
