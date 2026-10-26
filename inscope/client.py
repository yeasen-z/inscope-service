"""同步/异步 HTTP 客户端。契约字段名来自 inscope_contract，不在这里重复。"""
from dataclasses import dataclass

import httpx

from inscope_contract import DEFAULT_API_PORT, ENDPOINTS


class InScopeError(Exception):
    """服务不可达、鉴权失败、或返回非法结果。"""


@dataclass(frozen=True)
class DetectResult:
    """一次检测的结果。

    probability 是探针输出经 sigmoid 后的概率；阈值单位与之一致。
    label 为 'attack' 或 'benign'，由服务端按 threshold 判定。
    """

    logit: float
    probability: float
    threshold: float
    label: str
    risk: str
    input_tokens: int
    request_id: str
    latency_ms: float = 0.0
    composer_profile: str = ""

    @property
    def flagged(self) -> bool:
        """兼容旧调用方的布尔语义。"""
        return self.label == "attack"

    @classmethod
    def from_payload(cls, payload):
        try:
            return cls(
                logit=float(payload["logit"]),
                probability=float(payload["probability"]),
                threshold=float(payload["threshold"]),
                label=str(payload["label"]),
                risk=str(payload.get("risk", "")),
                input_tokens=int(payload["input_tokens"]),
                request_id=str(payload.get("request_id", "")),
                latency_ms=float(payload.get("latency_ms", 0.0)),
                composer_profile=str(payload.get("composer_profile", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InScopeError("Malformed detect response: %s" % exc) from exc


class Client:
    """指向 InScope API 的薄客户端。

        client = Client("http://127.0.0.1:8302", api_key="...")
        result = client.detect([{"role": "user", "content": "..."}])
        ...
        client.close()          # 或 with Client(...) as client:

    连接是**复用**的：一个 Client 持有一条 keep-alive 连接链，服务端
    timeout_keep_alive 之内不会重新握手。实测每次重建连接约多花 50ms，
    在高频调用下这部分开销比推理本身还显眼。
    """

    def __init__(self, base_url=None, api_key="", timeout=120.0):
        self.base_url = (base_url or "http://127.0.0.1:%d" % DEFAULT_API_PORT).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = None

    def _client(self):
        """返回持有连接的会话；首次调用时创建，之后一直复用。"""
        if self._session is None:
            headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
            self._session = httpx.Client(base_url=self.base_url, headers=headers,
                                         timeout=self.timeout)
        return self._session

    def close(self):
        """释放连接。调用方不再使用时应当调用；重复调用是安全的。"""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def info(self):
        response = self._client().get(ENDPOINTS["info"])
        self._raise(response)
        return response.json()

    def detect(self, messages, tools=None, request_id=None):
        body = {"messages": messages}
        if tools:
            body["tools"] = tools
        headers = {"X-Request-ID": request_id} if request_id else {}
        response = self._client().post(ENDPOINTS["detect"], json=body, headers=headers)
        self._raise(response)
        return DetectResult.from_payload(response.json())

    def audit(self, messages, tools=None, *, checkpoint_roles=None,
              start_index=0, request_id=None):
        """沿轨迹逐检查点评分，返回审计时间线（含首次检出下标）。

        返回值原样是服务端 JSON：审计结果字段较多且仍在演进，
        这里不做过早的强类型收敛。超时通常需要比 detect 大得多。
        """
        body = {"messages": messages, "start_index": start_index}
        if tools:
            body["tools"] = tools
        if checkpoint_roles:
            body["checkpoint_roles"] = list(checkpoint_roles)
        headers = {"X-Request-ID": request_id} if request_id else {}
        response = self._client().post(ENDPOINTS["audit"], json=body, headers=headers)
        self._raise(response)
        return response.json()

    def ready(self):
        """服务级探活：200 才算就绪，其余状态一律视为未就绪。"""
        return self._client().get(ENDPOINTS["ready"]).status_code == 200

    @staticmethod
    def _raise(response):
        if response.status_code == 200:
            return
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise InScopeError("%d: %s" % (response.status_code, str(detail)[:400]))


class AsyncClient:
    """与 Client 等价的异步实现，用于高并发调用方。"""

    def __init__(self, base_url=None, api_key="", timeout=120.0):
        self.base_url = (base_url or "http://127.0.0.1:%d" % DEFAULT_API_PORT).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    def _ensure(self):
        if self._client is None:
            headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                             timeout=self.timeout)
        return self._client

    async def info(self):
        response = await self._ensure().get(ENDPOINTS["info"])
        Client._raise(response)
        return response.json()

    async def detect(self, messages, tools=None, request_id=None):
        body = {"messages": messages}
        if tools:
            body["tools"] = tools
        headers = {"X-Request-ID": request_id} if request_id else {}
        response = await self._ensure().post(ENDPOINTS["detect"], json=body, headers=headers)
        Client._raise(response)
        return DetectResult.from_payload(response.json())

    async def audit(self, messages, tools=None, *, checkpoint_roles=None,
                    start_index=0, request_id=None):
        """异步版轨迹审计，语义与 Client.audit 一致。"""
        body = {"messages": messages, "start_index": start_index}
        if tools:
            body["tools"] = tools
        if checkpoint_roles:
            body["checkpoint_roles"] = list(checkpoint_roles)
        headers = {"X-Request-ID": request_id} if request_id else {}
        response = await self._ensure().post(ENDPOINTS["audit"], json=body, headers=headers)
        Client._raise(response)
        return response.json()

    async def ready(self):
        response = await self._ensure().get(ENDPOINTS["ready"])
        return response.status_code == 200

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # 同步上下文里没有事件循环，交给调用方显式 aclose。
        return False
