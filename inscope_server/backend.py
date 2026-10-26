import asyncio
from pathlib import Path
import httpx
import torch

from inscope_contract import (
    CAPTURE_SCHEMA_VERSION,
    CAPTURE_SITE,
    FEATURE_ENCODING,
    LAYER_HIDDEN,
    SELECTED_POSITIONS,
)


class BackendError(Exception):
    pass


def decode_features(result, layers, positions):
    """Never score missing, duplicate, wrong-position, or malformed captures."""
    try:
        records = result["meta_info"]["features"]
        if not isinstance(records, list) or len(records) != len(layers):
            raise ValueError("Expected one record per requested layer")
        # 位置数量由探针决定，不能假定只有单个位置。
        wanted = list(positions)
        expected_shape = [len(wanted), LAYER_HIDDEN]
        by_layer = {}
        for r in records:
            if (r["schema_version"] != CAPTURE_SCHEMA_VERSION
                    or type(r["layer"]) is not int or r["layer"] not in layers or r["layer"] in by_layer
                    or r["site"] != CAPTURE_SITE or r["positions"] != wanted
                    or r["shape"] != expected_shape or r["encoding"] != FEATURE_ENCODING
                    or r["dtype"] not in ("bfloat16", "float16", "float32")):
                raise ValueError("Feature contract mismatch")
            h = torch.tensor(r["data"], dtype=torch.float32)
            if h.shape != tuple(expected_shape) or not torch.isfinite(h).all():
                raise ValueError("Invalid feature data")
            by_layer[r["layer"]] = h
        # 按请求的 layers 顺序 stack；不能依赖响应 records 的排列顺序。
        return [by_layer[layer] for layer in layers]
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise BackendError("Invalid InScope feature response") from exc


class InScopeBackend:
    def __init__(self, settings, client=None):
        self.settings = settings
        headers = ({"Authorization": "Bearer " + settings.backend_api_key}
                   if settings.backend_api_key else {})
        self.client = client or httpx.AsyncClient(
            base_url=settings.backend_url, headers=headers, timeout=settings.timeout,
            limits=httpx.Limits(max_connections=settings.max_inflight + 1), trust_env=False)
        self._verified = False

    async def capture(self, ids, layers, positions=None):
        if len(ids) < 1:
            raise BackendError("Capture requires at least one prompt token")
        results = await self.capture_batch([ids], layers, positions)
        return results[0]

    async def capture_batch(self, batch_ids, layers, positions=None):
        """一次 /generate 取回多份 prompt 的层特征，按输入顺序返回。

        批量协议（已用真实引擎验证）：
            请求 = 单个对象，``input_ids`` 为二维列表；``feature_capture``
                  是一个 spec，由该批全部条目共用。
            响应 = 顶层数组，每项对应一个输入，顺序与请求一致。

        负向位置按各自长度解析后，不同长度的 prompt 会落在不同的绝对下标上。
        由于 spec 是整批共享的，必须按"解析后的位置"分组，逐组发请求
        （各组并发），最后按原始顺序拼回——否则会拿 A 的位置读 B 的序列。
        """
        batch = [list(ids) for ids in batch_ids]
        if not batch:
            raise BackendError("Capture requires at least one prompt")
        if any(len(ids) < 1 for ids in batch):
            raise BackendError("Capture requires at least one prompt token per request")
        try:
            async with asyncio.timeout(self.settings.timeout):
                await self._verify_model()
                active = tuple(int(layer) for layer in layers)
                wanted = tuple(int(p) for p in (positions or SELECTED_POSITIONS))
                groups = {}
                for position, ids in enumerate(batch):
                    index = tuple(p if p >= 0 else len(ids) + p for p in wanted)
                    if any(i < 0 or i >= len(ids) for i in index):
                        raise BackendError(
                            "Capture position out of range for %d tokens" % len(ids))
                    groups.setdefault(index, []).append(position)
                if len(groups) == 1:
                    only = next(iter(groups))
                    return await self._capture_group(batch, active, only)
                # 位置不同无法合并成一批，逐组并发后按原顺序还原。
                async def run(index, members):
                    return index, await self._capture_group(
                        [batch[m] for m in members], active, index)

                results = [None] * len(batch)
                for index, features in await asyncio.gather(
                        *(run(i, m) for i, m in groups.items())):
                    for slot, position in enumerate(groups[index]):
                        results[position] = features[slot]
                return results
        except (httpx.TimeoutException, TimeoutError):
            raise TimeoutError("SGLang request timed out") from None
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError("SGLang unavailable or returned invalid JSON") from exc

    async def _capture_group(self, batch, active, index):
        """对共享同一采集位置的一组 prompt 发一次批量 /generate。"""
        payload = {
            "input_ids": batch if len(batch) > 1 else batch[0],
            "sampling_params": {"max_new_tokens": 0},
            "feature_capture": {"layers": list(active), "site": "block_output",
                                "positions": {"mode": "explicit", "index": list(index)},
                                "encoding": "float_list"},
        }
        response = await self.client.post("/generate", json=payload)
        response.raise_for_status()
        body = response.json()
        items = body if isinstance(body, list) else [body]
        if len(items) != len(batch):
            raise BackendError("Batch response does not match %d requests" % len(batch))
        return [decode_features(item, active, index) for item in items]

    async def _verify_model(self):
        """首次请求校验模型身份；结果缓存到本实例关闭。

        注意：HTTP 连接重建或远端引擎重启不会清除 _verified。
        当前不能据此保证运行期间模型未被替换；修复缓存策略需单独验证。
        """
        if self._verified:
            return
        info_response = await self.client.get("/model_info")
        info_response.raise_for_status()
        info = info_response.json()
        model_path = str(Path(self.settings.model_path).resolve())
        if (not isinstance(info, dict) or not isinstance(info.get("model_path"), str)
                or (info.get("tokenizer_path") is not None
                    and not isinstance(info["tokenizer_path"], str))):
            raise BackendError("Invalid SGLang model identity")
        if (info.get("model_type") != "qwen3"
                or str(Path(info.get("model_path", "")).resolve()) != model_path
                or (info.get("tokenizer_path") and
                    str(Path(info["tokenizer_path"]).resolve()) != model_path)):
            raise BackendError("SGLang model/tokenizer differs from MODEL_PATH")
        self._verified = True

    async def close(self):
        self._verified = False
        await self.client.aclose()
