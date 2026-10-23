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
            raise ValueError("Expected exactly three layer records")
        by_layer = {}
        for r in records:
            if (r["schema_version"] != CAPTURE_SCHEMA_VERSION
                    or type(r["layer"]) is not int or r["layer"] not in layers or r["layer"] in by_layer
                    or r["site"] != CAPTURE_SITE or r["positions"] != list(positions)
                    or r["shape"] != [1, LAYER_HIDDEN] or r["encoding"] != FEATURE_ENCODING
                    or r["dtype"] not in ("bfloat16", "float16", "float32")):
                raise ValueError("Feature contract mismatch")
            h = torch.tensor(r["data"], dtype=torch.float32)
            if h.shape != (1, LAYER_HIDDEN) or not torch.isfinite(h).all():
                raise ValueError("Invalid feature data")
            by_layer[r["layer"]] = h
        # 按请求的 layers 顺序 stack；不能依赖响应 records 的排列顺序。
        return torch.stack([by_layer[layer] for layer in layers], dim=0).unsqueeze(0)
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

    async def capture(self, ids, layers):
        if len(ids) < 1:
            raise BackendError("Capture requires at least one prompt token")
        try:
            async with asyncio.timeout(self.settings.timeout):
                # Verify the running model, including after backend restarts or a port change.
                info_response = await self.client.get("/model_info")
                info_response.raise_for_status()
                info = info_response.json()
                model_path = str(Path(self.settings.model_path).resolve())
                if (not isinstance(info, dict) or not isinstance(info.get("model_path"), str)
                        or (info.get("tokenizer_path") is not None and not isinstance(info["tokenizer_path"], str))):
                    raise BackendError("Invalid SGLang model identity")
                if (info.get("model_type") != "qwen3"
                        or str(Path(info.get("model_path", "")).resolve()) != model_path
                        or (info.get("tokenizer_path") and
                            str(Path(info["tokenizer_path"]).resolve()) != model_path)):
                    raise BackendError("SGLang model/tokenizer differs from MODEL_PATH")
                response = await self.client.post("/generate", json={
                    "input_ids": ids,
                    "sampling_params": {"max_new_tokens": 0},
                    "feature_capture": {"layers": list(layers), "site": "block_output",
                                        "positions": {"mode": "explicit", "index": list(SELECTED_POSITIONS)},
                                        "encoding": "float_list"},
                })
                response.raise_for_status()
                return decode_features(response.json(), layers, range(len(ids)-1, len(ids)))
        except (httpx.TimeoutException, TimeoutError):
            raise TimeoutError("SGLang request timed out") from None
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError("SGLang unavailable or returned invalid JSON") from exc

    async def close(self):
        await self.client.aclose()
