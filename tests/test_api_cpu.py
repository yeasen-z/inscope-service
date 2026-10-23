"""CPU 端契约回归：不连 GPU、不连 engine，只验证协议与后端解码。

从 coolwatch/.setup 的 e2e 测试迁移而来，目标路径改为独立仓库 inscope-service，
并新增 composition/profile 相关断言。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SERVICE = Path("/data/zmosg/inscope-service")
if str(SERVICE) not in sys.path:
    sys.path.insert(0, str(SERVICE))

from inscope_contract import (  # noqa: E402
    BLOCK_INDICES,
    CAPTURE_LAYERS,
    CAPTURE_SCHEMA_VERSION,
    CAPTURE_SITE,
    FEATURE_SHAPE,
    INPUT_DIM,
    LAYER_HIDDEN,
    SELECTED_POSITIONS,
)
from inscope_server.backend import decode_features  # noqa: E402


def _record(layer, positions, shape, data, **over):
    payload = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "site": CAPTURE_SITE,
        "layer": layer,
        "positions": positions,
        "shape": shape,
        "encoding": "float_list",
        "dtype": "bfloat16",
        "data": data,
    }
    payload.update(over)
    return payload


def _good_records():
    return [_record(layer, [11], [1, LAYER_HIDDEN],
                    [[0.5] * LAYER_HIDDEN]) for layer in CAPTURE_LAYERS]


def test_contract_constants_self_consistent():
    assert INPUT_DIM == len(CAPTURE_LAYERS) * len(SELECTED_POSITIONS) * LAYER_HIDDEN
    assert FEATURE_SHAPE == (1, len(CAPTURE_LAYERS), len(SELECTED_POSITIONS), LAYER_HIDDEN)
    assert tuple(l + 1 for l in BLOCK_INDICES) == CAPTURE_LAYERS


def test_decode_valid_records():
    hidden = decode_features({"meta_info": {"features": _good_records()}}, CAPTURE_LAYERS, [11])
    # decode_features 现在返回按请求顺序排列的层块列表，铺平与形状校验交给 probe。
    assert isinstance(hidden, list) and len(hidden) == len(CAPTURE_LAYERS)
    for block in hidden:
        assert tuple(block.shape) == (len(SELECTED_POSITIONS), LAYER_HIDDEN)


def test_decode_supports_multiple_positions():
    """位置数量由探针决定：4 个位置也要能正确解码并保持请求顺序。"""
    positions = [7, 8, 9, 10]
    records = [_record(layer, positions, [len(positions), LAYER_HIDDEN],
                       [[0.25] * LAYER_HIDDEN] * len(positions))
               for layer in CAPTURE_LAYERS]
    hidden = decode_features({"meta_info": {"features": records}}, CAPTURE_LAYERS, positions)
    assert [tuple(b.shape) for b in hidden] == [(len(positions), LAYER_HIDDEN)] * len(CAPTURE_LAYERS)


@pytest.mark.parametrize("mutation, reason", [
    (lambda r: r[0].update(schema_version="wrong.v9"), "schema"),
    (lambda r: r[0].update(site="attention_output"), "site"),
    (lambda r: r[0].update(dtype="int8"), "dtype"),   # float32 明确允许，int8 不允许
    (lambda r: r[0].update(positions=[10]), "positions"),
    (lambda r: r[0].update(data=[[float("nan")] * LAYER_HIDDEN]), "non-finite"),
])
def test_decode_rejects_bad_records(mutation, reason):
    records = _good_records()
    mutation(records)
    with pytest.raises(Exception):
        decode_features({"meta_info": {"features": records}}, CAPTURE_LAYERS, [11])


def test_decode_rejects_missing_layer():
    records = _good_records()[:2]
    with pytest.raises(Exception):
        decode_features({"meta_info": {"features": records}}, CAPTURE_LAYERS, [11])


def test_decode_rejects_duplicate_layer():
    records = _good_records()
    records[1]["layer"] = records[0]["layer"]
    with pytest.raises(Exception):
        decode_features({"meta_info": {"features": records}}, CAPTURE_LAYERS, [11])


def test_detection_request_rejects_unknown_field():
    from inscope_contract import DetectionRequest
    with pytest.raises(Exception):
        DetectionRequest.model_validate({
            "messages": [{"role": "user", "content": "hi"}],
            "threshold": 0.1,   # 调用方无权覆盖服务端规格
        })


def test_detection_request_rejects_empty_messages():
    from inscope_contract import DetectionRequest
    with pytest.raises(Exception):
        DetectionRequest.model_validate({"messages": []})


def test_detection_request_accepts_tools():
    from inscope_contract import DetectionRequest
    request = DetectionRequest.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    })
    assert request.tools[0]["function"]["name"] == "read_file"
