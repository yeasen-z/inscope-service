"""block checkpoint 加载器（fyh.probe_checkpoint.v3）的拒绝路径与单元行为。"""
import hashlib
import pathlib

import pytest
import torch

from inscope_server.block_probe import decode_scaled_e4m3, encode_scaled_e4m3, load_block_probe
from inscope_server.probe import load_probe

CHECKPOINT = pathlib.Path("/mnt/kpfs_qyang/djs/ipi-doom/results/attention-head-sweep-20261027/attention-h1/best.pt")
needs_checkpoint = pytest.mark.skipif(not CHECKPOINT.is_file(), reason="ipi-doom attention-h1 checkpoint unavailable")

PROBE_ID = "attention_block:attention-h1:observation_blocks_retain1_discardcot_merged_v3"


def manifest(**overrides):
    base = {
        "schema": "fyh.probe_checkpoint.v3",
        "probe_architecture": "attention_block",
        "inscope_layers": [8, 16, 24],
        "layer_semantics": "native_residual_stream_boundary",
        "storage_codec": "e4m3_scaled_v1",
        "feature_layout": "observation_blocks",
        "composer_profile": "observation_blocks_v3",
        "empty_anchor": "attend",
        "threshold_space": "logit",
        "probe_id": PROBE_ID,
        "sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    }
    base.update(overrides)
    return base


@needs_checkpoint
def test_load_probe_dispatches_on_manifest_schema(tmp_path):
    """load_probe 读到 v3 manifest 时走 block 加载分支。"""
    import json
    target = tmp_path / "attention-h1"
    target.mkdir()
    copied = target / "best.pt"
    copied.write_bytes(CHECKPOINT.read_bytes())
    (target / "manifest.json").write_text(json.dumps(manifest()))
    loaded = load_probe(copied)
    assert loaded.feature_layout == "observation_blocks"
    assert loaded.capture_layers == (8, 16, 24)


@needs_checkpoint
def test_reject_empty_anchor_mask():
    with pytest.raises(ValueError, match="empty_anchor"):
        load_block_probe(CHECKPOINT, manifest(empty_anchor="mask"), PROBE_ID)


@needs_checkpoint
def test_reject_probability_threshold_space():
    with pytest.raises(ValueError, match="threshold_space"):
        load_block_probe(CHECKPOINT, manifest(threshold_space="sigmoid"), PROBE_ID)


@needs_checkpoint
def test_reject_sha_mismatch():
    with pytest.raises(ValueError, match="sha256"):
        load_block_probe(CHECKPOINT, manifest(sha256="0" * 64), PROBE_ID)


@needs_checkpoint
def test_reject_wrong_layers():
    with pytest.raises(ValueError, match="layers"):
        load_block_probe(CHECKPOINT, manifest(inscope_layers=[8, 16]), PROBE_ID)


def test_e4m3_codec_roundtrip():
    torch.manual_seed(7)
    values = torch.randn(5, 32) * 3
    packed, scales = encode_scaled_e4m3(values)
    decoded = decode_scaled_e4m3(packed, scales)
    # E4M3 相对误差有界（尾数 3 位，正常数量级约 6%），零行保持为零。
    assert torch.isfinite(decoded).all()
    assert torch.allclose(decoded, values, rtol=0.07, atol=1e-6)
    assert decoded[values.abs().amax(dim=-1) == 0].abs().sum() == 0


def test_e4m3_codec_rejects_nonfinite():
    values = torch.zeros(2, 8)
    values[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        encode_scaled_e4m3(values)


@needs_checkpoint
def test_score_blocks_rejects_bad_rows():
    probe = load_block_probe(CHECKPOINT, manifest(), PROBE_ID)
    rows = torch.zeros(2, 12288)
    blocks = ({"kind": "empty", "capture_slot": 5},)
    with pytest.raises(ValueError, match="capture_slot"):
        probe.score_blocks(rows, blocks)


def test_info_reports_block_probe_fields():
    """/info 必须能服务 block 探针：只在真引擎链路上才暴露过 AttributeError。"""
    import json
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from inscope_server.app import create_app

    probe = load_block_probe(CHECKPOINT, manifest(), PROBE_ID)
    settings = SimpleNamespace(
        model_path="/nonexistent", api_key="k", probe_checkpoint=str(CHECKPOINT),
        composer_profile="observation_blocks_v3", backend_url="http://127.0.0.1:1",
        backend_api_key="", max_tokens=40959, max_inflight=2, timeout=30,
        analyze_timeout=60, analyze_batch=2, audit_timeout=60, detect_batch=2,
        max_body_bytes=10 ** 7, max_batch_items=8, backend_max_inflight=2,
        probes={}, detector_model="", primary_risk="ipi")
    app = create_app(settings, tokenizer=object(), probe=probe, backend=None)
    with TestClient(app) as client:
        info = client.get("/info", headers={"authorization": "Bearer k"}).json()
    assert info["input_dim"] == 12288
    assert info["capture_layers"] == [8, 16, 24]
    assert info["selected_positions"] == []
    assert info["flatten_order"] == ["layer", "hidden"]
    assert info["probe_architecture"] == "attention_block"
    assert info["risks"][0]["probe_id"] == PROBE_ID


def test_score_ids_forwards_block_plan():
    """score_ids 必须把编译产物交给 block 打分，否则会掉进线性路径。"""
    import asyncio
    from types import SimpleNamespace

    from inscope_server.block_probe import LAYER_HIDDEN
    from inscope_server.scoring import ScoringPipeline

    probe = load_block_probe(CHECKPOINT, manifest(), PROBE_ID)
    captured = {}

    class Backend:
        async def capture(self, ids, layers, positions):
            captured["positions"] = tuple(positions)
            return [torch.zeros((len(positions), LAYER_HIDDEN)) for _ in layers]

    plan = SimpleNamespace(input_ids=[1, 2, 3], text="t", positions=(1, 2),
                           blocks=({"kind": "empty", "capture_slot": 0},
                                   {"kind": "block", "capture_slot": 1}))
    settings = SimpleNamespace(max_inflight=1, max_tokens=40959)
    pipeline = ScoringPipeline({"probe": probe, "backend": Backend()}, settings, None)
    scores = asyncio.run(pipeline.score_ids(list(plan.input_ids), plan))
    assert captured["positions"] == (1, 2)
    assert isinstance(scores, tuple) and len(scores) == 2
