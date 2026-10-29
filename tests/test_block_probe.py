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
