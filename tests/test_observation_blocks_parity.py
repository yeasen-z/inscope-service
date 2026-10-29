"""observation_blocks 编译与打分的黄金样本对拍。

基准是 ipi-doom 训练侧实现（tests/generate_ob_blocks_goldens.py 执行）：
- 编译：featurize.observation_blocks.build_block_prompt（canonical simplified-v3）
- 打分：probe_training.debug.ProbeScorer.score_features（E4M3 量化往返后 attention）

对拍逐项一致：compiled text、token_ids、capture positions、block 元数据、
DP logit、per-block attention_weight / attention_contribution。
黄金样本由 ipi-doom 侧生成；编译规则任何变更都必须重新生成并双方 review。
"""
import base64
import json
import pathlib

import pytest

from inscope_server.block_probe import load_block_probe
from inscope_server.observation_blocks import compile_observation_blocks

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "observation_blocks_golden.json"
MODEL = "/mnt/kpfs_qyang/djs/models/Qwen3-8B-FP8"
CHECKPOINT = pathlib.Path("/mnt/kpfs_qyang/djs/ipi-doom/results/attention-head-sweep-20261027/attention-h1/best.pt")

pytestmark = pytest.mark.skipif(not pathlib.Path(MODEL).is_dir()
                                or not FIXTURES.is_file() or not CHECKPOINT.is_file(),
                                reason="Qwen3 tokenizer / golden fixtures unavailable")

PROBE_ID = "attention_block:attention-h1:observation_blocks_retain1_discardcot_merged_v3"


@pytest.fixture(scope="module")
def goldens():
    return json.loads(FIXTURES.read_text())


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


@pytest.fixture(scope="module")
def block_probe():
    import hashlib
    manifest = {
        "schema": "fyh.probe_checkpoint.v3",
        "probe_architecture": "attention_block",
        "inscope_layers": [8, 16, 24],
        "layer_semantics": "native_residual_stream_boundary",
        "storage_codec": "e4m3_scaled_v1",
        "feature_layout": "observation_blocks",
        "composer_profile": "observation_blocks_v3",
        "empty_anchor": "attend",
        "threshold": 3.7808122634887695,
        "threshold_space": "logit",
        "probe_id": PROBE_ID,
        "sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    }
    return load_block_probe(CHECKPOINT, manifest, PROBE_ID)


def _raw_features(case):
    import numpy as np
    payload = base64.b64decode(case["raw_features_f32_b64"])
    return np.frombuffer(payload, dtype="<f4").reshape(len(case["positions"]), -1)


def test_compile_matches_ipi_doom(tokenizer, goldens):
    for case in goldens["cases"]:
        compiled = compile_observation_blocks(
            tokenizer, case["request"]["messages"], case["request"].get("tools"))
        assert compiled.text == case["text"], case["name"]
        assert list(compiled.input_ids) == case["token_ids"], case["name"]
        assert list(compiled.positions) == case["positions"], case["name"]
        assert [dict(block) for block in compiled.blocks] == case["blocks"], case["name"]


def test_score_matches_ipi_doom(block_probe, goldens):
    import torch
    for case in goldens["cases"]:
        raw = torch.from_numpy(_raw_features(case).copy())
        scored = block_probe.score_blocks(raw, case["blocks"], explain=True)
        assert scored["logit"] == pytest.approx(case["expected_logit"], abs=1e-5), case["name"]
        weight = [block["attention_weight"] for block in scored["blocks"]]
        contribution = [block["attention_contribution"] for block in scored["blocks"]]
        assert weight == pytest.approx(case["expected_attention_weight"], abs=1e-5), case["name"]
        assert contribution == pytest.approx(
            case["expected_attention_contribution"], abs=1e-5), case["name"]


def test_empty_anchor_attends_like_training(block_probe, goldens):
    """empty anchor 参与 softmax 且出现在通道里：训练侧不对它置零。"""
    import torch
    case = goldens["cases"][0]
    raw = torch.from_numpy(_raw_features(case).copy())
    scored = block_probe.score_blocks(raw, case["blocks"], explain=True)
    empty = [block for block in scored["blocks"] if block["kind"] == "empty"]
    assert empty, "该样本必须包含 empty anchor 才能覆盖此语义"
    assert any(block["attention_weight"] > 0.0 for block in empty)
    total = sum(block["attention_weight"] for block in scored["blocks"])
    assert total == pytest.approx(1.0, abs=1e-5)


def test_threshold_is_sigmoid_of_logit(block_probe):
    import torch
    assert block_probe.threshold_logit == pytest.approx(3.7808122634887695, abs=1e-7)
    assert block_probe.threshold == pytest.approx(
        float(torch.sigmoid(torch.tensor(3.7808122634887695))), abs=1e-7)
    assert block_probe.capture_layers == (8, 16, 24)
