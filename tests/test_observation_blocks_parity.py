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
    """同一份特征下服务与训练侧必须逐位一致：量化分组、归一化、attention 全在路径上。"""
    import torch
    for case in goldens["cases"]:
        raw = torch.from_numpy(_raw_features(case).copy())
        scored = block_probe.score_blocks(raw, case["blocks"], explain=True)
        assert scored["logit"] == pytest.approx(case["expected_logit"], abs=1e-6), case["name"]
        weight = [block["attention_weight"] for block in scored["blocks"]]
        contribution = [block["attention_contribution"] for block in scored["blocks"]]
        assert weight == pytest.approx(case["expected_attention_weight"], abs=1e-9), case["name"]
        assert contribution == pytest.approx(
            case["expected_attention_contribution"], abs=1e-9), case["name"]


def test_quantization_is_per_layer_not_per_slot(block_probe, goldens):
    """回归：E4M3 scale 必须按 (layer, position) 分组，不能整行 12288 维共用一个。"""
    import torch
    case = goldens["cases"][0]
    raw = torch.from_numpy(_raw_features(case).copy())
    scored = block_probe.score_blocks(raw, case["blocks"])
    layered = raw.reshape(raw.shape[0], len(block_probe.capture_layers), -1)
    from inscope_server.block_probe import encode_scaled_e4m3
    packed, scales = encode_scaled_e4m3(layered)
    assert scales.shape == (raw.shape[0], len(block_probe.capture_layers))
    assert scales.shape != (raw.shape[0],) and scales.shape != (raw.shape[0], 1)
    assert abs(scored["logit"] - case["expected_logit"]) < 1e-6


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


def test_message_sources_map_blocks_back_to_request(goldens, tokenizer):
    """块级定位必须能指回原始请求：编译会丢旧 turn、会合并连续 assistant。"""
    from inscope_server.observation_blocks import compile_observation_blocks
    for case in goldens["cases"]:
        request = case["request"]
        compiled = compile_observation_blocks(tokenizer, request["messages"],
                                             request.get("tools"))
        assert len(compiled.message_sources) > max(
            block["message_index"] for block in compiled.blocks), case["name"]
        for index, sources in enumerate(compiled.message_sources):
            assert sources, (case["name"], index)
            assert all(0 <= source < len(request["messages"]) for source in sources)
            role = request["messages"][sources[0]]["role"]
            if role == "developer":
                role = "system"
            assert role in ("system", "user", "assistant", "tool")
        flattened = [source for sources in compiled.message_sources for source in sources]
        assert flattened == sorted(flattened), "映射必须保持请求顺序"
        for block in compiled.blocks:
            assert compiled.message_sources[block["message_index"]], case["name"]


def test_message_sources_handle_dropped_and_merged(tokenizer):
    """两个真实失真来源：选窗丢消息造成位移，合并 assistant 造成一对多。"""
    from inscope_server.observation_blocks import compile_observation_blocks
    dropped = compile_observation_blocks(tokenizer, [
        {"role": "user", "content": "task one"},
        {"role": "assistant", "content": "done one"},
        {"role": "user", "content": "task two"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "read", "content": "B1"},
    ])
    assert dropped.message_sources == ((0,), (3,), (4,))
    assert {block["message_index"] for block in dropped.blocks} == {2}

    merged = compile_observation_blocks(tokenizer, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "thinking out loud"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c3", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "read", "content": "C1"},
    ])
    assert merged.message_sources == ((0,), (1, 2), (3,))
