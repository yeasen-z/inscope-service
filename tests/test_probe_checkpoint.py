"""Probe 必须完全由 checkpoint 自描述决定，不允许任何写死的 A5 假设。"""

import json

import pytest
import torch

from inscope_server.probe import load_probe

LAYERS = 3
POSITIONS = 4
HIDDEN = 4096
DIM = LAYERS * POSITIONS * HIDDEN


def _state(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "input_mean": torch.randn(1, DIM, generator=g) * 10,
        "input_std": torch.rand(1, DIM, generator=g) + 0.5,
        "probe.weight": torch.randn(1, DIM, generator=g) * 0.01,
        "probe.bias": torch.tensor([-0.1]),
    }


def _write(tmp_path, *, schema="fyh.probe_checkpoint.v2", arch="linear",
           blocks=(8, 16, 24), positions=(-4, -3, -2, -1), dim=DIM,
           manifest="auto", state=None):
    ck = {
        "schema": schema,
        "probe_architecture": arch,
        "input_dim": dim,
        "selected_layer_indices": list(blocks),
        "selected_positions": list(positions),
        "feature_normalization": "standard",
        "threshold": 0.9,
        "labeling_protocol": "risk_faced",
        "feature_name": "no_prefix_cot_last4_20261015",
        "model_state_dict": state if state is not None else _state(),
    }
    path = tmp_path / "best.pt"
    torch.save(ck, path)
    if manifest == "auto":
        # 与真实 handoff 一致：由 manifest 显式给出边界层号。
        manifest = {"inscope_layers": [b + 1 for b in blocks],
                    "flatten_order": ["batch", "layer", "token", "hidden"]}
    if manifest is not None:
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return path


def test_loads_real_shaped_checkpoint(tmp_path):
    probe = load_probe(_write(tmp_path))
    assert probe.input_dim == DIM
    assert tuple(probe.selected_positions) == (-4, -3, -2, -1)
    assert probe.threshold == 0.9


def test_manifest_layers_take_precedence(tmp_path):
    """manifest 显式给出 inscope_layers 时直接采用。"""
    path = _write(tmp_path, manifest={"inscope_layers": [9, 17, 25],
                                      "flatten_order": ["batch", "layer", "token", "hidden"]})
    assert tuple(load_probe(path).capture_layers) == (9, 17, 25)


def test_block_indices_convert_with_plus_one(tmp_path):
    path = _write(tmp_path, manifest=None)
    # 没有 layer_semantics 且无 manifest 时拒绝；补上语义后应 +1。
    with pytest.raises(ValueError):
        load_probe(path)
    ck = torch.load(path, weights_only=False)
    ck["layer_semantics"] = "zero_based_decoder_block_output_before_final_RMSNorm"
    torch.save(ck, path)
    assert tuple(load_probe(path).capture_layers) == (9, 17, 25)


def test_manifest_sha256_is_enforced(tmp_path):
    path = _write(tmp_path, manifest={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="sha256"):
        load_probe(path)


def test_unknown_schema_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        load_probe(_write(tmp_path, schema="something.else.v1"))


def test_missing_layer_semantics_without_manifest_is_rejected(tmp_path):
    """checkpoint 未声明语义又没有 manifest 时，必须报错而不是猜。"""
    with pytest.raises(ValueError):
        load_probe(_write(tmp_path, manifest=None))


def test_dim_must_match_layers_positions_hidden(tmp_path):
    """input_dim 与 层数x位置数xhidden 不符时必须拒绝。"""
    half = DIM // 2
    g = torch.Generator().manual_seed(1)
    bad_state = {
        "input_mean": torch.zeros(1, half),
        "input_std": torch.ones(1, half),
        "probe.weight": torch.randn(1, half, generator=g),
        "probe.bias": torch.tensor([0.0]),
    }
    with pytest.raises(ValueError, match="input_dim|Normalization"):
        load_probe(_write(tmp_path, dim=half, state=bad_state))


def test_unsupported_architecture_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="probe_architecture"):
        load_probe(_write(tmp_path, arch="transformer"))


def test_missing_checkpoint_reports_path(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_probe(tmp_path / "absent.pt")


def test_flatten_orders_layer_major(tmp_path):
    """flatten 必须按 [layer, token, hidden] 铺平。"""
    probe = load_probe(_write(tmp_path))
    blocks = [torch.full((POSITIONS, HIDDEN), float(i)) for i in range(LAYERS)]
    flat = probe.flatten(blocks)
    assert tuple(flat.shape) == (1, DIM)
    assert flat[0, 0].item() == 0.0
    assert flat[0, DIM - 1].item() == float(LAYERS - 1)


def test_flatten_rejects_wrong_layer_count(tmp_path):
    probe = load_probe(_write(tmp_path))
    with pytest.raises(ValueError, match="layer blocks"):
        probe.flatten([torch.zeros(POSITIONS, HIDDEN)] * 2)


def test_score_rejects_wrong_feature_dim(tmp_path):
    probe = load_probe(_write(tmp_path))
    with pytest.raises(ValueError, match="Feature dim"):
        probe.score(torch.zeros(1, DIM - 1))


def test_score_is_deterministic_and_bounded(tmp_path):
    probe = load_probe(_write(tmp_path))
    x = torch.randn(1, DIM)
    assert probe.score(x) == probe.score(x)
    assert 0.0 <= probe.score(x)[1] <= 1.0
