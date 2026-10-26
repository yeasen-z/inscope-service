"""Probe 必须完全由 checkpoint 自描述决定，不允许任何写死的 A5 假设。"""

import pytest
import torch

from inscope_server.probe import load_probe

from conftest import DIM, HIDDEN, LAYERS, POSITIONS, write_checkpoint


def test_loads_real_shaped_checkpoint(tmp_path):
    probe = load_probe(write_checkpoint(tmp_path))
    assert probe.input_dim == DIM
    assert tuple(probe.selected_positions) == (-4, -3, -2, -1)
    assert probe.threshold == 0.9


def test_manifest_layers_take_precedence(tmp_path):
    """manifest 显式给出 inscope_layers 时直接采用。"""
    path = write_checkpoint(tmp_path, manifest={"inscope_layers": [9, 17, 25],
                                                "flatten_order": ["batch", "layer",
                                                                  "token", "hidden"]})
    assert tuple(load_probe(path).capture_layers) == (9, 17, 25)


def test_block_indices_convert_with_plus_one(tmp_path):
    path = write_checkpoint(tmp_path, manifest=None)
    # 没有 layer_semantics 且无 manifest 时拒绝；补上语义后应 +1。
    with pytest.raises(ValueError):
        load_probe(path)
    ck = torch.load(path, weights_only=False)
    ck["layer_semantics"] = "zero_based_decoder_block_output_before_final_RMSNorm"
    torch.save(ck, path)
    assert tuple(load_probe(path).capture_layers) == (9, 17, 25)


def test_manifest_sha256_is_enforced(tmp_path):
    path = write_checkpoint(tmp_path, manifest={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="sha256"):
        load_probe(path)


def test_unknown_schema_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        load_probe(write_checkpoint(tmp_path, schema="something.else.v1"))


def test_missing_layer_semantics_without_manifest_is_rejected(tmp_path):
    """checkpoint 未声明语义又没有 manifest 时，必须报错而不是猜。"""
    with pytest.raises(ValueError):
        load_probe(write_checkpoint(tmp_path, manifest=None))


def test_dim_must_match_layers_positions_hidden(tmp_path):
    """input_dim 与 层数x位置数xhidden 不符时必须拒绝。"""
    half = DIM // 2
    with pytest.raises(ValueError, match="input_dim|Normalization"):
        load_probe(write_checkpoint(tmp_path, dim=half))


def test_unsupported_architecture_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="probe_architecture"):
        load_probe(write_checkpoint(tmp_path, arch="transformer"))


def test_missing_checkpoint_reports_path(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_probe(tmp_path / "absent.pt")


def test_flatten_orders_layer_major(tmp_path):
    """flatten 必须按 [layer, token, hidden] 铺平。"""
    probe = load_probe(write_checkpoint(tmp_path))
    blocks = [torch.full((POSITIONS, HIDDEN), float(i)) for i in range(LAYERS)]
    flat = probe.flatten(blocks)
    assert tuple(flat.shape) == (1, DIM)
    assert flat[0, 0].item() == 0.0
    assert flat[0, DIM - 1].item() == float(LAYERS - 1)


def test_flatten_rejects_wrong_layer_count(tmp_path):
    probe = load_probe(write_checkpoint(tmp_path))
    with pytest.raises(ValueError, match="layer blocks"):
        probe.flatten([torch.zeros(POSITIONS, HIDDEN)] * 2)


def test_flatten_rejects_wrong_position_count(tmp_path):
    """位置数与 checkpoint 声明不符时必须拒绝，而不是静默截断。"""
    probe = load_probe(write_checkpoint(tmp_path))
    blocks = [torch.zeros(POSITIONS - 1, HIDDEN) for _ in range(LAYERS)]
    with pytest.raises((ValueError, RuntimeError)):
        probe.flatten(blocks)


def test_score_rejects_wrong_feature_dim(tmp_path):
    probe = load_probe(write_checkpoint(tmp_path))
    with pytest.raises(ValueError, match="Feature dim"):
        probe.score(torch.zeros(1, DIM - 1))


def test_score_rejects_non_finite_features(tmp_path):
    probe = load_probe(write_checkpoint(tmp_path))
    bad = torch.zeros(1, DIM)
    bad[0, 3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        probe.score(bad)


def test_score_is_deterministic_and_bounded(tmp_path):
    probe = load_probe(write_checkpoint(tmp_path))
    x = torch.randn(1, DIM)
    assert probe.score(x) == probe.score(x)
    assert 0.0 <= probe.score(x)[1] <= 1.0


def test_checkpoint_sha256_is_reported(tmp_path):
    import hashlib
    path = write_checkpoint(tmp_path)
    assert load_probe(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01])
def test_rejects_invalid_probability_threshold(tmp_path, threshold):
    path = write_checkpoint(tmp_path)
    ck = torch.load(path, weights_only=False)
    ck["threshold"] = threshold
    torch.save(ck, path)
    with pytest.raises(ValueError, match="threshold.*finite.*0..1"):
        load_probe(path)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_accepts_threshold_endpoints(tmp_path, threshold):
    path = write_checkpoint(tmp_path)
    ck = torch.load(path, weights_only=False)
    ck["threshold"] = threshold
    torch.save(ck, path)
    assert load_probe(path).threshold == threshold


def test_mlp_is_not_silently_loaded_as_linear(tmp_path):
    with pytest.raises(ValueError, match="probe_architecture.*mlp"):
        load_probe(write_checkpoint(tmp_path, arch="mlp"))


@pytest.mark.parametrize("layers", [
    [9, 9, 25], [-1, 17, 25], [9, 17, 37],
    [9.5, 17, 25], ["9", 17, 25], [True, 17, 25], [],
])
def test_rejects_invalid_manifest_capture_layers(tmp_path, layers):
    path = write_checkpoint(tmp_path, manifest={"inscope_layers": layers})
    with pytest.raises(ValueError, match="inscope_layers"):
        load_probe(path)


def test_accepts_boundary_layer_endpoints(tmp_path):
    path = write_checkpoint(tmp_path, manifest={"inscope_layers": [0, 17, 36]})
    assert tuple(load_probe(path).capture_layers) == (0, 17, 36)


@pytest.mark.parametrize("blocks", [(-1, 16, 24), (8, 16, 36), (8, 8, 24), (8.5, 16, 24)])
def test_rejects_invalid_block_layers_without_manifest(tmp_path, blocks):
    path = write_checkpoint(tmp_path, blocks=blocks, manifest=None)
    ck = torch.load(path, weights_only=False)
    ck["layer_semantics"] = "zero_based_decoder_block_output_before_final_RMSNorm"
    torch.save(ck, path)
    with pytest.raises(ValueError, match="block indices"):
        load_probe(path)


@pytest.mark.parametrize("positions", [(-4, -3, -2, -2), (-4, -3, -2, -1.5),
                                         (-4, -3, -2, "-1"), (-4, -3, -2, True)])
def test_rejects_duplicate_or_noninteger_positions(tmp_path, positions):
    with pytest.raises(ValueError, match="selected_positions"):
        load_probe(write_checkpoint(tmp_path, positions=positions))


def test_accepts_distinct_absolute_and_relative_positions(tmp_path):
    positions = (0, 1, -2, -1)
    probe = load_probe(write_checkpoint(tmp_path, positions=positions))
    assert tuple(probe.selected_positions) == positions


@pytest.mark.parametrize("kind", ["cast", "subtract", "divide"])
def test_rejects_nonfinite_normalized_features(tmp_path, kind):
    probe = load_probe(write_checkpoint(tmp_path))
    x = torch.zeros(1, DIM)
    if kind == "cast":
        x = x.double()
        x[0, 0] = 1e300  # finite float64, overflows the float32 normalization
    elif kind == "subtract":
        x[0, 0] = torch.finfo(torch.float32).max
        probe.mean[0] = -torch.finfo(torch.float32).max
    else:
        x[0, 0] = 1.0
        probe.std[0] = torch.finfo(torch.float32).tiny / 16
    assert torch.isfinite(x).all()
    with pytest.raises(ValueError, match="Normalized.*non-finite"):
        probe.score(x)


def test_rejects_overflowing_linear_output(tmp_path):
    from conftest import make_state

    state = make_state()
    state["input_mean"].zero_()
    state["input_std"].fill_(1)
    state["probe.weight"].fill_(torch.finfo(torch.float32).max)
    probe = load_probe(write_checkpoint(tmp_path, state=state))
    with pytest.raises(ValueError, match="output.*non-finite"):
        probe.score(torch.ones(1, DIM))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_module_output(tmp_path, value):
    probe = load_probe(write_checkpoint(tmp_path))

    class Nonfinite(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[value]])

    probe.module = Nonfinite()
    with pytest.raises(ValueError, match="output.*non-finite"):
        probe.score(torch.zeros(1, DIM))


def test_missing_linear_weight_is_a_validation_error(tmp_path):
    from conftest import make_state

    state = make_state()
    del state["probe.weight"]
    with pytest.raises(ValueError, match="missing tensor"):
        load_probe(write_checkpoint(tmp_path, state=state))


@pytest.mark.parametrize("layers", [(-1, 17, 25), (9, 17, 37), (9, 9, 25), (9.5, 17, 25)])
def test_rejects_invalid_boundary_layers_without_manifest(tmp_path, layers):
    path = write_checkpoint(tmp_path, blocks=layers, manifest=None)
    ck = torch.load(path, weights_only=False)
    ck["layer_semantics"] = "residual_stream_boundary"
    torch.save(ck, path)
    with pytest.raises(ValueError, match="layer indices"):
        load_probe(path)


@pytest.mark.parametrize("semantics,declared,expected", [
    ("residual_stream_boundary", (36, 0, 17), (36, 0, 17)),
    ("zero_based_decoder_block_output_before_final_RMSNorm", (35, 0, 16), (36, 1, 17)),
])
def test_accepts_valid_layer_endpoints_and_preserves_order(tmp_path, semantics, declared, expected):
    path = write_checkpoint(tmp_path, blocks=declared, manifest=None)
    ck = torch.load(path, weights_only=False)
    ck["layer_semantics"] = semantics
    torch.save(ck, path)
    assert tuple(load_probe(path).capture_layers) == expected
