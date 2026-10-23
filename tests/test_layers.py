"""层号语义测试：block index <-> SGLang residual boundary。"""
import pytest

from inscope_contract import LayerSemantics, build_layer_spec
from inscope_contract.constants import BLOCK_INDICES, CAPTURE_LAYERS

BLOCK = LayerSemantics.BLOCK_OUTPUT
BOUNDARY = LayerSemantics.BOUNDARY


def test_a5_contract_constant_matches_mapping():
    """A5 契约里的 block(8,16,24) 必须换算成 capture(9,17,25)。"""
    spec = build_layer_spec(BLOCK_INDICES, BLOCK, num_layers=36)
    assert spec.capture_layers == CAPTURE_LAYERS


def test_block_output_maps_to_next_boundary():
    spec = build_layer_spec((22,), BLOCK, num_layers=36)
    assert spec.block_indices == (22,)
    assert spec.capture_layers == (23,)
    assert spec.mapping_rule == "block_output -> boundary = block + 1"


def test_boundary_semantics_is_identity():
    spec = build_layer_spec((9, 17, 25), BOUNDARY, num_layers=36)
    assert spec.capture_layers == (9, 17, 25)
    assert spec.mapping_rule == "boundary -> boundary (identity)"


def test_missing_semantics_is_rejected():
    with pytest.raises(ValueError, match="未声明 layer_semantics"):
        build_layer_spec((22,), None)


def test_unknown_semantics_is_rejected():
    with pytest.raises(ValueError, match="unsupported layer_semantics"):
        build_layer_spec((22,), "something_else")


def test_shorthand_semantics_parsing():
    spec = build_layer_spec(
        (14,), "zero_based_decoder_block_output_before_final_RMSNorm", num_layers=36)
    assert spec.capture_layers == (15,)


def test_block_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="num_layers"):
        build_layer_spec((36,), BLOCK, num_layers=36)


def test_negative_block_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        build_layer_spec((-1,), BLOCK)


def test_duplicate_capture_layers_rejected():
    """显式构造重复层必须被拒（防止 checkpoint 里写了重复层号）。"""
    from inscope_contract.layers import LayerSpec
    with pytest.raises(ValueError, match="unique"):
        LayerSpec(block_indices=(22, 22), capture_layers=(23, 23),
                  semantics=BLOCK, mapping_rule="x")


def test_num_layers_optional_but_checked_when_given():
    spec = build_layer_spec((35,), BLOCK)
    assert spec.capture_layers == (36,)
    assert spec.describe()["mapping_rule"].startswith("block_output")
