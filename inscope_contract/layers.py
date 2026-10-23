"""层号语义：探针 checkpoint（block index）与 SGLang（residual boundary）的唯一换算点。

为什么单独一个模块
------------------
三层仓库各自对「层」有自己的说法：

  sglang-inscope  用 residual-stream boundary：
                  L=0 embedding 输出，1..N-1 是第 L-1 个 block 输出，N 是最后一层
  探针 checkpoint 用 decoder block index（第几个 block），不同 checkpoint 字段名还不一样
  调用方          只关心「探针要的层」

换算是跨仓库的数值约定，散在任何一处实现里都会漂移，所以集中在这里。
**任何其它模块都不允许自己写 block+1。**

语义声明
--------
换算只在 checkpoint 显式声明语义时执行。缺声明 -> 报错，不默认成 block index。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LayerSemantics(str, Enum):
    """checkpoint 里 layer 字段的含义。"""

    # 第 b 个 decoder block 的输出，在 final RMSNorm 之前（最常见的训练侧说法）
    BLOCK_OUTPUT = "zero_based_decoder_block_output_before_final_RMSNorm"
    # 已经是 SGLang 的 residual-stream boundary 编号
    BOUNDARY = "residual_stream_boundary"

    @classmethod
    def parse(cls, value: str | None) -> "LayerSemantics":
        """宽松解析 checkpoint 的 layer_semantics 字段；不认识就报错。"""
        if value is None:
            raise ValueError(
                "checkpoint 未声明 layer_semantics，无法区分 block index 与 residual "
                "boundary；拒绝猜测换算规则"
            )
        text = str(value).strip()
        for member in cls:
            if text == member.value:
                return member
        # 允许省略后缀的简写，但必须是明确的子串
        if "decoder_block_output" in text:
            return cls.BLOCK_OUTPUT
        if "residual_stream_boundary" in text or text == "boundary":
            return cls.BOUNDARY
        raise ValueError(f"unsupported layer_semantics: {value!r}")


@dataclass(frozen=True)
class LayerSpec:
    """一次特征采集的层规格；构造即校验。"""

    block_indices: tuple[int, ...]
    capture_layers: tuple[int, ...]
    semantics: LayerSemantics
    mapping_rule: str
    num_layers: int | None = None

    def __post_init__(self) -> None:
        if not self.block_indices:
            raise ValueError("block_indices must be non-empty")
        if len(self.block_indices) != len(self.capture_layers):
            raise ValueError("block_indices and capture_layers must have equal length")
        if len(set(self.capture_layers)) != len(self.capture_layers):
            raise ValueError(f"capture_layers must be unique, got {self.capture_layers}")
        for layer in self.capture_layers:
            if type(layer) is not int or layer < 0:
                raise ValueError(f"capture layer must be a non-negative int, got {layer!r}")
            if self.num_layers is not None and layer > self.num_layers:
                raise ValueError(
                    f"capture layer {layer} exceeds num_layers {self.num_layers}"
                )

    def describe(self) -> dict:
        payload = {
            "block_indices": list(self.block_indices),
            "capture_layers": list(self.capture_layers),
            "layer_semantics": self.semantics.value,
            "mapping_rule": self.mapping_rule,
        }
        if self.num_layers is not None:
            payload["num_layers"] = self.num_layers
        return payload


_RULE = {
    LayerSemantics.BLOCK_OUTPUT: "block_output -> boundary = block + 1",
    LayerSemantics.BOUNDARY: "boundary -> boundary (identity)",
}


def build_layer_spec(
    block_indices,
    semantics: LayerSemantics | str,
    *,
    num_layers: int | None = None,
) -> LayerSpec:
    """把探针声明的层号换算成 SGLang 的 boundary 编号。

    BLOCK_OUTPUT: block b 的输出在 SGLang 里是 boundary b+1
                  （因为 1..N-1 定义为「第 L-1 个 block 输出」）
    BOUNDARY:     原样透传
    """
    semantics = LayerSemantics.parse(semantics) if not isinstance(semantics, LayerSemantics) \
        else semantics
    blocks = tuple(block_indices)
    for block in blocks:
        if type(block) is not int or block < 0:
            raise ValueError(f"block index must be a non-negative int, got {block!r}")
        if num_layers is not None and block >= num_layers:
            raise ValueError(f"block index {block} >= num_layers {num_layers}")

    if semantics is LayerSemantics.BLOCK_OUTPUT:
        capture = tuple(block + 1 for block in blocks)
    else:
        capture = blocks

    return LayerSpec(
        block_indices=blocks,
        capture_layers=capture,
        semantics=semantics,
        mapping_rule=_RULE[semantics],
        num_layers=num_layers,
    )
