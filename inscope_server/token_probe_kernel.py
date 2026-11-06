"""Inference codec with the frozen FP16-scale/FP8/BF16 rounding boundaries."""
import triton
import triton.language as tl


@triton.jit
def codec_normalize(X, Mean, Std, Slots, Y, Valid, D: tl.constexpr, M: tl.constexpr,
                    BLOCK: tl.constexpr, NORMALIZE: tl.constexpr = True):
    row = tl.program_id(0).to(tl.int64)
    model = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    slot = tl.load(Slots + model)
    x = tl.load(X + (row * M + slot) * D + d, d < D, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) / 448., 2.**-24).to(tl.float16).to(tl.float32)
    valid = tl.sum(((x == x) & (tl.abs(x) != float('inf'))).to(tl.int32), 0) == BLOCK
    valid = valid & (scale < float('inf'))
    tl.store(Valid + row * tl.num_programs(1) + model, valid)
    q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.), 448.).to(tl.float8e4nv).to(tl.float32)
    decoded = (q * scale).to(tl.bfloat16).to(tl.float32)
    if NORMALIZE:
        mean = tl.load(Mean + model * D + d, d < D, other=0)
        std = tl.load(Std + model * D + d, d < D, other=1)
        y = tl.div_rn(decoded - mean, std).to(tl.bfloat16)
    else:
        y = decoded.to(tl.bfloat16)
    tl.store(Y + (model * tl.num_programs(0) + row) * D + d, y, d < D)
