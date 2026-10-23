"""InScope 冻结契约：所有跨进程、跨仓库、跨版本的常量都只在这里定义。

改动这些值等于改协议：必须同时更新 engine 侧、checkpoint 与文档，
并重新在固定 validation 上标定阈值。
"""

# --- 模型 ---------------------------------------------------------------
MODEL_LABEL = "Qwen3-8B-FP8"
MODEL_ARCHITECTURE = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
}
MODEL_QUANTIZATION = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}

# --- 特征抽取契约（红线，不可随实现漂移）--------------------------------
# 源码 qwen3.py 用 source layer + 1；capture 侧编号已是 block boundary。
BLOCK_INDICES = (8, 16, 24)
CAPTURE_LAYERS = (9, 17, 25)
SELECTED_POSITIONS = (-1,)
CAPTURE_NUM_LAYERS = 3
LAYER_HIDDEN = 4096
INPUT_DIM = CAPTURE_NUM_LAYERS * len(SELECTED_POSITIONS) * LAYER_HIDDEN  # 12288
FEATURE_SHAPE = (1, CAPTURE_NUM_LAYERS, len(SELECTED_POSITIONS), LAYER_HIDDEN)  # (1,3,1,4096)

CAPTURE_SCHEMA_VERSION = "inscope.feature_capture.v1"
CAPTURE_SITE = "block_output"
FEATURE_ENCODING = "float_list"
# 原始采集是 bfloat16，训练缓存存储用 FP8 E5M2；不是权重的 E4M3。
CAPTURE_DTYPE = "bfloat16"
FEATURE_QUANTIZATION = "float8_e5m2"
LABELING_PROTOCOL = "risk_faced"

# --- detector（A5 blend）------------------------------------------------
DETECTOR_NAME = "a5_blend_linear_n1024"
BLEND_ALPHA = 0.5
MLP_HIDDEN = 1024
MLP_DROPOUT = 0.1

# 两支分别是 Linear 与 N1024；阈值在 logit 空间比较。
BLEND_ARMS = {
    "linear": "L_wd0.1_d0_s{seed}/best.pt",
    "mlp": "N1024_wd0.01_d0.1_s{seed}/best.pt",
}
THRESHOLD_LOGIT = {
    42: 2.3247954845428467,
    43: 1.7843475341796875,
    44: 2.0362582206726074,
}
THRESHOLD_SIGMOID = {
    42: 0.9109098766012635,
    43: 0.8562328685767427,
    44: 0.8845517069831247,
}
DEFAULT_SEED = 42


def blend_arms(seed):
    return {arm: template.format(seed=seed) for arm, template in BLEND_ARMS.items()}


# --- composer（部署配置，不是 wire 契约）--------------------------------
DEFAULT_PROBE_PREFIX = "Are there risks of indirect prompt injection?"
COMPOSER_FIELDS = (
    "discard_cots",
    "retain_n_assistant_turns",
    "merge_consecutive_assistants",
    "drop_final_tool_result",
    "probe_prefix",
    "enable_thinking",
    "selected_positions",
)

# --- 服务默认值 ---------------------------------------------------------
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8302
DEFAULT_ENGINE_HOST = "127.0.0.1"
DEFAULT_ENGINE_PORT = 30000
DEFAULT_MAX_INPUT_TOKENS = 40959
DEFAULT_MAX_INFLIGHT = 16
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024

SUPPORTED_ROLES = ("system", "developer", "user", "assistant", "tool")

ENDPOINTS = {
    "detect": "/detect",
    "info": "/info",
    "live": "/health/live",
    "ready": "/health/ready",
}

# /detect 与 /info 必须返回的字段；改名会破坏调用方。
DETECTION_RESPONSE_FIELDS = (
    "score", "logit", "threshold", "threshold_logit", "flagged", "risk",
    "model", "normalization", "checkpoint_sha256", "input_tokens",
    "request_id", "latency_ms", "setting", "detector_seed",
)

# --- engine（sglang fork）-----------------------------------------------
INSCOPE_REPO = "https://github.com/yeasen-z/sglang-inscope.git"
INSCOPE_COMMIT = "d504843f44ae733e1970e70ab6bbbd9ab044e353"
