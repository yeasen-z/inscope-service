"""本地预量化 FP8 权重校验：不允许静默回退到 BF16。"""
import json
from pathlib import Path

from .constants import MODEL_ARCHITECTURE, MODEL_QUANTIZATION


def validate_model_weights(path):
    root = Path(path)
    config = json.loads((root / "config.json").read_text())
    if any(config.get(key) != value for key, value in MODEL_ARCHITECTURE.items()):
        raise ValueError("Expected Qwen3-8B architecture")
    quantization = config.get("quantization_config") or {}
    if any(quantization.get(key) != value for key, value in MODEL_QUANTIZATION.items()):
        raise ValueError("Expected Qwen3-8B-FP8 blockwise FP8 quantization config")
    if not list(root.glob("*.safetensors")):
        raise ValueError("No model safetensors found in model path")
    return config
