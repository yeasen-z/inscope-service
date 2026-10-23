"""路径解析。checkpoint 与模型都不随包发布，只记录约定位置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = str(ROOT / "checkpoints/blend")
