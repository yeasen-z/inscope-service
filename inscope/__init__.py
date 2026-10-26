"""InScope: interpretable indirect-prompt-injection detection.

对外只暴露三个入口，内部包结构可变：
    inscope.detect(...)  -- 客户端，调用 HTTP 服务
    inscope.app          -- 服务端 ASGI 应用工厂
    inscope_contract     -- 契约常量与 schema
"""
from .client import AsyncClient, Client, DetectBatchResult, DetectItemError, DetectResult, InScopeError
from inscope_contract.schemas import DetectionBatchResult, DetectionItemError, DetectionResult

__all__ = ["Client", "AsyncClient", "DetectResult", "DetectBatchResult", "DetectItemError",
           "DetectionResult", "DetectionBatchResult", "DetectionItemError", "InScopeError", "__version__"]
__version__ = "0.6.0"
