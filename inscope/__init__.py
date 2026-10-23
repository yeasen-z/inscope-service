"""InScope: interpretable indirect-prompt-injection detection.

对外只暴露三个入口，内部包结构可变：
    inscope.detect(...)  -- 客户端，调用 HTTP 服务
    inscope.app          -- 服务端 ASGI 应用工厂
    inscope_contract     -- 契约常量与 schema
"""
from .client import Client, DetectResult, InScopeError

__all__ = ["Client", "DetectResult", "InScopeError", "__version__"]
__version__ = "0.4.0"
