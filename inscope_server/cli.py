"""inscope-serve：用环境变量配置启动 API。参数不重复造，全走 Settings.from_env。"""
import argparse
import logging
import os

import uvicorn

from inscope_contract import DEFAULT_API_HOST, DEFAULT_API_PORT

from .config import Settings

log = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the InScope detection API")
    parser.add_argument("--host", default=os.getenv("INSCOPE_HOST", DEFAULT_API_HOST))
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("INSCOPE_PORT", str(DEFAULT_API_PORT))))
    parser.add_argument("--log-level", default=os.getenv("INSCOPE_LOG_LEVEL", "info"))
    args = parser.parse_args(argv)

    # 提前校验配置，让缺项在启动前报出来，而不是首个请求时才炸。
    settings = Settings.from_env()
    log.info("engine=%s model=%s seed=%d", settings.backend_url,
             settings.model_path, settings.detector_seed)

    from .app import create_app
    uvicorn.run(create_app(settings), host=args.host, port=args.port,
                log_level=args.log_level)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
