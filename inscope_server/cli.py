"""inscope-serve：用环境变量配置启动 API。参数不重复造，全走 Settings.from_env。"""
import argparse
import logging
import os
import sys

import uvicorn

from inscope_contract import DEFAULT_API_HOST, DEFAULT_API_PORT

from .config import Settings

log = logging.getLogger(__name__)


def _preflight(settings):
    """启动前把所有会致命的错误一次性报出来，而不是等首个请求。"""
    from .context_profile import bind_profile, get_profile
    from .probe import load_probe

    profile = get_profile(settings.composer_profile)
    probe = load_probe(settings.probe_checkpoint)
    bind_profile(profile, probe.spec.probe_id)

    wanted = profile.composer.selected_positions if profile.composer else None
    if wanted is not None and tuple(wanted) != tuple(probe.selected_positions):
        raise ValueError(
            "profile %r 期望位置 %s，但探针声明 %s"
            % (profile.profile_id, list(wanted), list(probe.selected_positions)))
    return profile, probe


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the InScope detection API")
    parser.add_argument("--host", default=os.getenv("INSCOPE_HOST", DEFAULT_API_HOST))
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("INSCOPE_PORT", str(DEFAULT_API_PORT))))
    parser.add_argument("--log-level", default=os.getenv("INSCOPE_LOG_LEVEL", "info"))
    parser.add_argument("--preflight-only", action="store_true",
                        help="只做配置与 checkpoint 校验后退出，不启动服务")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper())

    # 提前校验配置，让缺项在启动前报出来，而不是首个请求时才炸。
    try:
        settings = Settings.from_env()
        profile, probe = _preflight(settings)
    except (ValueError, OSError) as exc:
        log.error("启动自检失败: %s", exc)
        return 2

    log.info("engine=%s model=%s", settings.backend_url, settings.model_path)
    log.info("probe=%s id=%s", settings.probe_checkpoint, probe.spec.probe_id)
    log.info("layers=%s positions=%s dim=%d threshold=%.4f",
             list(probe.capture_layers), list(probe.selected_positions),
             probe.input_dim, probe.threshold)
    log.info("composer_profile=%s", profile.profile_id)

    if args.preflight_only:
        log.info("自检通过，未启动服务")
        return 0

    from .app import create_app
    # timeout_keep_alive 默认只有 5 秒：空闲一会儿就断，下次调用要重新握手。
    # 实测重建连接约多花 50ms，比检测本身还显眼，所以放宽到 75 秒
    # （高于常见负载均衡的 60 秒空闲超时，避免连接被两头同时掐断）。
    uvicorn.run(create_app(settings, probe=probe), host=args.host, port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=int(os.getenv("INSCOPE_KEEPALIVE", "75")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
