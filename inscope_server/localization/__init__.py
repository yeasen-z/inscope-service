"""Observation 级区域定位：算法本体，从 probe_training.debug 原样引入。

来源：`probe_training/debug/{segments,localization,attribution,api}.py`
（见 djs 的 Probe Debug API 交接，2026-10-24）。

为什么复制而不是依赖 ipi-doom：这四个模块只依赖标准库，复制过来可以
让 inscope-service 独立部署；算法本身保持逐行一致，便于用同一份 case
与训练侧结果对齐（验收基准：status=localized，start=2636、end=3382）。

改动纪律：不在此目录内改算法。需要修行为时回到上游改，再整体同步过来，
否则两边结果会静默分叉。
"""
