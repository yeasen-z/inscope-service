"""Context Composer：按探针配置(profile)组织真正送入特征提取的上下文。

设计要点
--------
* Composer 的**行为**不再写死在 server 里，而是由 profile 决定。
* 每个 profile 必须显式声明它服务哪个探针、由哪次实验标定；未声明即拒绝启动。
* profile 与 probe 绑定成 (ProbeSpec, ComposerProfile)，两者版本一起进入响应，
  避免「用 A profile 产出的特征喂 B 探针」这种静默错配。

与旧版 ComposerConfig 的关系
----------------------------
旧版仍作为 `a5_simplified_v1` 一种 profile 的实现保留（逻辑逐字节不变），
只是不再有「不传配置就默认走这套规则」的路径。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .composer import ComposerConfig, ComposedPrompt, compose_prompt


# ---------------------------------------------------------------- profile
@dataclass(frozen=True)
class ComposerProfile:
    """一个可复现的上下文编排规格。

    profile_id / version 会进入响应，便于事后判断「这批特征是用什么规则产的」。
    """

    profile_id: str
    version: str
    # 服务哪个探针；'*' 表示通用（仅调试用）
    probe_id: str
    description: str = ""
    # 旧的简化 composer 参数；None 表示该 profile 不走这套规则
    composer: ComposerConfig | None = None
    # 是否在 mask 前给 prompt 加固定探测问题（来自探针训练时的 gauge）
    probe_prefix: str = ""
    # 位置语义：'last' = 完整输入最后一个 token
    position_mode: str = "last"
    # 额外的不变量，用于启动期自检
    invariants: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id or not self.version:
            raise ValueError("profile needs profile_id and version")
        if not self.probe_id:
            raise ValueError("profile must declare probe_id (no implicit default)")
        if self.position_mode not in ("last",):
            raise ValueError(f"unsupported position_mode {self.position_mode!r}")

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "profile_id": self.profile_id,
            "profile_version": self.version,
            "probe_id": self.probe_id,
            "position_mode": self.position_mode,
        }
        if self.composer is not None:
            payload["composer"] = self.composer.describe()
        return payload

    def with_composer(self, composer: ComposerConfig) -> "ComposerProfile":
        return replace(self, composer=composer)


# ------------------------------------------------------- profile 注册表
def _a5_simplified_v1(prefix: str) -> ComposerProfile:
    return ComposerProfile(
        profile_id="a5_simplified_v1",
        version="1",
        probe_id="a5_blend",
        description=(
            "simplified/context_composer 移植：选窗 + 去 CoT + 模板 + 尾部探测问题。"
            "仅用于 A5 blend 探针。"
        ),
        composer=ComposerConfig(probe_prefix=prefix),
        probe_prefix=prefix,
        position_mode="last",
        invariants={"discard_cots": True, "retain_n_assistant_turns": 1},
    )


PROFILES = {
    "a5_simplified_v1": _a5_simplified_v1,
}


def get_profile(profile_id: str, *, probe_prefix: str | None = None) -> ComposerProfile:
    """按 id 取 profile；未知 id 直接报错，不回落到任何默认策略。"""
    factory = PROFILES.get(profile_id)
    if factory is None:
        raise ValueError(
            f"unknown composer profile {profile_id!r}; available: {sorted(PROFILES)}"
        )
    if profile_id == "a5_simplified_v1":
        return factory(probe_prefix if probe_prefix is not None
                       else "Are there risks of indirect prompt injection?")
    return factory()


def bind_profile(profile: ComposerProfile, probe_id: str) -> ComposerProfile:
    """profile 与探针必须匹配，否则不做特征提取。"""
    if profile.probe_id == "*":
        return profile
    if profile.probe_id != probe_id:
        raise ValueError(
            f"composer profile {profile.profile_id!r} is bound to probe "
            f"{profile.probe_id!r}, but the loaded probe is {probe_id!r}"
        )
    return profile


# --------------------------------------------------------------- 执行
@dataclass(frozen=True)
class ComposedContext:
    """Composer 的完整输出。input_ids 是唯一真相，不得由后端重新编码。"""

    input_ids: tuple[int, ...]
    text: str
    profile: dict[str, Any]
    token_count: int

    def capture_positions(self) -> tuple[int, ...]:
        return (len(self.input_ids) - 1,)


def compose(tokenizer, messages, tools, profile: ComposerProfile) -> ComposedContext:
    """按 profile 生成送进 engine 的上下文。"""
    if profile.composer is None:
        raise NotImplementedError(
            f"profile {profile.profile_id!r} 尚未实现 compose 实现；拒绝用默认规则代替"
        )
    composed: ComposedPrompt = compose_prompt(tokenizer, messages, tools, profile.composer)
    if not composed.token_ids:
        raise ValueError("composer produced an empty prompt")
    return ComposedContext(
        input_ids=tuple(composed.token_ids),
        text=composed.text,
        profile=profile.describe(),
        token_count=len(composed.token_ids),
    )
