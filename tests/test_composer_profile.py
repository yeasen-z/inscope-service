"""Composer profile 测试：profile 必须显式、绑定探针、未知即拒。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE = Path("/data/zmosg/inscope-service")
if str(SERVICE) not in sys.path:
    sys.path.insert(0, str(SERVICE))

from inscope_server.context_profile import (  # noqa: E402
    PROFILES,
    bind_profile,
    get_profile,
)


def test_known_profile_loads():
    profile = get_profile("a5_simplified_v1")
    assert profile.profile_id == "a5_simplified_v1"
    assert profile.probe_id == "a5_blend"
    assert profile.version


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown composer profile"):
        get_profile("does_not_exist")


def test_profile_declares_probe_binding():
    for name in PROFILES:
        profile = get_profile(name)
        assert profile.probe_id, f"{name} must declare probe_id"


def test_bind_rejects_mismatch():
    profile = get_profile("a5_simplified_v1")
    with pytest.raises(ValueError, match="bound to probe"):
        bind_profile(profile, "indirect_prompt_injection")


def test_bind_accepts_match():
    profile = get_profile("a5_simplified_v1")
    assert bind_profile(profile, "a5_blend") is profile


def test_describe_carries_identity():
    described = get_profile("a5_simplified_v1").describe()
    assert described["profile_id"] == "a5_simplified_v1"
    assert described["probe_id"] == "a5_blend"
    assert "composer" in described


def test_unimplemented_profile_raises_not_falls_back():
    """没有 compose 实现的 profile 必须抛错，不能回落到别的规则。"""
    from inscope_server.context_profile import ComposerProfile, compose

    bare = ComposerProfile(profile_id="x", version="1", probe_id="y")
    with pytest.raises(NotImplementedError):
        compose(None, [{"role": "user", "content": "hi"}], None, bare)
