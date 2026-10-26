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


def test_no_reasoning_profile_is_explicit_a6_ablation():
    base = get_profile('a6_nemotron_risk_faced_v1')
    profile = get_profile('no_reasoning')
    assert bind_profile(profile, base.probe_id) is profile
    assert profile.composer.discard_cots is True
    assert profile.composer.enable_thinking is False
    assert profile.composer.retain_n_assistant_turns == -1
    assert profile.composer.selected_positions == (-4, -3, -2, -1)
    assert profile.composer.probe_prefix == ''
    assert profile.describe()['calibration_status'] == 'experimental_not_recalibrated'
    assert base.composer.discard_cots is False
    assert base.composer.enable_thinking is True


@pytest.mark.parametrize('content', [
    '<think>secret</think>visible', '<think>secret',
    [{'type': 'thinking', 'content': 'secret'}, {'type': 'text', 'text': 'visible'}],
])
def test_no_reasoning_strips_assistant_only_preserves_input_and_calls(content):
    import copy
    from inscope_server.composer import normalize_prompt_messages
    profile = get_profile('no_reasoning')
    messages = [
        {'role': 'user', 'content': '<think>user text</think>'},
        {'role': 'assistant', 'content': content, 'reasoning': 'secret',
         'reasoning_content': 'secret', 'tool_calls': [
             {'id': 'call1', 'function': {'name': 'lookup', 'arguments': {'q': 'keep'}}}]},
        {'role': 'tool', 'content': '<think>tool text</think>', 'tool_call_id': 'call1'},
    ]
    original = copy.deepcopy(messages)
    rows = normalize_prompt_messages(messages, config=profile.composer)
    assert messages == original
    assert 'reasoning' not in rows[1] and 'reasoning_content' not in rows[1]
    assert 'secret' not in rows[1]['content']
    assert rows[1]['tool_calls'][0]['function']['name'] == 'lookup'
    assert rows[0]['content'] == '<think>user text</think>'
    assert rows[2]['content'] == '<think>tool text</think>'


def test_no_reasoning_template_flags_and_full_history():
    from inscope_server.context_profile import compose
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return 'rendered'
        def encode(self, text, **kwargs): return [1, 2, 3, 4]
    tokenizer = Tokenizer()
    messages = [{'role': 'user', 'content': 'opening'},
                {'role': 'assistant', 'content': '<think>secret</think>visible'},
                {'role': 'user', 'content': 'observation'}]
    context = compose(tokenizer, messages, [], get_profile('no_reasoning'))
    assert len(tokenizer.messages) == 3
    assert tokenizer.messages[1]['content'] == 'visible'
    assert tokenizer.kwargs['enable_thinking'] is False
    assert tokenizer.kwargs['add_generation_prompt'] is True
    assert context.input_ids == (1, 2, 3, 4)
    assert context.profile['profile_id'] == 'no_reasoning'
