"""客户端与服务的字段契约必须一致；profile 与探针必须互相认领。"""

import pytest
import torch

from inscope.client import DetectResult, InScopeError
from inscope_server.context_profile import bind_profile, get_profile
from inscope_server.probe import load_probe

from conftest import DIM, LAYERS, POSITIONS, write_checkpoint


# ---------------------------------------------------------------- 客户端契约
def _payload(**over):
    base = {"logit": 1.25, "probability": 0.78, "threshold": 0.9, "label": "benign",
            "risk": "indirect_prompt_injection", "input_tokens": 42,
            "request_id": "rid-1", "latency_ms": 12.5,
            "composer_profile": "a6_nemotron_risk_faced_v1"}
    base.update(over)
    return base


def test_client_parses_current_service_payload():
    result = DetectResult.from_payload(_payload())
    assert result.logit == 1.25
    assert result.probability == 0.78
    assert result.threshold == 0.9
    assert result.label == "benign"
    assert result.flagged is False


def test_client_flagged_tracks_label():
    assert DetectResult.from_payload(_payload(label="attack", probability=0.99)).flagged is True


def test_client_rejects_payload_missing_required_fields():
    with pytest.raises(InScopeError, match="Malformed"):
        DetectResult.from_payload({"score": 1.0, "flagged": True})


def test_client_tolerates_absent_optional_fields():
    payload = _payload()
    for key in ("latency_ms", "composer_profile", "risk", "request_id"):
        payload.pop(key)
    result = DetectResult.from_payload(payload)
    assert result.latency_ms == 0.0
    assert result.composer_profile == ""


# ------------------------------------------------------- profile / probe 绑定
def test_profile_matches_real_checkpoint_id(tmp_path):
    """a6 profile 的 probe_id 必须等于该 checkpoint 派生出的 id。"""
    probe = load_probe(write_checkpoint(tmp_path))
    profile = get_profile("a6_nemotron_risk_faced_v1")
    # 不抛异常即表示绑定成立。
    bind_profile(profile, probe.spec.probe_id)


def test_profile_positions_match_checkpoint(tmp_path):
    probe = load_probe(write_checkpoint(tmp_path))
    profile = get_profile("a6_nemotron_risk_faced_v1")
    assert tuple(profile.composer.selected_positions) == tuple(probe.selected_positions)
    # 位置数必须与 input_dim 自洽。
    assert probe.input_dim == len(probe.capture_layers) * len(probe.selected_positions) * 4096


def test_a6_profile_has_no_probe_prefix():
    """该探针训练时无前缀；加上会改变特征分布（实测分数反转）。"""
    profile = get_profile("a6_nemotron_risk_faced_v1")
    assert profile.composer.probe_prefix == ""


def test_mismatched_profile_is_rejected():
    profile = get_profile("a5_simplified_v1")
    with pytest.raises(ValueError, match="bound to probe"):
        bind_profile(profile, "linear:other:dataset")


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown composer profile"):
        get_profile("does_not_exist")


def test_probe_id_is_stable_across_loads(tmp_path):
    path = write_checkpoint(tmp_path)
    assert load_probe(path).spec.probe_id == load_probe(path).spec.probe_id


def test_probe_id_differs_for_other_feature_name(tmp_path):
    a = load_probe(write_checkpoint(tmp_path, sub="a", feature_name="feat_a"))
    b = load_probe(write_checkpoint(tmp_path, sub="b", feature_name="feat_b"))
    assert a.spec.probe_id != b.spec.probe_id


def test_client_reuses_one_connection_across_calls():
    """同步 Client 必须复用连接：每次重建约多花 50ms，高频调用下不可接受。"""
    import httpx
    from inscope import client as client_module

    built = []

    class FakeSession:
        def __init__(self, **kw):
            built.append(kw)
            self.closed = False
            self.calls = 0

        def get(self, path, **kw):
            self.calls += 1
            return httpx.Response(200, json={"status": "ready"},
                                  request=httpx.Request("GET", "http://x" + path))

        def post(self, path, **kw):
            self.calls += 1
            return httpx.Response(200, json={
                "logit": 0.0, "probability": 0.5, "threshold": 0.9, "label": "benign",
                "risk": "r", "model": "m", "input_tokens": 1, "request_id": "id",
                "latency_ms": 1.0}, request=httpx.Request("POST", "http://x" + path))

        def close(self):
            self.closed = True

    original = client_module.httpx.Client
    client_module.httpx.Client = FakeSession
    try:
        client = client_module.Client("http://127.0.0.1:8302", api_key="k")
        client.ready()
        client.ready()
        client.detect([{"role": "user", "content": "x"}])
        # 三次调用只应创建一个会话
        assert len(built) == 1, built
        assert built[0]["headers"] == {"Authorization": "Bearer k"}
        session = client._session
        client.close()
        assert session.closed is True
        assert client._session is None
        # 关闭后再次使用会重新创建，不会复用已关闭的会话
        client.ready()
        assert len(built) == 2
        client.close()
    finally:
        client_module.httpx.Client = original


def test_client_usable_as_context_manager():
    import httpx
    from inscope import client as client_module

    class FakeSession:
        def __init__(self, **kw):
            self.closed = False

        def get(self, path, **kw):
            return httpx.Response(200, json={}, request=httpx.Request("GET", "http://x"))

        def close(self):
            self.closed = True

    original = client_module.httpx.Client
    client_module.httpx.Client = FakeSession
    try:
        with client_module.Client("http://127.0.0.1:8302") as client:
            client.ready()
            session = client._session
        assert session.closed is True
    finally:
        client_module.httpx.Client = original
