"""多风险 Probe Bank 兼容：一次采集、按层并集、各探针独立阈值。"""
from types import SimpleNamespace

import pytest
import torch

from inscope_server.scoring import ScoringPipeline


def _probe(risks_layer, positions, threshold, probe_id):
    """假探针：层 = risks_layer，位置与阈值各自独立。"""
    class Spec:
        pass
    spec = Spec()
    spec.probe_id = probe_id
    spec.capture_layers = (risks_layer,)
    spec.selected_positions = positions
    probe = SimpleNamespace(threshold=threshold, capture_layers=(risks_layer,),
                            selected_positions=positions, spec=spec)
    probe.flatten = lambda blocks: torch.cat([b.reshape(1, -1) for b in blocks], dim=-1)
    probe.score = lambda x: (float(x[0, 0]), float(x[0, 0]))
    return probe


def test_capture_plan_unions_layers_and_positions():
    primary = _probe(9, (-4, -3, -2, -1), 0.9, "ipi")
    other = _probe(17, (-1,), 0.5, "harmful")
    pipeline = ScoringPipeline({"probe": primary}, SimpleNamespace(max_inflight=2), None)
    pipeline.request_probes = {"ipi": primary, "harmful": other}
    layers, positions = pipeline.capture_plan()
    # 不同 checkpoint 的层不同：一次采集取并集，而不是各发一次。
    assert layers == (9, 17)
    assert positions == (-4, -3, -2, -1)


def test_each_probe_scores_with_its_own_threshold():
    primary = _probe(9, (-1,), 0.9, "ipi")
    other = _probe(17, (-1,), 0.5, "harmful")
    pipeline = ScoringPipeline({"probe": primary}, SimpleNamespace(max_inflight=2), None)
    assert pipeline.score_features([torch.zeros(1, 1)]) == (0.0, 0.0)
    pipeline.request_probes = {"ipi": primary, "harmful": other}
    features = {9: torch.full((1, 1), 0.8), 17: torch.full((1, 1), 0.2)}
    scores = pipeline.score_features(features)
    assert scores["ipi"][0] == pytest.approx(0.8) and scores["harmful"][0] == pytest.approx(0.2)
    # 阈值各自独立，不能共用主探针那一个。
    assert primary.threshold == 0.9 and other.threshold == 0.5


def test_multi_probe_rows_carry_all_scores():
    import asyncio

    primary = _probe(9, (-1,), 0.9, "ipi")
    other = _probe(17, (-1,), 0.5, "harmful")
    pipeline = ScoringPipeline({"probe": primary}, SimpleNamespace(max_inflight=2, detect_batch=2), None)
    pipeline.request_probes = {"ipi": primary, "harmful": other}

    async def prepare(messages, tools=None):
        return [1, 2, 3], {"profile_id": "x"}, None

    async def capture_batch(ids, layers, positions):
        assert layers == (9, 17)
        return [{9: torch.full((1, 1), 0.7), 17: torch.full((1, 1), 0.6)} for _ in ids]

    pipeline.prepare = prepare
    from inscope_contract import DetectionRequest
    pipeline.state = {"probe": primary, "backend": SimpleNamespace(capture_batch=capture_batch)}
    request = DetectionRequest(messages=[{"role": "user", "content": "hi"}])
    rows = asyncio.run(pipeline.detect_batch([request]))
    assert rows[0]["scores"]["ipi"][0] == pytest.approx(0.7) and rows[0]["scores"]["harmful"][0] == pytest.approx(0.6)
    assert rows[0]["label"] == "benign"


def test_probe_registry_parses_risk_to_checkpoint_mapping():
    from inscope_server.config import parse_probe_spec

    spec = parse_probe_spec("ipi=/a/best.pt,harmful=/b/best.pt")
    assert list(spec) == ["ipi", "harmful"]
    assert spec["ipi"] == "/a/best.pt"
    for bad in ("", "ipi", "=x", "ipi=", "ipi=/a,ipi=/b", " bad key=/a"):
        with pytest.raises(ValueError):
            parse_probe_spec(bad)


def test_registry_rejects_duplicate_risk_across_sources():
    from inscope_server.config import Settings

    with pytest.raises(ValueError, match="duplicate risk"):
        Settings(model_path="/m", api_key="k", probe_checkpoint="/p",
                 composer_profile="a6_nemotron_risk_faced_v1", probes={"ipi": "/other"})


# ------------------------------------------------- Probe Bank 多风险（HTTP 层）
def _bank_app(score_for):
    """主探针 ipi + 另一风险 harmful，两层不同；用于验证一次采集、各自打分。"""
    import torch
    from types import SimpleNamespace
    import inscope_server.app as app_module
    import inscope_server.context_profile as profiles

    # 允许 per-request profile 绑定到合成探针 id；不影响真实注册表。
    original_bind = app_module.bind_profile
    app_module.bind_profile = lambda profile, probe_id: profile
    from inscope_server.context_profile import get_profile as real_profile
    class _Profile:
        profile_id = "a6_nemotron_risk_faced_v1"
        composer = real_profile("a6_nemotron_risk_faced_v1").composer
        position_mode = real_profile("a6_nemotron_risk_faced_v1").position_mode

        def describe(self):
            return real_profile("a6_nemotron_risk_faced_v1").describe()
    profiles_original = app_module.get_profile
    app_module.get_profile = lambda name: _Profile()
    original_resolve = getattr(app_module, "request_profile", None)
    _bank_app._restore = lambda: (setattr(app_module, "bind_profile", original_bind),
                                  setattr(app_module, "get_profile", profiles_original),
                                  None if original_resolve is None
                                  else setattr(app_module, "request_profile", original_resolve))
    from inscope_server.app import create_app

    class Spec:
        schema = "x"; probe_architecture = "linear"; layer_semantics = "block"
        block_indices = (8,); feature_normalization = "x"
        flatten_order = ("layer", "position", "hidden")
        feature_name = "f"; labeling_protocol = "p"; probe_prefix = ""

    def make(risk):
        spec = Spec()
        spec.probe_id = f"probe:{risk}"
        spec.selected_positions = (-4, -3, -2, -1)
        state = SimpleNamespace(threshold=0.9 if risk == "ipi" else 0.5, sha256=risk + "hash",
                                capture_layers=(9,) if risk == "ipi" else (17,),
                                selected_positions=(-4, -3, -2, -1), input_dim=4,
                                spec=spec)

        def flatten(hidden):
            return torch.cat([h.reshape(-1) for h in hidden])

        def score(x):
            return (score_for(risk), score_for(risk))
        state.flatten, state.score = flatten, score
        return state

    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            return "".join(str(m.get("content") or "") for m in messages)

        def encode(self, text, **kw):
            return list(range(len(text) % 17 + 8))

    calls = []

    class Backend:
        async def capture(self, ids, layers, positions):
            calls.append(("capture", tuple(layers)))
            return [torch.zeros((len(positions), 4)) for _ in layers]

        async def capture_batch(self, batch_ids, layers, positions):
            calls.append(("batch", tuple(layers)))
            return [[torch.zeros((len(positions), 4))] for _ in layers] * len(batch_ids)

        async def close(self):
            pass

    settings = SimpleNamespace(
        model_path="/data/djs/models/Qwen3-8B-FP8", api_key="k",
        probe_checkpoint="x", composer_profile="a6_nemotron_risk_faced_v1",
        backend_url="http://127.0.0.1:1", backend_api_key="",
        max_tokens=100000, max_inflight=4, timeout=60, analyze_timeout=600, analyze_batch=8,
        audit_timeout=1200, detect_batch=8, max_body_bytes=10 ** 7, max_batch_items=128,
        backend_max_inflight=4, probes={}, detector_model="shadow-model-x", primary_risk="ipi")
    primary = make("ipi")
    app = create_app(settings, tokenizer=Tokenizer(), probe=primary,
                     probes={"harmful": make("harmful")}, backend=Backend())
    return app, calls


def test_bank_payload_matches_coolwatch_contract():
    from fastapi.testclient import TestClient

    scores = {"ipi": 0.95, "harmful": 0.2}
    app, calls = _bank_app(scores.__getitem__)
    try:
        with TestClient(app) as client:
            response = client.post("/detect", headers={"authorization": "Bearer k"},
                                   json={"messages": [{"role": "user", "content": "hello"}]})
    finally:
        _bank_app._restore()
    assert response.status_code == 200, response.text
    body = response.json()
    # coolwatch ProbeBankGuard 的硬校验逐条对齐。
    assert set(body["per_risk"]) == set(body["enabled_risks"])
    assert body["enabled_risks"] == ["ipi", "harmful"]
    assert body["disabled_risks"] == ["prompt_leakage"]
    assert set(body["per_entry"]) == set(body["enabled_risks"])
    for row in body["per_risk"].values():
        assert type(row["flagged"]) is bool
        assert 0 <= row["score"] <= 1 and 0 <= row["threshold"] <= 1
    assert body["overall"] is True
    assert body["triggered_entries"] == ["ipi"]
    assert body["detector_model"] == "shadow-model-x"
    assert body["bank_version"].startswith("sha256:")
    assert body["input_tokens"] > 0
    assert isinstance(body["latency_ms"], dict)
    # 不同层只在一次采集里取并集。
    assert calls == [("capture", (9, 17))]


def test_bank_disabled_risks_cover_unregistered_risks():
    """未注册的风险只能出现在 disabled_risks，绝不编造分数。"""
    from fastapi.testclient import TestClient

    app, _calls = _bank_app(lambda risk: 0.1)
    try:
        with TestClient(app) as client:
            body = client.post("/detect", headers={"authorization": "Bearer k"},
                               json={"messages": [{"role": "user", "content": "hello"}]}).json()
    finally:
        _bank_app._restore()
    assert body["enabled_risks"] == ["ipi", "harmful"]
    assert body["disabled_risks"] == ["prompt_leakage"]
    assert set(body["per_risk"]) == set(body["enabled_risks"])
    assert body["overall"] is False


def test_settings_expose_bank_options_from_env(monkeypatch):
    from inscope_server.config import Settings

    monkeypatch.setenv("INSCOPE_MODEL_PATH", "/m")
    monkeypatch.setenv("INSCOPE_API_KEY", "k")
    monkeypatch.setenv("INSCOPE_PROBE_CHECKPOINT", "/p")
    monkeypatch.setenv("INSCOPE_COMPOSER_PROFILE", "no_reasoning")
    monkeypatch.setenv("INSCOPE_PROBES", "harmful=/h")
    monkeypatch.setenv("INSCOPE_DETECTOR_MODEL", "shadow-agent")
    settings = Settings.from_env()
    assert settings.probes == {"harmful": "/h"}
    assert settings.detector_model == "shadow-agent"
    assert settings.primary_risk == "ipi"


def test_settings_reject_probe_overlapping_primary_risk(monkeypatch):
    from inscope_server.config import Settings

    for key, value in {"INSCOPE_MODEL_PATH": "/m", "INSCOPE_API_KEY": "k",
                       "INSCOPE_PROBE_CHECKPOINT": "/p",
                       "INSCOPE_COMPOSER_PROFILE": "no_reasoning",
                       "INSCOPE_PROBES": "ipi=/other"}.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="duplicate risk"):
        Settings.from_env()


def test_request_can_narrow_enabled_risks():
    """服务加载多个风险，调用方只要一个：返回子集，不报错，也不伪造。"""
    from fastapi.testclient import TestClient

    app, _calls = _bank_app(lambda risk: 0.95 if risk == "ipi" else 0.2)
    try:
        with TestClient(app) as client:
            body = client.post("/detect", headers={"authorization": "Bearer k"},
                               json={"messages": [{"role": "user", "content": "hello"}],
                                     "enabled_risks": ["ipi"]}).json()
    finally:
        _bank_app._restore()
    assert body["enabled_risks"] == ["ipi"]
    assert set(body["per_risk"]) == {"ipi"}
    assert set(body["per_entry"]) == {"ipi"}
    assert body["triggered_entries"] == ["ipi"]
    assert set(body["disabled_risks"]) == {"harmful", "prompt_leakage"}
    assert body["overall"] is True


def test_requesting_unloaded_risk_is_422():
    """调用方不能要求服务没加载的风险，否则只能伪造或静默缺失。"""
    from fastapi.testclient import TestClient

    app, _calls = _bank_app(lambda risk: 0.1)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/detect", headers={"authorization": "Bearer k"},
                json={"messages": [{"role": "user", "content": "hello"}],
                      "enabled_risks": ["prompt_leakage"]})
    finally:
        _bank_app._restore()
    assert response.status_code == 422
    assert "prompt_leakage" in response.json()["detail"]


def test_enabled_risks_shape_is_validated():
    from inscope_server.normalization import split_detect_payload

    base = {"messages": [{"role": "user", "content": "hi"}]}
    requests, _tools, _batch = split_detect_payload({**base, "enabled_risks": ["ipi"]})
    assert requests[0]["enabled_risks"] == ["ipi"]
    for bad in ([], ["ipi", "ipi"], [""], "ipi", [1]):
        with pytest.raises(ValueError):
            split_detect_payload({**base, "enabled_risks": bad})


def test_info_reports_every_loaded_risk():
    """/info 必须自述模型和多风险清单：调用方不配置模型名，只能靠这里识别。"""
    from fastapi.testclient import TestClient

    app, _calls = _bank_app(lambda risk: 0.1)
    try:
        with TestClient(app) as client:
            info = client.get("/info", headers={"authorization": "Bearer k"}).json()
            detect = client.post("/detect", headers={"authorization": "Bearer k"},
                                 json={"messages": [{"role": "user", "content": "hi"}]}).json()
    finally:
        _bank_app._restore()
    assert info["model"] and info["detector_model"] == info["model"]
    # 两个接口必须自述同一个模型，否则调用方展示与判定会对不上。
    assert detect["detector_model"] == info["detector_model"]
    assert set(info["enabled_risks"]) == {"ipi", "harmful"}
    assert info["disabled_risks"] == ["prompt_leakage"]
    by_risk = {row["risk"]: row for row in info["risks"]}
    assert set(by_risk) == {"ipi", "harmful"}
    assert by_risk["ipi"]["capture_layers"] == [9]
    assert by_risk["harmful"]["capture_layers"] == [17]
