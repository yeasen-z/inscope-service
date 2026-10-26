

# ------------------------------------------------------- /detect 批量
def test_split_detect_payload_single_vs_batch():
    from inscope_server.normalization import split_detect_payload

    single = {"messages": [{"role": "user", "content": "a"}]}
    requests, tools, batch = split_detect_payload(single)
    assert batch is False and len(requests) == 1 and tools is None

    many = {"messages": [[{"role": "user", "content": "a"}],
                         [{"role": "user", "content": "b"}]]}
    requests, _, batch = split_detect_payload(many)
    assert batch is True and len(requests) == 2


def test_split_detect_payload_rejects_ragged_and_unknown_fields():
    import pytest
    from inscope_server.normalization import split_detect_payload

    ragged = {"messages": [{"role": "user", "content": "a"},
                           [{"role": "user", "content": "b"}]]}
    with pytest.raises(ValueError):
        split_detect_payload(ragged)
    with pytest.raises(ValueError):
        split_detect_payload({"messages": [], "extra": 1})


def test_http_detect_batch_isolates_failures():
    """一条失败不能拖垮整批，也不能被伪装成 benign。

    用真实的 score_prompts 路径：让 prepare（编排+分词）对某一条抛错，
    验证错误被隔离到那一条，其余两条照常出分。
    """
    from fastapi.testclient import TestClient
    from inscope_server.app import create_app
    from inscope_server import scoring as scoring_module
    from test_analyze import _http_app

    original = scoring_module.ScoringPipeline.prepare

    async def prepare(self, messages, tools=None):
        text = "".join(str(m.get("content") or "") for m in messages)
        if text == "boom":
            raise ValueError("cannot tokenize")
        return [1, 2, 3, 4, 5], {"profile_id": "p"}

    scoring_module.ScoringPipeline.prepare = prepare
    try:
        app = _http_app(lambda: (0.0, 0.5))
        # 让探针按内容区分好坏：正分=attack
        with TestClient(app) as client:
            response = client.post(
                "/detect", headers={"authorization": "Bearer k"},
                json={"messages": [
                    [{"role": "user", "content": "ok"}],
                    [{"role": "user", "content": "fine"}],
                    [{"role": "user", "content": "boom"}],
                ]})
    finally:
        scoring_module.ScoringPipeline.prepare = original
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema"] == "inscope.detect_batch.v1"
    assert body["count"] == 3
    rows = body["results"]
    assert "error" in rows[2], rows[2]
    assert "label" not in rows[2]
    assert rows[0].get("label") == "benign" and rows[1].get("label") == "benign"


def test_http_detect_single_returns_input_tokens():
    """单条路径必须带 input_tokens：SDK 依赖该字段，缺了就抛 InScopeError。"""
    from fastapi.testclient import TestClient
    from test_analyze import _http_app

    app = _http_app(lambda: (1.0, 0.6))
    with TestClient(app) as client:
        response = client.post(
            "/detect", headers={"authorization": "Bearer k"},
            json={"messages": [{"role": "user", "content": "hello there"}]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "input_tokens" in body and body["input_tokens"] > 0
    assert "errors" not in body


def test_http_detect_batch_rows_carry_input_tokens():
    """批量路径的每一行也要带 input_tokens，与单条保持同一组字段。"""
    from fastapi.testclient import TestClient
    from test_analyze import _http_app

    app = _http_app(lambda: (1.0, 0.6))
    with TestClient(app) as client:
        response = client.post(
            "/detect", headers={"authorization": "Bearer k"},
            json={"messages": [[{"role": "user", "content": "a"}],
                               [{"role": "user", "content": "b"}]]})
    assert response.status_code == 200, response.text
    for row in response.json()["results"]:
        assert row.get("input_tokens", 0) > 0, row
