"""HTTP contracts, error isolation and resource limit regressions."""
import asyncio
from types import SimpleNamespace
import httpx
import pytest
from fastapi.testclient import TestClient
from inscope_contract import DetectionBatchResult, DetectionResult
from inscope_server.backend import BackendError, InScopeBackend
from inscope_server.config import Settings
from test_analyze import _http_app, _replay

HEADERS = {"authorization": "Bearer k"}

@pytest.mark.parametrize('path,payload', [
    ('/detect', {'messages': [{'role': 'user', 'content': 'hello'}]}),
    ('/analyze', {'messages': _replay()['messages'], 'observation_index': 2}),
])
def test_backend_failures_are_503(path, payload):
    app = _http_app(lambda: (0.0, 0.5), allow_single_capture=True)
    async def fail(*args, **kwargs):
        raise BackendError('offline')
    app.state.inscope['backend'].capture = fail
    app.state.inscope['backend'].capture_batch = fail
    with TestClient(app) as client:
        assert client.post(path, headers=HEADERS, json=payload).status_code == 503


def test_batch_validation_isolated_and_public_contract_matches():
    with TestClient(_http_app(lambda: (0.0, 0.5), allow_single_capture=True)) as client:
        r = client.post('/detect', headers=HEADERS, json={'messages': [
            [{'role': 'human', 'text': 'hello'}], [{'role': 'invalid'}],
            [{'role': 'user', 'content': 'world'}]]})
        assert r.status_code == 200, r.text
        result = DetectionBatchResult.model_validate(r.json())
        assert result.count == 3 and result.results[1].status_code == 422
        assert isinstance(result.results[0], DetectionResult)
        assert isinstance(result.results[2], DetectionResult)
        assert result.results[0].normalization['applied']
        r = client.post('/detect', headers=HEADERS, json={'messages': [
            [{'role': 'user', 'content': 'hello'}]] * 129})
        assert r.status_code == 413


def test_openapi_shared_request_and_response_refs_resolve():
    with TestClient(_http_app(lambda: (0.0, 0.5), allow_single_capture=True)) as client:
        doc = client.get('/openapi.json').json()
    for route in ('/detect', '/analyze', '/audit'):
        assert doc['paths'][route]['post']['requestBody']['required']
    schemas = doc['components']['schemas']
    for name in ('DetectionRequest', 'DetectionBatchRequest', 'DetectionResult', 'DetectionBatchResult', 'DetectionItemError'):
        assert name in schemas
    def walk(value):
        if isinstance(value, dict):
            if '$ref' in value:
                assert value['$ref'].startswith('#/components/schemas/')
                assert value['$ref'].split('/')[-1] in schemas
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(doc)


def test_pipeline_does_not_fan_out_all_buckets():
    async def run():
        from inscope_server.scoring import ScoringPipeline
        from inscope_contract import DetectionRequest
        probe = SimpleNamespace(threshold=0.9)
        probe.spec = SimpleNamespace(probe_id='p')
        pipeline = ScoringPipeline({'probe': probe},
                                   SimpleNamespace(max_inflight=2, detect_batch=2), None)
        active = peak = calls = 0
        async def score(bucket, **kwargs):
            nonlocal active, peak, calls
            active += 1; calls += 1; peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                return [{'logit': 0.0, 'probability': 0.5, 'input_tokens': 3} for _ in bucket]
            finally: active -= 1
        pipeline.score_prompts = score
        request = DetectionRequest(messages=[{'role': 'user', 'content': 'hello'}])
        rows = await pipeline.detect_batch([request] * 100)
        assert len(rows) == 100 and calls == 50 and peak == 1
    asyncio.run(run())


def test_backend_limits_http_groups_and_rechecks_model():
    async def run():
        active = peak = verifications = 0
        changed = False
        async def handler(request):
            nonlocal active, peak, verifications
            if request.url.path == '/model_info':
                verifications += 1
                return httpx.Response(200, json={'model_type': 'qwen3', 'model_path': '/other' if changed else '/model'})
            active += 1; peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                import json
                body = json.loads(request.content)
                ids = body['input_ids']; rows = ids if isinstance(ids[0], list) else [ids]
                pos = body['feature_capture']['positions']['index']
                result = [{'meta_info': {'features': [{'schema_version': 'inscope.feature_capture.v1', 'site': 'block_output', 'layer': 9, 'positions': pos, 'shape': [len(pos), 4096], 'dtype': 'float32', 'encoding': 'float_list', 'data': [[0.0] * 4096 for _ in pos]}]}} for _ in rows]
                return httpx.Response(200, json=result if len(rows) > 1 else result[0])
            finally: active -= 1
        settings = SimpleNamespace(model_path='/model', backend_api_key='', max_inflight=8, backend_max_inflight=2, timeout=10)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='http://engine')
        backend = InScopeBackend(settings, client=client)
        await backend.capture_batch([list(range(n)) for n in range(4, 12)], (9,), (-1,))
        assert peak <= 2 and verifications == 8
        changed = True
        with pytest.raises(BackendError, match='differs'):
            await backend.capture([1, 2, 3], (9,), (-1,))
        await backend.close()
    asyncio.run(run())


@pytest.mark.parametrize('override', [
    {'detect_batch': -1}, {'analyze_batch': 0}, {'audit_timeout': float('nan')},
    {'audit_timeout': -1}, {'backend_max_inflight': 0}, {'max_batch_items': 0},
    {'backend_url': 'http://'}, {'backend_url': 'http://user:secret@example.com'},
])
def test_invalid_settings_rejected(override):
    with pytest.raises(ValueError):
        Settings(model_path='/model', api_key='test', probe_checkpoint='/probe',
                 composer_profile='a6_nemotron_risk_faced_v1', **override)


def test_settings_limits_from_env(monkeypatch):
    for key, value in {'INSCOPE_MODEL_PATH': '/model', 'INSCOPE_API_KEY': 'test', 'INSCOPE_PROBE_CHECKPOINT': '/probe', 'INSCOPE_COMPOSER_PROFILE': 'a6_nemotron_risk_faced_v1', 'INSCOPE_MAX_BATCH_ITEMS': '9', 'INSCOPE_BACKEND_MAX_INFLIGHT': '3', 'INSCOPE_MAX_BODY_BYTES': '1000'}.items():
        monkeypatch.setenv(key, value)
    s = Settings.from_env()
    assert (s.max_batch_items, s.backend_max_inflight, s.max_body_bytes) == (9, 3, 1000)


def test_all_scoring_routes_share_request_limit_and_release_on_cancel():
    async def run():
        from dataclasses import replace
        from inscope_server.app import create_app
        from inscope_server.context_profile import get_profile
        seed = _http_app(lambda: (0.0, 0.5), allow_single_capture=True)
        state = seed.state.inscope
        settings = Settings(model_path='/model', api_key='k', probe_checkpoint='/probe',
                            composer_profile='a6_nemotron_risk_faced_v1', max_inflight=1)
        app = create_app(settings, tokenizer=state['tokenizer'], probe=state['probe'], backend=state['backend'])
        entered = asyncio.Event()
        async def stall(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        state['backend'].capture_batch = stall
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')
        first = asyncio.create_task(client.post('/analyze', headers=HEADERS,
            json={'messages': _replay()['messages'], 'observation_index': 2}))
        await asyncio.wait_for(entered.wait(), 2)
        for path, payload in [('/detect', {'messages': [{'role': 'user', 'content': 'hello'}]}),
                              ('/audit', {'messages': [{'role': 'user', 'content': 'hello'}]})]:
            assert (await client.post(path, headers=HEADERS, json=payload)).status_code == 429
        assert (await client.get('/health/ready', headers=HEADERS)).status_code == 429
        first.cancel()
        with pytest.raises(asyncio.CancelledError): await first
        # Capacity released even on cancellation.
        assert (await client.get('/health/ready', headers=HEADERS)).status_code == 200
        await client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('path', ['/analyze', '/audit'])
def test_body_read_counts_toward_total_deadline(path):
    async def run():
        from inscope_server.app import create_app
        seed = _http_app(lambda: (0.0, 0.5), allow_single_capture=True)
        s = Settings(model_path='/model', api_key='k', probe_checkpoint='/probe',
                     composer_profile='a6_nemotron_risk_faced_v1', timeout=0.01,
                     analyze_timeout=0.03, audit_timeout=0.03)
        app = create_app(s, tokenizer=seed.state.inscope['tokenizer'],
                         probe=seed.state.inscope['probe'], backend=seed.state.inscope['backend'])
        async def chunks():
            yield b'{'
            await asyncio.sleep(0.2)
            yield b'}'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as c:
            r = await c.post(path, headers=HEADERS, content=chunks())
            assert r.status_code == 504
    asyncio.run(run())


def test_backend_failed_group_cancels_remaining_groups():
    async def run():
        settings = SimpleNamespace(model_path='/model', backend_api_key='', max_inflight=2, timeout=10)
        client = httpx.AsyncClient(base_url='http://engine')
        backend = InScopeBackend(settings, client=client)
        entered = asyncio.Event(); cancelled = asyncio.Event()
        async def group(batch, active, index, by_position=False):
            if len(batch[0]) == 3:
                await entered.wait()
                raise BackendError('failed group')
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        backend._capture_group = group
        with pytest.raises(BackendError):
            await backend.capture_batch([[1, 2, 3], [1, 2, 3, 4]], (9,), (-1,))
        assert cancelled.is_set()
        await backend.close()
    asyncio.run(run())


@pytest.mark.parametrize('url', ['http://engine:notaport', 'http://engine:70000', 'http://engine:0', 'http://engine/#fragment'])
def test_invalid_engine_ports_and_fragments(url):
    with pytest.raises(ValueError):
        Settings(model_path='/model', api_key='test', probe_checkpoint='/probe',
                 composer_profile='a6_nemotron_risk_faced_v1', backend_url=url)


@pytest.mark.parametrize('error,status', [(BackendError('offline'), 503), (TimeoutError('slow'), 504)])
def test_single_and_batch_backend_error_status(error, status):
    app = _http_app(lambda: (0.0, 0.5), allow_single_capture=True)
    async def fail(*args, **kwargs): raise error
    # 单条路径走 capture()（内部复用 capture_batch），两者都要注入失败。
    app.state.inscope['backend'].capture = fail
    app.state.inscope['backend'].capture_batch = fail
    with TestClient(app) as c:
        conversation = [{'role': 'user', 'content': 'hello'}]
        assert c.post('/detect', headers=HEADERS, json={'messages': conversation}).status_code == status
        r = c.post('/detect', headers=HEADERS, json={'messages': [conversation]})
        assert r.status_code == 200
        assert r.json()['results'][0]['status_code'] == status
        assert 'label' not in r.json()['results'][0]
        assert c.post('/analyze', headers=HEADERS, json={'messages': _replay()['messages'], 'observation_index': 2}).status_code == status


def test_each_request_can_choose_composer_profile():
    import inscope_server.app as app_module
    app = _http_app(lambda: (0.0, 0.5), allow_single_capture=True)
    original_analyze = app_module.localize_observation
    calls = []
    async def fake_analyze(request, *args, **kwargs):
        calls.append(dict(kwargs))
        return {'status': 'insufficient_comparison', 'mode_requested': 'auto'}
    app_module.localize_observation = fake_analyze
    try:
        with TestClient(app) as client:
            base = client.post('/detect', headers=HEADERS, json={'messages': [{'role': 'user', 'content': 'hello'}]})
            assert base.json()['composer_profile'] == 'a6_nemotron_risk_faced_v1'
            assert 'calibration_status' not in base.json()['composer']
            for name in ('no_reasoning', None):
                payload = {'messages': [{'role': 'user', 'content': 'hello'}]}
                if name:
                    payload['composer_profile'] = name
                assert client.post('/detect', headers=HEADERS, json=payload).json()['composer_profile'] == (name or 'a6_nemotron_risk_faced_v1')
                assert client.post('/analyze', headers=HEADERS, json={**payload, 'observation_index': 0}).status_code == 200
                assert calls[-1].get('case_id')
            assert client.post('/detect', headers=HEADERS, json={'messages': [{'role': 'user', 'content': 'hello'}], 'composer_profile': 'no_reasoning'}).json()['composer']['calibration_status'] == 'experimental_not_recalibrated'
            r = client.post('/audit', headers=HEADERS, json={'messages': [{'role': 'user', 'content': 'hello'}], 'composer_profile': 'no_reasoning'})
            assert r.status_code == 200 and r.json()['composer_profile'] == 'no_reasoning'
            for payload in ({'messages': [{'role': 'user', 'content': 'hello'}], 'composer_profile': 'missing'},
                            {'messages': [{'role': 'user', 'content': 'hello'}], 'composer_profile': 'a5_simplified_v1'},
                            {'messages': [{'role': 'user', 'content': 'hello'}], 'composer_profile': 1}):
                assert client.post('/detect', headers=HEADERS, json=payload).status_code == 422
            r = client.post('/detect', headers=HEADERS, json={'composer_profile': 'no_reasoning', 'messages': [
                [{'role': 'invalid'}], [{'role': 'user', 'content': 'hello'}]]})
            assert r.status_code == 200 and r.json()['composer_profile'] == 'no_reasoning'
            assert r.json()['results'][0]['status_code'] == 422
            assert isinstance(r.json()['results'][1]['composer_profile'], str)
    finally:
        app_module.localize_observation = original_analyze


def test_concurrent_requests_do_not_share_profile():
    async def run():
        import os
        os.environ.update({"INSCOPE_MODEL_PATH": "/model", "INSCOPE_API_KEY": "k",
                           "INSCOPE_PROBE_CHECKPOINT": "/probe",
                           "INSCOPE_COMPOSER_PROFILE": "a6_nemotron_risk_faced_v1"})
        from inscope_server.app import create_app
        seed = _http_app(lambda: (0.0, 0.5), allow_single_capture=True)
        state = seed.state.inscope
        app = create_app(Settings.from_env(), tokenizer=state['tokenizer'],
                         probe=state['probe'], backend=state['backend'])
        started = asyncio.Event()
        original = state['backend'].capture_batch
        original_capture = state['backend'].capture
        async def slow(batch_ids, layers, positions, **kwargs):
            started.set()
            await asyncio.sleep(0.05)
            return await original(batch_ids, layers, positions, **kwargs)
        async def slow_capture(ids, layers, positions, **kwargs):
            started.set()
            await asyncio.sleep(0.05)
            return await original_capture(ids, layers, positions, **kwargs)
        state['backend'].capture_batch = slow
        state['backend'].capture = slow_capture
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as c:
            async def call(profile):
                payload = {'messages': [{'role': 'user', 'content': 'hello'}]}
                if profile:
                    payload['composer_profile'] = profile
                return await c.post('/detect', headers=HEADERS, json=payload)
            other = asyncio.create_task(call('no_reasoning'))
            await asyncio.wait_for(started.wait(), 2)
            base = await call(None)
            assert base.json()['composer_profile'] == 'a6_nemotron_risk_faced_v1'
            assert (await other).json()['composer_profile'] == 'no_reasoning'
            follow = await call(None)
            assert follow.json()['composer_profile'] == 'a6_nemotron_risk_faced_v1'
    asyncio.run(run())


def test_sdk_sends_composer_profile_on_every_endpoint():
    import json
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        path = request.url.path
        if path == '/detect':
            return httpx.Response(200, json={
                'probability': 0.1, 'logit': -2.0, 'threshold': 0.9, 'label': 'benign',
                'risk': 'r', 'model': 'm', 'input_tokens': 3, 'request_id': 'id',
                'latency_ms': 1.0, 'composer_profile': 'no_reasoning', 'checkpoint_sha256': 's'})
        if path == '/analyze':
            return httpx.Response(200, json={'status': 'insufficient_comparison'})
        if path == '/audit':
            return httpx.Response(200, json={'status': 'clear', 'steps': []})
        raise AssertionError(path)
    transport = httpx.MockTransport(handler)
    from inscope.client import Client
    client = Client('http://test', 'k')
    client._session = httpx.Client(transport=transport, base_url='http://test')
    try:
        assert client.detect([{'role': 'user', 'content': 'hi'}], composer_profile='no_reasoning').composer_profile == 'no_reasoning'
        client.analyze([{'role': 'user', 'content': 'hi'}], 0, composer_profile='no_reasoning')
        client.audit([{'role': 'user', 'content': 'hi'}], composer_profile='no_reasoning')
    finally:
        client.close()
    assert [body.get('composer_profile') for body in seen] == ['no_reasoning'] * 3
