"""Optional real-engine contract check: no GPU/server, real request normalization.

The transport only supplies synthetic feature values. Request validation uses
SGLang's actual FastAPI/Pydantic request boundary and normalization methods,
not a handwritten assertion that merely accepts the service's payload.
"""
import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

sglang = pytest.importorskip('sglang', reason='real engine contract requires optional sglang environment')
from pydantic import TypeAdapter, ValidationError
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.feature_capture.spec import validate as normalize_feature_request
from inscope_server.backend import InScopeBackend


def test_real_generate_request_boundary_rejects_old_top_level_array():
    adapter = TypeAdapter(GenerateReqInput)
    with pytest.raises(ValidationError):
        adapter.validate_python([{'input_ids': [1, 2, 3]}])


def test_service_payload_runs_real_engine_normalization():
    async def run():
        seen = []
        adapter = TypeAdapter(GenerateReqInput)
        def handler(request):
            if request.url.path == '/model_info':
                return httpx.Response(200, json={'model_type': 'qwen3', 'model_path': '/model'})
            payload = json.loads(request.content)
            req = adapter.validate_python(copy.deepcopy(payload))
            req.normalize_batch_and_arguments()
            ids = payload['input_ids']
            rows = ids if isinstance(ids[0], list) else [ids]
            normalized_specs = req.feature_capture if isinstance(req.feature_capture, list) else [req.feature_capture]
            results = []
            for row, raw in zip(rows, normalized_specs):
                spec = normalize_feature_request(raw, num_layers=36, hidden_size=4096, prompt_ids=row)
                results.append({'meta_info': {'features': [
                    {'schema_version': 'inscope.feature_capture.v1', 'layer': target.layer,
                     'site': target.site, 'positions': list(spec.positions), 'shape': [len(spec.positions), 4096],
                     'dtype': 'float32', 'encoding': spec.encoding,
                     'data': [[float(len(row))] * 4096 for _ in spec.positions]}
                    for target in spec.targets]}})
            seen.append(len(rows))
            return httpx.Response(200, json=results if len(rows) > 1 else results[0])
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='http://engine')
        settings = SimpleNamespace(model_path='/model', backend_api_key='', timeout=10, max_inflight=2)
        backend = InScopeBackend(settings, client=client)
        equal = await backend.capture_batch([list(range(20))] * 3, (9, 17), (-4, -1))
        mixed = await backend.capture_batch([list(range(10)), list(range(40))], (9,), (-1,))
        assert seen == [3, 1, 1]
        assert len(equal) == 3 and all(len(item) == 2 for item in equal)
        assert [float(item[0][0, 0]) for item in mixed] == [10.0, 40.0]
        await backend.close()
    asyncio.run(run())
