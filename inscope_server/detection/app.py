"""HTTP adapter for the modular pipeline. Existing legacy API stays unchanged."""
import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
import msgspec
from pydantic import ValidationError

from inscope_contract.detection_pipeline import CompiledDetectionRequest,CompiledDetectionResponse
from .config import build_pipeline,load_config
from .scheduler import Overloaded

log=logging.getLogger(__name__)


def create_pipeline_app(config_path=None,*,api_key,pipeline=None,max_inflight=4096,
                        timeout=900,max_body_bytes=4*1024**2,device='cuda'):
    if not api_key: raise ValueError('INSCOPE_API_KEY is required')
    if min(max_inflight,timeout,max_body_bytes)<=0: raise ValueError('Invalid API resource limits')
    state=dict(pipeline=pipeline,ready=False,active=0)
    @asynccontextmanager
    async def lifespan(app):
        config=None
        if state['pipeline'] is None:
            config=await asyncio.to_thread(load_config,config_path)
            state['pipeline']=await asyncio.to_thread(build_pipeline,config,device)
        try:
            await state['pipeline'].backend.verify()
            bank=state['pipeline'].scheduler.bank
            if hasattr(bank,'warmup'):
                await asyncio.to_thread(bank.warmup)
            await state['pipeline'].scheduler.start()
            if config is not None and config.get('readiness_request'):
                request=CompiledDetectionRequest.model_validate(config['readiness_request'])
                await state['pipeline'].detect(request.model_dump())
            if hasattr(state['pipeline'].backend,'start_health_monitor'):state['pipeline'].backend.start_health_monitor()
            state['ready']=True
            yield
        finally:
            state['ready']=False
            await state['pipeline'].close()
    app=FastAPI(title='InScope compiled-context detection API',version='0.7.0',lifespan=lifespan)
    app.state.inscope=state
    def authenticate(request):
        if not hmac.compare_digest(request.headers.get('authorization','').encode(),('Bearer '+api_key).encode()):
            raise HTTPException(401,'Invalid API key',headers={'WWW-Authenticate':'Bearer'})
    @app.get('/health/live')
    async def live(): return dict(status='alive')
    @app.get('/health/ready')
    async def ready(request:Request):
        authenticate(request)
        backend=state['pipeline'].backend if state['pipeline'] else None
        worker=state['pipeline'].scheduler.worker if state['pipeline'] else None
        if not state['ready'] or getattr(backend,'verified',True) is False or (worker is not None and worker.done()) or (getattr(backend,'errors',None) and len(backend.errors)==len(backend.endpoints)):
            raise HTTPException(503,'Detection pipeline unavailable')
        bank=state['pipeline'].scheduler.bank
        return dict(status='ready',layers=bank.layers,checkpoint_sha256=getattr(bank,'identities',[]),
            readiness='checkpoint load, GPU warmup and native endpoint identity verified; use /detect for end-to-end canary')
    @app.get('/status')
    async def status(request:Request):
        authenticate(request)
        return state['pipeline'].status()
    @app.post('/detect',response_model=CompiledDetectionResponse,openapi_extra={
        'requestBody':{'required':True,'content':{'application/json':{'schema':CompiledDetectionRequest.model_json_schema()}}}})
    async def detect(request:Request):
        authenticate(request)
        if not state['ready']: raise HTTPException(503,'Detection pipeline unavailable')
        if state['active']>=max_inflight: raise HTTPException(429,'Too many in-flight requests',headers={'Retry-After':'1'})
        state['active']+=1
        try:
            async with asyncio.timeout(timeout):
                body=bytearray()
                async for chunk in request.stream():
                    if len(body)+len(chunk)>max_body_bytes: raise HTTPException(413,'Request body too large')
                    body.extend(chunk)
                try:
                    value=CompiledDetectionRequest.model_validate_json(bytes(body))
                except ValidationError as error:
                    raise HTTPException(422,'Invalid compiled detection request') from error
                try:
                    result=await state['pipeline'].detect(value.model_dump())
                    encoded=await asyncio.to_thread(msgspec.json.encode,result)
                    return Response(encoded,media_type='application/json')
                except Overloaded as error: raise HTTPException(429,str(error),headers={'Retry-After':'1'}) from error
                except ValueError as error: raise HTTPException(422,str(error)) from error
                except RuntimeError as error:
                    log.warning('Detection runtime failed: %s',error)
                    raise HTTPException(503,'Detection backend unavailable') from error
        except TimeoutError as error: raise HTTPException(504,'Detection request deadline exceeded') from error
        finally: state['active']-=1
    return app
