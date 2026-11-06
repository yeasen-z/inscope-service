import asyncio
import msgspec
import torch
import zmq
import zmq.asyncio
import pytest
from inscope_server.detection.composer import CompiledComposer
from inscope_server.detection.planner import AllPositionPlanner
from inscope_server.detection.native_backend import NativeCaptureBackend


@pytest.mark.parametrize('broken',[False,True])
def test_native_partial_frames_identity_and_cleanup(broken):
    async def run():
        ctx=zmq.asyncio.Context(); sock=ctx.socket(zmq.ROUTER)
        port=sock.bind_to_random_port('tcp://127.0.0.1'); endpoint=f'tcp://127.0.0.1:{port}'
        async def serve():
            while True:
                identity,payload=await sock.recv_multipart()
                if payload.startswith(b'fyh-abort:'):continue
                if payload.startswith(b'fyh-info:'):
                    info=dict(rid=payload[len(b'fyh-info:'):].decode(),model_path='model',
                        feature_capture_graph_layers=[0,3],feature_capture_early_exit_layer=16)
                    await sock.send_multipart([identity,b'fyh-info:'+msgspec.json.encode(info)]);continue
                req=msgspec.msgpack.decode(payload); positions=req['selected_positions']
                for start in range(0,len(positions),2):
                    selected=positions[start:start+2]; n=len(selected)
                    values=torch.tensor([[[p,p+1],[p+2,p+3]] for p in selected],dtype=torch.bfloat16)
                    frame=dict(schema='fyh.feature_batch.v1',batch_seq=start,rids=[req['rid']],
                        records=[dict(rid=req['rid'],row_start=0,row_count=n,positions=selected,
                            complete=start+n==len(positions))],layer_ids=[0,3],shape=[n,2,2],
                        dtype='bfloat16',hidden_bytes=values.view(torch.uint8).numpy().tobytes())
                    if broken:frame['layer_ids']=[0,4]
                    await sock.send_multipart([identity,msgspec.msgpack.encode(frame)])
        task=asyncio.create_task(serve())
        backend=NativeCaptureBackend([endpoint],hidden_dim=2,model_path='model',layers=[0,3],timeout=2)
        try:
            await backend.verify()
            prompt=CompiledComposer('v1').compose(dict(composer_id='v1',input_ids=list(range(9)),
                pre_position=1,observation_tail_position=3,post_position=5,suffix_position=7))
            plan=AllPositionPlanner().plan(prompt,(0,3))
            if broken:
                with pytest.raises(ValueError):await backend.capture(prompt,plan)
            else:
                results=await asyncio.gather(*(backend.capture(prompt,plan) for _ in range(10)))
                for region,anchors in results:
                    assert region.shape==(3,4) and region[:,0].tolist()==[1,2,3]
                    assert anchors[:,0].tolist()==[5,7]
            assert backend.tokens==[0] and backend.requests==[0] and not backend.channels[0]
        finally:
            await backend.close(); task.cancel();await asyncio.gather(task,return_exceptions=True)
            sock.close(0);ctx.destroy(linger=0)
    asyncio.run(run())
