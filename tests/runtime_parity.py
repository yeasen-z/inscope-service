"""runtime_parity：真实引擎特征下，inscope-service 与 ipi-doom 训练侧逐项对拍。

流程（同一请求、同一次真实 prefill）：
1. 用 ipi-doom 的 build_block_prompt 编译请求，与服务 /detect 返回的
   compiled_context / capture_positions 逐项比对；
2. 用服务暴露的 input_ids + 位置，直接从真引擎按 base64 抓 BF16 特征；
3. 走 ProbeScorer._score_features（含 E4M3 量化往返）得到 logit 与 block 通道；
4. 与服务响应里的同名字段比对。

用法：
    /mnt/kpfs_qyang/djs/ipi-doom/.venv-inscope/bin/python runtime_parity.py <detect_response.json> <request.json>
"""
import json
import pathlib
import sys
import urllib.request
from types import SimpleNamespace

sys.path.insert(0, "/mnt/kpfs_qyang/djs/ipi-doom/src")
sys.path.insert(0, "/mnt/kpfs_qyang/djs/ipi-doom")

import torch
from transformers import AutoTokenizer

from featurize.observation_blocks import build_block_prompt
from follow_your_heart.data_collection.schema import DecisionPointRecord
from probe_training.debug import ProbeScorer

MODEL = "/mnt/kpfs_qyang/djs/models/Qwen3-8B-FP8"
CHECKPOINT = ("/mnt/kpfs_qyang/djs/ipi-doom/results/attention-head-sweep-20261027/"
              "attention-h1/best.pt")
ENGINE = "http://127.0.0.1:31001"
LAYERS = (8, 16, 24)
SPEC = {"components": [CHECKPOINT], "feature_layout": "observation_blocks",
        "layers": list(LAYERS), "storage_dtype": "float8_e4m3fn"}


def fetch_features(input_ids, positions):
    """按训练侧口径（base64）抓取 BF16 特征。"""
    payload = {
        "input_ids": list(input_ids),
        "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
        "feature_capture": {"layers": list(LAYERS), "site": "block_output",
                            "positions": {"mode": "explicit", "index": list(positions)},
                            "encoding": "base64"},
    }
    request = urllib.request.Request(
        ENGINE + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    body = json.load(urllib.request.urlopen(request, timeout=300))
    records = body["meta_info"]["features"]
    by_layer = {int(record["layer"]): record for record in records}
    hidden = []
    for layer in LAYERS:
        record = by_layer[layer]
        raw = bytearray(__import__("base64").b64decode(record["data"], validate=True))
        values = torch.frombuffer(raw, dtype=getattr(torch, record["dtype"])).reshape(record["shape"])
        assert record["positions"] == list(positions), (record["positions"], positions)
        hidden.append(values)
    return hidden


def main():
    response = json.loads(pathlib.Path(sys.argv[1]).read_text())
    request = json.loads(pathlib.Path(sys.argv[2]).read_text())
    request.pop("explain", None)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    point = DecisionPointRecord.from_dict(dict(
        decision_point_id="parity", trace_id="parity", grid_point_id="probe-debug",
        decision_index=0, assistant_message_index=len(request["messages"]),
        replay_request_kind="openai_chat", replay_request=request))
    compiled = build_block_prompt(point, tokenizer=tokenizer)

    checks = []
    checks.append(("compiled_context", compiled.prompt.text == response["compiled_context"]))
    checks.append(("capture_positions", list(compiled.positions) == list(response["capture_positions"])))
    blocks = [dict(block) for block in compiled.blocks]
    service_blocks = [{key: value for key, value in block.items()
                       if key not in ("attention_weight", "attention_contribution",
                                      "source_message_indices")}
                      for block in response["blocks"]]
    checks.append(("blocks", blocks == service_blocks))
    print("[compile] text=%s positions=%s blocks=%s"
          % (checks[0][1], checks[1][1], checks[2][1]))
    if not all(ok for _, ok in checks):
        for name, ok in checks:
            if not ok:
                print("  MISMATCH:", name)
        raise SystemExit("compile parity failed")

    hidden = fetch_features(compiled.prompt.token_ids, compiled.positions)
    print("[engine] layers=%s per-layer shape=%s dtype=%s"
          % (list(LAYERS), tuple(hidden[0].shape), hidden[0].dtype))

    scorer = ProbeScorer(SPEC, [], tokenizer_path=MODEL)
    capture = SimpleNamespace(layer_ids=tuple(LAYERS), hidden_states=hidden)
    explained = scorer._score_features([capture], blocks=[blocks], explain=True)[0]
    channels = explained["component_importance"][0]["channels"]

    service_logit = response["logit"]
    training_logit = explained["logit"]
    print("[score] service_logit=%.6f training_logit=%.6f delta=%.2e"
          % (service_logit, training_logit, abs(service_logit - training_logit)))

    worst_w = worst_c = 0.0
    service_blocks = response["blocks"]
    for index, block in enumerate(service_blocks):
        worst_w = max(worst_w, abs(block["attention_weight"] - channels["attention_weight"][index]))
        worst_c = max(worst_c, abs(block["attention_contribution"]
                                   - channels["attention_contribution"][index]))
    print("[explain] max|Δweight|=%.2e max|Δcontribution|=%.2e" % (worst_w, worst_c))

    print("[verdict] logit_delta_ok=%s weight_ok=%s contribution_ok=%s"
          % (abs(service_logit - training_logit) < 1e-5, worst_w < 1e-9, worst_c < 1e-9))
    print("[service] label=%s prob=%.6f threshold=%.6f"
          % (response["label"], response["probability"], response["threshold"]))


if __name__ == "__main__":
    main()
