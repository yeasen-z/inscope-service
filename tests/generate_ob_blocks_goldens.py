"""生成 observation_blocks 对拍黄金样本（用 ipi-doom 训练侧实现执行）。

运行方式（在远端主机上）：
    /mnt/kpfs_qyang/djs/ipi-doom/.venv-inscope/bin/python tests/generate_ob_blocks_goldens.py

产物：tests/fixtures/observation_blocks_golden.json
每次 ipi-doom 编译规则变更后必须重新生成，否则对拍测试会暴露漂移。
"""
import base64
import json
import pathlib
import sys

sys.path.insert(0, "/mnt/kpfs_qyang/djs/ipi-doom/src")
sys.path.insert(0, "/mnt/kpfs_qyang/djs/ipi-doom")

import torch
from transformers import AutoTokenizer

from follow_your_heart.data_collection.schema import DecisionPointRecord
from featurize.observation_blocks import build_block_prompt
from featurize.feature_quantization import encode_scaled_e4m3
from probe_training.debug import ProbeScorer

MODEL = "/mnt/kpfs_qyang/djs/models/Qwen3-8B-FP8"
CHECKPOINT = ("/mnt/kpfs_qyang/djs/ipi-doom/results/attention-head-sweep-20261027/"
              "attention-h1/best.pt")
SPEC = {
    "components": [CHECKPOINT],
    "feature_layout": "observation_blocks",
    "layers": [8, 16, 24],
    "storage_dtype": "float8_e4m3fn",
}

TOOLS = [{
    "type": "function",
    "function": {"name": "read_file", "parameters": {"type": "object",
                 "properties": {"path": {"type": "string"}}}},
}, {
    "type": "function",
    "function": {"name": "web_search", "parameters": {"type": "object",
                 "properties": {"query": {"type": "string"}}}},
}]

CASES = [
    ("single_tool_observation", {
        "messages": [
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": "Read the report and summarize it."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\": \"report.txt\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
             "content": "Section 1: revenue grew 12% year over year.\n\nSection 2: costs remain stable.\n\nSection 3: outlook is positive."},
        ], "tools": TOOLS}),
    ("user_carried_tool_response", {
        "messages": [
            {"role": "user", "content": "Check the weather for me."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_2", "type": "function",
                 "function": {"name": "web_search", "arguments": "{\"query\": \"weather today\"}"}}]},
            {"role": "user",
             "content": "<tool_response>Search results: sunny, 22C, light wind.\n\nTomorrow: cloudy.</tool_response>"},
        ], "tools": TOOLS}),
    ("parallel_tool_results", {
        "messages": [
            {"role": "user", "content": "Compare file A and file B."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}},
                {"id": "call_b", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\": \"b.txt\"}"}}]},
            {"role": "tool", "tool_call_id": "call_a", "name": "read_file",
             "content": "A: alpha beta gamma."},
            {"role": "tool", "tool_call_id": "call_b", "name": "read_file",
             "content": "B: gamma delta epsilon.\n\nB second paragraph here."},
        ], "tools": TOOLS}),
    ("discard_cots_closed_think", {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant",
             "content": "<think>Simple arithmetic.</think>The answer is 4."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_3", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\": \"notes.txt\"}"}}]},
            {"role": "tool", "tool_call_id": "call_3", "name": "read_file",
             "content": "Notes confirm the result.\n\nAdditional context line."},
        ], "tools": TOOLS}),
    ("developer_role_and_merge", {
        "messages": [
            {"role": "developer", "content": "Always cite sources."},
            {"role": "user", "content": "Find a source."},
            {"role": "assistant", "content": "Searching now.", "reasoning_content": "need a source"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_4", "type": "function",
                 "function": {"name": "web_search", "arguments": "{\"query\": \"citations\"}"}}]},
            {"role": "tool", "tool_call_id": "call_4", "name": "web_search",
             "content": "Found 3 sources.\n\nSource 1 is most relevant."},
        ], "tools": TOOLS}),
    ("multi_block_single_observation", {
        "messages": [
            {"role": "user", "content": "Inspect the log."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_5", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\": \"log.txt\"}"}}]},
            {"role": "tool", "tool_call_id": "call_5", "name": "read_file",
             "content": "INFO boot ok.\n\nWARN disk slow.\n\nERROR retry failed.\n\nINFO recovered."},
        ], "tools": TOOLS}),
]


def point(request, index):
    return DecisionPointRecord.from_dict(dict(
        decision_point_id=str(index), trace_id=str(index), grid_point_id="probe-debug",
        decision_index=0, assistant_message_index=len(request["messages"]),
        replay_request_kind="openai_chat", replay_request=request))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    scorer = ProbeScorer(SPEC, [], tokenizer_path=MODEL)
    records = []
    for index, (name, request) in enumerate(CASES):
        compiled = build_block_prompt(point(request, index), tokenizer=tokenizer)
        raw = torch.randn(len(compiled.positions), 12288,
                          generator=torch.Generator().manual_seed(20261029 + index)) * 2.0
        packed, scales = encode_scaled_e4m3(raw)
        decoded = packed.float() * scales.float().unsqueeze(-1)
        explained = scorer.score_features([decoded], blocks=[list(compiled.blocks)],
                                          explain=True)[0]
        channels = explained["component_importance"][0]["channels"]
        records.append({
            "name": name,
            "request": request,
            "text": compiled.prompt.text,
            "token_ids": list(compiled.prompt.token_ids),
            "positions": list(compiled.positions),
            "blocks": [dict(block) for block in compiled.blocks],
            "raw_features_f32_b64": base64.b64encode(
                raw.numpy().astype("<f4").tobytes()).decode("ascii"),
            "expected_logit": explained["logit"],
            "expected_attention_weight": channels["attention_weight"],
            "expected_attention_contribution": channels["attention_contribution"],
        })
    out = pathlib.Path(__file__).parent / "fixtures" / "observation_blocks_golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "checkpoint": CHECKPOINT,
                               "layers": [8, 16, 24], "cases": records},
                              ensure_ascii=False, indent=1))
    print("wrote", out, len(records), "cases")


if __name__ == "__main__":
    main()
