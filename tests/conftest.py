"""测试共用夹具：构造形状合规的 checkpoint，避免各文件重复拼装。"""

import json

import torch

LAYERS = 3
POSITIONS = 4
HIDDEN = 4096
DIM = LAYERS * POSITIONS * HIDDEN

DEFAULT_FEATURE = "no_prefix_cot_last4_20261015"
DEFAULT_DATASET = "arm6_Nemotron-Post-Training-Dataset-v1-tool_calling"


def make_state(dim=DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "input_mean": torch.randn(1, dim, generator=g) * 10,
        "input_std": torch.rand(1, dim, generator=g) + 0.5,
        "probe.weight": torch.randn(1, dim, generator=g) * 0.01,
        "probe.bias": torch.tensor([-0.1]),
    }


def write_checkpoint(tmp_path, *, sub=None, blocks=(8, 16, 24),
                     positions=(-4, -3, -2, -1), dim=DIM,
                     arch="linear", schema="fyh.probe_checkpoint.v2",
                     feature_name=DEFAULT_FEATURE, dataset=None,
                     manifest="auto", state=None, probe_id=None):
    """写一份 checkpoint（可选 manifest.json），返回文件路径。"""
    directory = tmp_path / sub if sub else tmp_path
    directory.mkdir(parents=True, exist_ok=True)

    ck = {
        "schema": schema,
        "probe_architecture": arch,
        "input_dim": dim,
        "dataset_name": dataset if dataset is not None else DEFAULT_DATASET,
        "selected_layer_indices": list(blocks),
        "selected_positions": list(positions),
        "feature_normalization": "standard",
        "threshold": 0.9,
        "labeling_protocol": "risk_faced",
        "feature_name": feature_name,
        "model_state_dict": state if state is not None else make_state(dim),
    }
    if probe_id:
        ck["probe_id"] = probe_id
    path = directory / "best.pt"
    torch.save(ck, path)

    if manifest == "auto":
        # 与真实 handoff 一致：由 manifest 显式给出边界层号与铺平顺序。
        manifest = {"inscope_layers": [b + 1 for b in blocks],
                    "flatten_order": ["batch", "layer", "token", "hidden"]}
    if manifest is not None:
        (directory / "manifest.json").write_text(json.dumps(manifest))
    return path
