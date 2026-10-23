# InScope

基于模型激活的间接提示注入（IPI）检测服务。

InScope 在推理服务内部**只读地**采集指定 decoder 层的残差流激活，用一个轻量探针
给出风险分，不改变模型的输出行为。

## 定位

InScope 是三层架构的**核心服务层**：

```text
┌──────────────────────────────────────────────┐
│ coolwatch          业务平台                   │
│ 前端 / 管理后台 / 评估 / 数据管线              │
└──────────────────┬───────────────────────────┘
                   │ pip install inscope
                   │ from inscope import Client
                   ▼
┌──────────────────────────────────────────────┐
│ inscope-service    ← 本仓库                   │
│ 契约 + 检测服务 + 客户端 SDK                   │
└──────────────────┬───────────────────────────┘
                   │ HTTP /generate + feature_capture
                   ▼
┌──────────────────────────────────────────────┐
│ sglang-inscope     底层驱动                   │
│ SGLang fork，提供 residual-stream 激活采集     │
└──────────────────────────────────────────────┘
```

本仓库不包含模型权重，也不包含探针 checkpoint。**权重单独分发。**

## 核心设计：探针由 checkpoint 自描述

服务**不预设任何一套探针结构**。层号、位置、归一化方式、拼接顺序、架构和阈值
全部从 checkpoint（及其同目录的 `manifest.json`）读取：

| 参数 | 来源 |
| --- | --- |
| 层号 | `manifest.inscope_layers`，否则按 `layer_semantics` 由 `block_indices` 换算 |
| 位置 | `selected_positions`，如 `[-4,-3,-2,-1]` |
| 归一化 | `model_state_dict.input_mean` / `input_std` |
| 拼接顺序 | `manifest.flatten_order` |
| 架构 | `probe_architecture` |
| 阈值 | `threshold` |

未知 schema、未知架构、未知归一化方式**一律报错**，不回退到任何默认探针。
`manifest.sha256` 存在时会校验，权重与 manifest 不匹配直接拒绝。

这意味着换一个 checkpoint 不需要改代码，只需要换文件并选对 composer profile。

## 快速开始

```bash
# 只装客户端（调用方）：不会拖进 torch / fastapi
pip install inscope

# 装服务端：起服务必须装这个
pip install inscope[server]
```

> 只装 `inscope` 后直接起服务会在运行期报 ImportError，这是刻意的取舍：
> 客户端保持轻量，服务端必须显式安装 `[server]` extra。

启动服务（需要先把 sglang-inscope 的 engine 跑起来）：

```bash
export INSCOPE_MODEL_PATH=/path/to/Qwen3-8B-FP8
export INSCOPE_API_KEY=<对外鉴权口令>
export INSCOPE_ENGINE_URL=http://127.0.0.1:31100
export INSCOPE_PROBE_CHECKPOINT=/path/to/best.pt
export INSCOPE_COMPOSER_PROFILE=a6_nemotron_risk_faced_v1

inscope-serve --host 0.0.0.0 --port 8302
```

启动前可先做一次自检（不启动服务，失败返回码 2）：

```bash
inscope-serve --preflight-only
```

自检会打印探针 id、层号、位置、维度、阈值和 profile，并在以下情况直接失败：

- 缺 `INSCOPE_PROBE_CHECKPOINT` 或 `INSCOPE_COMPOSER_PROFILE`
- profile id 未知
- profile 与探针的 `probe_id` 不匹配
- profile 声明的取位置与 checkpoint 声明不一致
- checkpoint 缺失、schema 不支持、维度不自洽

调用：

```python
from inscope import Client

client = Client("http://127.0.0.1:8302", api_key="...")
result = client.detect([{"role": "user", "content": "..."}])
print(result.label, result.probability, result.threshold)
```

异步调用用 `inscope.client.AsyncClient`。

## 包结构

| 包 | 作用 | 依赖方向 |
| --- | --- | --- |
| `inscope_contract` | 协议：端点、常量、数据模型、层号语义 | 不依赖任何上层包 |
| `inscope_server` | FastAPI 服务：Composer → 特征采集 → 打分 | → contract |
| `inscope` | 同步/异步客户端 SDK | → contract |

依赖是**严格单向**的，由 `tests/test_architecture.py` 静态检查 import 图来保证。
`inscope_server` 与 `inscope` 之间禁止互相 import。

## 层号语义（重要）

三层对「层」的说法不同，换算**只有一处**：`inscope_contract/layers.py`。

| 概念 | 含义 |
| --- | --- |
| block index | 探针 checkpoint 里的说法：「第 b 个 decoder block 的输出」 |
| residual boundary | SGLang 的说法：`0`=embedding 输出，`1..N-1`=第 `L-1` 个 block 输出，`N`/`-1`=最后一层 |

换算规则：

```text
BLOCK_OUTPUT  ->  boundary = block + 1
BOUNDARY      ->  原样透传
```

Qwen3-8B 是 36 层，所以 `block_indices [8,16,24]` 对应 `inscope_layers [9,17,25]`。

**checkpoint 未声明语义且无 manifest 时直接报错**，不做默认假设。
任何模块都不允许自己写 `block + 1`。

## Context Composer

送进模型提取特征的上下文不一定是调用方原样传的 messages。Composer 按 **profile**
决定如何组织（选窗、去 CoT、合并 assistant、模板渲染、追加探测问题），profile 与
探针通过 `probe_id` **绑定**：

```python
ComposerProfile(
    profile_id="a6_nemotron_risk_faced_v1",
    version="1",
    probe_id="linear:arm6_...:no_prefix_cot_last4_20261015",  # 不匹配就拒绝启动
    position_mode="last",
)
```

- 未知 profile → 启动失败，不回落默认
- profile 与探针 `probe_id` 不匹配 → 启动失败
- profile 取的 token 位置与 checkpoint 声明不一致 → 启动失败
- 没有实现的 profile → HTTP 501，不用别的规则凑
- 响应里回显 `profile_id` / `profile_version`，便于事后审计

`probe_id` 未在 checkpoint 中显式声明时，由 `架构:训练集:feature_name` 派生，
保证同一份权重得到同一个 id、不同权重得到不同 id。

### 已注册 profile

| profile | 适用探针 | 前缀 | 位置 |
| --- | --- | --- | --- |
| `a6_nemotron_risk_faced_v1` | `fyh.probe_checkpoint.v2` 线性探针 | **无** | `[-4,-3,-2,-1]` |
| `a5_simplified_v1` | A5 blend（**当前未支持其 checkpoint 布局**） | 有 | `[-1]` |

> `a6` 这个探针训练时**没有**探测问题前缀（`feature_name` 即 `no_prefix_...`）。
> 给它加前缀会让分数完全反转 —— 实测 attack 样本的 logit 从 `+28` 掉到 `-12`。
> profile 里的参数必须与训练配置严格一致，不能凭语义相似去猜。

## 开发

```bash
pip install -e ".[dev]"
pytest -q
```

测试全部可无 GPU 运行：

| 文件 | 覆盖 |
| --- | --- |
| `test_architecture.py` | import 方向约束 |
| `test_layers.py` | 层号换算 |
| `test_api_cpu.py` | 契约常量、特征解码（含多位置）、请求校验 |
| `test_composer_profile.py` | profile 注册与绑定 |
| `test_probe_checkpoint.py` | checkpoint 自适应加载的各种拒绝路径 |
| `test_client_contract.py` | 客户端与服务字段契约、profile/probe 绑定 |

`tests/conftest.py` 提供构造合规 checkpoint 的共用夹具。

## 不做什么

- 不训练探针，不产出权重
- 不实现 SGLang 的激活采集（在 `sglang-inscope`）
- 不改写模型输出，只读激活

## License

见 [LICENSE](LICENSE)。
