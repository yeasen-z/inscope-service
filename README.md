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

## 快速开始

```bash
# 只装客户端（调用方）
pip install inscope

# 装服务端
pip install inscope[server]
```

启动服务（需要先把 sglang-inscope 的 engine 跑起来）：

```bash
export INSCOPE_MODEL_PATH=/path/to/Qwen3-8B-FP8
export INSCOPE_API_KEY=<your-key>
export INSCOPE_ENGINE_URL=http://127.0.0.1:30000
export INSCOPE_DETECTOR_DIR=/path/to/detector
export INSCOPE_COMPOSER_PROFILE=a5_simplified_v1

inscope-serve --host 0.0.0.0 --port 8302
```

调用：

```python
from inscope import Client

client = Client("http://127.0.0.1:8302", api_key="...")
result = client.detect([{"role": "user", "content": "..."}])
print(result.flagged, result.logit, result.threshold_logit)
```

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
| residual boundary | SGLang 的说法：`0`=embedding 输出，`1..N-1`=第 `L-1` 个 block 输出，`N`=最后一层 |

换算规则：

```text
BLOCK_OUTPUT  ->  boundary = block + 1
BOUNDARY      ->  原样透传
```

**checkpoint 未声明 `layer_semantics` 时直接报错**，不做默认假设。任何模块都不允许
自己写 `block + 1`。

## Context Composer

送进模型提取特征的上下文不一定是调用方原样传的 messages。Composer 按 **profile**
决定如何组织（选窗、去 CoT、合并 assistant、模板渲染、追加探测问题），profile 与
探针**绑定**：

```python
ComposerProfile(
    profile_id="a5_simplified_v1",
    version="1",
    probe_id="a5_blend",       # 不匹配就拒绝启动
    position_mode="last",
)
```

- 未知 profile → 启动失败，不回落默认
- 没有实现的 profile → HTTP 501，不用别的规则凑
- 响应里回显 `profile_id` / `profile_version`，便于事后审计

## 开发

```bash
pip install -e ".[dev]"
pytest -q
```

测试分三类，均可无 GPU 运行：

- `test_architecture.py` — import 方向约束
- `test_layers.py` — 层号换算
- `test_api_cpu.py` — 契约常量、特征解码校验、请求校验
- `test_composer_profile.py` — profile 注册与绑定

## 不做什么

- 不训练探针，不产出权重
- 不实现 SGLang 的激活采集（在 `sglang-inscope`）
- 不改写模型输出，只读激活

## License

见 [LICENSE](LICENSE)。
