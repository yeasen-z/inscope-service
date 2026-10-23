# InScope Service — Handoff

> 交接日期：2026-10-23
> 仓库：https://github.com/yeasen-z/inscope-service
> 基线提交：`499a2c0`，本次改造在其之上

---

## 1. 这是什么

InScope Service 是激活值探针（activation probe）的检测服务，位于三层架构的中间层：

```text
┌─────────────────────────────┐
│ coolwatch                   │  最外层业务平台
├─────────────────────────────┤
│ inscope-service  ← 本仓库    │  核心服务层（contract + server + client）
├─────────────────────────────┤
│ sglang-inscope              │  底层驱动（带 feature_capture 的 SGLang fork）
└─────────────────────────────┘
```

一个仓库、两个安装面：

```bash
pip install inscope            # 仅客户端（不拖入 torch / fastapi）
pip install inscope[server]    # 服务端（含 fastapi / uvicorn / torch / transformers）
```

> 只装 `inscope` 后直接起服务会在运行期报 ImportError。这是刻意的设计取舍：
> 客户端保持轻量，服务端必须显式安装 `[server]` extra。

---

## 2. 本次改造的核心结论

**服务不能预设任何一套探针结构。层号、位置、归一化、拼接顺序、架构、阈值，
全部由 checkpoint 自己声明。**

改造前 `probe.py` 写死了 A5 blend（两个 arm 融合、层号 `(9,17,25)`、单位置）。
改造后改为按 checkpoint 自描述加载，这直接源于一条实际需求：

> "我要的就是根据权重来灵活调整，这不就是 server 的功能吗"

### 2.1 已落地的能力

| 能力 | 实现位置 | 说明 |
|---|---|---|
| 按 checkpoint 声明加载架构 | `probe.py::load_probe` | 未知 schema / 架构 / 归一化方式一律报错，不回退默认 |
| 层号自适应 | `probe.py::_resolve_layers` | 优先用 manifest 的 `inscope_layers`；否则按 `layer_semantics` 换算 |
| 位置自适应 | `backend.py::capture(positions=)` | 位置数量不再写死单点 |
| 拼接顺序自适应 | `probe.py::Probe.flatten` | 按 manifest 的 `flatten_order`，并校验每层 `[n_pos, hidden]` |
| 阈值来自 checkpoint | `probe.py::load_probe` | 不再查冻结的 seed 阈值表 |
| 校验 manifest sha256 | `probe.py::load_probe` | 权重与 manifest 不匹配直接拒绝 |
| probe_id 派生 | `probe.py::_derive_probe_id` | `架构:训练集:feature_name`，同一权重恒定、不同权重不同 |
| profile 与探针互认 | `app.py` / `cli.py::_preflight` | `probe_id` 不匹配或位置不一致 → 启动失败 |
| 启动自检 | `inscope-serve --preflight-only` | 缺项/错配在启动前报出，返回码 2 |

### 2.2 层号语义（重要）

SGLang 的 `layers` 是 **residual-stream 边界号**，不是 block 索引：

```text
0            = embedding 输出
1..N-1       = 第 L-1 个 decoder block 的输出
N 或 -1      = 最后一层输出（final norm 之前）
```

因此当 checkpoint 声明的是 **block 输出**时，需要换算：

```text
block b  →  boundary b + 1
```

Qwen3-8B 是 36 层（N=36），所以：

```text
block_indices [8, 16, 24]  →  inscope_layers [9, 17, 25]
```

本次拿到的 checkpoint 的 `manifest.json` 直接给出了 `inscope_layers: [9,17,25]`，
与上述换算规则一致 —— 这是权威出处，不是推测。

---

## 3. 本次验证过的 checkpoint

### 3.1 文件

```text
/data/zmosg/checkpoints/best_layers_08_16_24.pt   (593 KB)
/data/zmosg/checkpoints/manifest.json             (1253 B)
```

### 3.2 自述参数

```text
schema                fyh.probe_checkpoint.v2
probe_architecture    linear
input_dim             49152          = 3 层 × 4 位置 × 4096
block_indices         [8, 16, 24]
inscope_layers        [9, 17, 25]     ← 来自 manifest，权威
selected_positions    [-4, -3, -2, -1]
feature_normalization standard        (input_mean / input_std)
flatten_order         [batch, layer, token, hidden]
threshold             0.9
labeling_protocol     risk_faced
feature_name          no_prefix_cot_last4_20261015
model_state_dict      input_mean / input_std / probe.weight / probe.bias
runtime_parity        GPU validation pending   ← 原 manifest 自带
```

### 3.3 训练配置的定位过程（避免踩坑）

这个 checkpoint 的 `feature_name = no_prefix_cot_last4_20261015` 精确对应：

```text
/data/djs/ipi-releases/full-v5-r1/featurize/settings/external_catalog_setting2.json
→ [{"feature_name": "no_prefix_cot_last4_20261015",
    "selected_positions": [-4, -3, -2, -1]}]
```

**没有 `probe_prefix` 字段 —— 训练时没有加任何探测问题前缀。**

这一点极易踩坑，因为环境里同时存在多个相似前缀：

| 来源 | 前缀 |
|---|---|
| `simplified_v3_observation_fivepoint` | `Are there risks of prompt injection or misleading contents? What about your opinions?` |
| `joint_observation_assistant_prefix` | `Are there risks of indirect prompt injection?` |
| **本 checkpoint** | **（无前缀）** |

加前缀会让分数完全反转。实测证据见 §4.2。

---

## 4. 验证结果

### 4.1 单元测试

```text
59 passed  (无需 GPU)
```

| 文件 | 数量 | 覆盖 |
|---|---|---|
| `test_architecture.py` | — | import 单向约束 |
| `test_layers.py` | — | 层号换算 |
| `test_api_cpu.py` | — | 契约常量、多位置解码、请求校验 |
| `test_composer_profile.py` | — | profile 注册与绑定 |
| `test_probe_checkpoint.py` | 15 | checkpoint 自适应加载与各条拒绝路径 |
| `test_client_contract.py` | 10 | 客户端/服务字段契约、profile-probe 绑定 |

`tests/conftest.py` 提供 `write_checkpoint()` 夹具，避免各文件重复拼装 checkpoint。

### 4.2 端到端验证（真实权重 + 真实 GPU）

环境：GPU 7（唯一空闲卡），Qwen3-8B-FP8，SGLang 引擎端口 31100，服务端口 31200。

**正确配置（无前缀）**：

```text
case                    logit      prob  label
attack_explicit       28.1114  1.000000  attack  OK
attack_stealth        29.7582  1.000000  attack  OK
benign_summary        -7.0400  0.000875  benign  OK
benign_tool           -5.0434  0.006411  benign  OK
benign_question      -19.4764  0.000000  benign  OK
accuracy: 5/5
```

服务层输出的 `+28.1114` 与脱离服务独立计算的数值**逐位一致**，
说明 Composer → SGLang → 探针整条链路没有引入偏差。

**错误配置（加了前缀）的对照**：

```text
加 "Are there risks of prompt injection or misleading contents?" 前缀
  attack   logit  -12.00    全部误判为 benign
```

**这是本次最重要的教训：探针的上下文编排必须与其训练配置严格一致，不能凭语义相似去猜。**

### 4.3 已验证的边界

- 位置数量：从 1 个扩展到 4 个，`decode_features` 已按实际位置数校验形状
- 引擎回显绝对位置（如 `[6,7,8,9]`），校验必须用解析后的绝对索引而非原始负值
- 输入不足 4 token 时给出明确 422，而不是让引擎抛错

---

## 5. 代码改动清单

| 文件 | 改动 |
|---|---|
| `inscope_server/probe.py` | **重写**。`load_probe()` 按 checkpoint 自描述构造；`Probe.flatten()` 校验每层形状；`_derive_probe_id()` |
| `inscope_server/backend.py` | `capture()` 增加 `positions` 参数；`decode_features()` 支持多位置、返回层块列表 |
| `inscope_server/app.py` | **重写**。启动期校验 profile/probe 绑定与位置一致性；`/info` 报告探针自述；最小 token 数校验 |
| `inscope_server/config.py` | **重写**。移除与 profile 冲突的 `composer` 副本、`threshold_logit`、`models_dir`；强制要求 checkpoint 与 profile |
| `inscope_server/cli.py` | **重写**。新增 `_preflight()` 与 `--preflight-only` |
| `inscope_server/context_profile.py` | 新增 `a6_nemotron_risk_faced_v1`；A5 profile 标注未支持其 checkpoint 布局 |
| `inscope/client.py` | **重写**。对齐服务字段（`probability`/`label`）；恢复 `flagged` 兼容属性；新增 `AsyncClient` |
| `inscope_contract/constants.py` | 新增 `RISK_TYPE`，把「风险类别」与「探针名」分开 |
| `README.md` | 重写启动参数、层号语义、profile 表、测试说明 |
| `tests/conftest.py` | **新增**，共用 checkpoint 夹具 |
| `tests/test_probe_checkpoint.py` | **新增**，15 个用例 |
| `tests/test_client_contract.py` | **新增**，10 个用例 |
| `tests/test_api_cpu.py` | 适配新的 `decode_features` 返回值；新增多位置测试 |

---

## 6. 运行方式

### 6.1 环境变量

```bash
export INSCOPE_MODEL_PATH=/data/djs/models/Qwen3-8B-FP8
export INSCOPE_API_KEY=<对外鉴权用的 Bearer token>
export INSCOPE_ENGINE_URL=http://127.0.0.1:31100
export INSCOPE_PROBE_CHECKPOINT=/data/zmosg/checkpoints/best_layers_08_16_24.pt
export INSCOPE_COMPOSER_PROFILE=a6_nemotron_risk_faced_v1
```

> `INSCOPE_API_KEY` 是本服务对外的鉴权口令，与 GitHub token 无关。
> `SGLANG_API_KEY` 才是访问引擎用的。

启动前自检（不启动服务）：

```bash
inscope-serve --preflight-only
```

实测能拦住的错误配置：

```text
缺 profile            → INSCOPE_COMPOSER_PROFILE is required
未知 profile          → unknown composer profile 'nope'; available: [...]
profile 与探针不匹配   → composer profile 'a5_simplified_v1' is bound to probe 'a5_blend',
                        but the loaded probe is 'linear:arm6_...'
checkpoint 不存在      → Probe checkpoint not found: /nope.pt
```

### 6.2 启动引擎

```bash
CUDA_VISIBLE_DEVICES=7 SGLANG_ENABLE_FEATURE_CAPTURE=1 \
python -m sglang.launch_server \
  --model-path /data/djs/models/Qwen3-8B-FP8 \
  --port 31100 --host 127.0.0.1 \
  --mem-fraction-static 0.85 --disable-decode-cuda-graph \
  --tp-size 1 --trust-remote-code
```

> `--disable-decode-cuda-graph` 是必需的。generation 模型上不用它，
> 会触发 `NoOpMHATokenToKVPool.set_kv_buffer was called`。

### 6.3 启动服务

```bash
inscope-serve --host 127.0.0.1 --port 31200 --log-level info
```

### 6.4 调用

```bash
curl -s -X POST http://127.0.0.1:31200/detect \
  -H 'Authorization: Bearer <INSCOPE_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"..."}]}'
```

返回：

```json
{"logit": 28.1114, "probability": 1.0, "threshold": 0.9,
 "label": "attack", "input_tokens": 23, "latency_ms": 105.2,
 "composer_profile": "a6_nemotron_risk_faced_v1", ...}
```

---

## 7. 尚未完成 / 风险

### 7.1 未验证

| 项 | 说明 |
|---|---|
| 真实流量下的 FPR/TPR | 只用 5 条人工构造样本验证过，**不是统计意义上的准确率** |
| 与训练侧数值对齐 | manifest 原文写着 `runtime_parity: GPU validation pending` |
| 多轮对话 / 带 tools 的场景 | 只测了单轮纯文本 user 消息 |
| `a5_simplified_v1` profile | 保留但**未用真实 A5 权重验证过**（真实 A5 权重不在本机） |

### 7.2 已知风险

- **阈值 0.9 未复核**。它来自 checkpoint，但对应的 FPR/TPR 在这个部署形态下没有测过。
  不要假定它等价于训练时的阈值语义。
- **GPU 7 是唯一可用卡**。GPU 0-6 被其他用户占满，不要干扰。
- 引擎必须带 `SGLANG_ENABLE_FEATURE_CAPTURE=1`，否则拿不到特征。

### 7.3 未做的清理

`coolwatch` 仓库里仍留着旧的三份包（`coolwatch/inscope/packages/`），
**尚未切换到本仓库**。切换步骤见 §8。

---

## 8. 下一步建议

### 8.1 coolwatch 切换（未开始）

```text
1. coolwatch 依赖改为 pip install -e ../inscope-service
2. 删除 coolwatch/inscope/packages/（先确认无 import 残留）
3. 跑 coolwatch 侧回归
```

已知 `coolwatch` 内的 import 点：

```text
inscope/scripts/verify_contract_migration.py
inscope/scripts/verify_app.py
```

注意：这两个脚本都依赖 `/mnt/kpfs_qyang/...` 外部数据，**本机跑不了**。

### 8.2 真实的 runtime parity 验证（建议优先）

拿到同一批样本在训练侧的 logit 输出，与本服务逐条比对。
这是唯一能给 `runtime_parity: GPU validation pending` 画上句号的方式。

### 8.3 接入 LeakGauge demo 数据

`/data/zmosg/LeakGauge`（已 clone）提供了 demo 数据集与检测范式，
可作为真实样本的来源。注意：**LeakGauge 走的是 logprobs 路线，与本仓库的激活值路线
是两条不同技术路线**，只能借用其数据与评测流程，不能直接复用其模型。

---

## 9. 快速排障

| 现象 | 原因 | 处理 |
|---|---|---|
| `No calibrated threshold for seed` | 旧配置残留 | 已移除，改用 checkpoint 自带阈值 |
| `Shadow feature backend unavailable` | 引擎不可达或特征契约不符 | 查引擎日志与 `/model_info` |
| `position -3 is outside the N-token prompt` | 输入不足 4 token | 加长输入 |
| `Feature contract mismatch` | 传给校验的位置与实际不符 | 已修：必须传解析后的绝对位置 |
| `Unsupported flatten_order` | tuple/list 类型不一致 | 已修 |
| 服务进程启动后消失 | `pkill -f` 误杀了启动脚本 | 用精确 pid 或 `pgrep -f 'inscope-serve --host'` |

---

## 10. 关键文件速查

```text
inscope-service/
├── inscope_contract/
│   ├── constants.py          契约常量（BLOCK_INDICES / CAPTURE_LAYERS 等）
│   ├── layers.py             层号语义与换算（唯一出口）
│   ├── schemas.py            请求/响应模型
│   └── model.py              模型权重校验
├── inscope_server/
│   ├── probe.py              ★ 按 checkpoint 自描述加载探针
│   ├── backend.py            ★ SGLang 抓特征（位置参数化）
│   ├── app.py                FastAPI 应用与三个端点
│   ├── context_profile.py    ★ Composer profile 注册表
│   ├── composer.py           简化 composer 实现
│   ├── config.py             Settings / 环境变量
│   └── cli.py                inscope-serve 入口
├── inscope/                  客户端 SDK
└── tests/                    45 个测试，无需 GPU
```
