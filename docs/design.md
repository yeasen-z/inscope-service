# 统一设计：InScope Service 现状与对接契约

本文是**唯一的设计基线**，回答三件事：当前实现到哪一步、上下游各自承担什么、哪些点还没对齐。
[README](../README.md) 面向部署者，回答“怎么装、怎么起、怎么调”；
[api-support.md](api-support.md) 面向调用方，回答“这个接口语义是什么”；
[performance.md](performance.md) 只记录历史测量及其边界；
[architecture.md](architecture.md) 是代码导览。本文负责把它们串成一条线，并明确**需求与对接点**。

凡本文与其它文档冲突，以本文为准，并应回改对应文档。

---

## 1. 系统边界

```text
┌──────────────┐   HTTP/JSON    ┌──────────────────┐   /model_info + /generate   ┌─────────────────┐
│ coolwatch    │ ─────────────► │ inscope-service  │ ──────────────────────────► │ sglang-inscope  │
│ (业务/前端)  │ ◄───────────── │ (本仓库)         │ ◄────────────────────────── │ (推理+抓特征)   │
└──────────────┘                └──────────────────┘                             └─────────────────┘
      ▲                                  ▲
      │ 展示 per_risk / 已禁用          │ 加载 checkpoint 自描述
      │                                  │
   前端三风险行                    探针训练任务与标签定义
```

三层职责不可互相替代：

| 层 | 负责 | 不负责 |
| --- | --- | --- |
| coolwatch | 业务编排、拦截决策、UI 呈现、风控记录 | 特征采集、探针数学 |
| inscope-service | 上下文编排、特征请求、探针打分、协议封装 | 定义风险语义、训练探针 |
| sglang-inscope | 推理与按层/位置返回激活 | 探针与阈值 |

**核心不变量**：服务执行探针，但不替探针定义业务含义。`attack`/`benign`、`indirect_prompt_injection` 是历史命名，不代表服务已支持任意任务语义。

---

## 2. 当前实现状态（0.6.0，工作区未提交）

### 2.1 已具备的能力

| 能力 | 实现位置 | 状态 |
| --- | --- | --- |
| checkpoint 自描述加载（层/位置/归一化/铺平/阈值） | [probe.py](../inscope_server/probe.py) | 已实现 |
| 层号语义换算（block → boundary） | [layers.py](../inscope_contract/layers.py) | 已实现 |
| Composer profile 注册与探针/位置互认 | [context_profile.py](../inscope_server/context_profile.py) | 已实现 |
| 请求级 profile 覆盖（`composer_profile` 字段） | [app.py](../inscope_server/app.py) | 已实现 |
| `no_reasoning` 消融策略 | [composer.py](../inscope_server/composer.py) | 已实现，**未重新标定** |
| `/detect` 单条 + 批量（逐项错误隔离） | [app.py](../inscope_server/app.py) | 已实现 |
| `/audit` 轨迹逐检查点回放 | [audit.py](../inscope_server/audit.py) | 已实现 |
| `/analyze` 相对归因与区间定位 | [localization/](../inscope_server/localization/) | 已实现 |
| 多风险 Probe Bank 输出（`per_risk` 等） | [app.py](../inscope_server/app.py) | **新增，未与 coolwatch 联调** |
| 层并集一次采集（不同 checkpoint 不同层） | [scoring.py](../inscope_server/scoring.py) | **新增，仅单测覆盖** |
| 共享契约 + OpenAPI | [schemas.py](../inscope_contract/schemas.py) | 已实现 |

### 2.2 明确未验收

- 真实流量下的 TPR/FPR：仅 5 条人工样本跑通，**不是准确率**。
- 与训练侧数值对齐：manifest 自带 `runtime_parity: GPU validation pending`。
- 并发与容量：历史测量均为单请求串行。
- 多风险模式的真实引擎验证：目前只有 CPU 单测（合成探针）。
- `no_reasoning` 复用 A6 权重但未重标定，`calibration_status=experimental_not_recalibrated`。

---

## 3. 对接点清单（按方向）

对接点分四类：**A. 服务对外**、**B. 服务对引擎**、**C. 服务对 checkpoint**、**D. 下游对服务**。
每项都给出“契约”“现状”“缺口”。

### A. 服务对外（HTTP/SDK）— 调用方视角

| 接口 | 契约要点 | 现状 | 缺口 |
| --- | --- | --- | --- |
| `POST /detect` | 单条扁平结果；二维 `messages` 为批量 | 已实现，逐项隔离 | 与 coolwatch 联调未做 |
| `POST /audit` | `clear`/`incomplete`/`flagged` 三态 | 已实现 | 响应非强类型模型 |
| `POST /analyze` | `regions`/`levels`/`cost`/不确定状态 | 已实现 | 响应非强类型模型 |
| `GET /info` | 当前探针与 profile | **未反映多风险注册表** | 需补 `risks` 列表 |
| `GET /health/ready` | 真实采集+打分 | 已实现 | 只覆盖主探针链路 |
| `GET /health/live` | 仅进程存活，免鉴权 | 已实现 | — |

错误码约定：`401` 鉴权、`413` 超限、`422` 非法输入、`429` 并发满、`503` 后端/打分失败、`504` 超时。
**不允许把失败包装成 HTTP 200 的阴性。**

### B. 服务对引擎（sglang-inscope）

| 项 | 契约 | 现状 |
| --- | --- | --- |
| 身份核验 | 每次 `/generate` 前 GET `/model_info` | 已实现，**非原子**（存在检查/采集间切换竞态） |
| 单条请求 | 单个 `GenerateReqInput` 对象 | 已对齐 |
| 批量请求 | `input_ids` 二维 + 共享 `feature_capture` | 已对齐 |
| 位置分组 | 负向位置按各 prompt 长度解析成绝对下标，按组发送 | 已实现 |
| 特征校验 | schema/层/位置/形状/编码/有限值逐个校验 | 已实现，任一不符即 503 |

**关键约束**：`feature_capture` spec 整批共享，因此位置不同必须分组；分组不等于 GPU 同次 forward，不要从请求数推断算力。

### C. 服务对 checkpoint

| 项 | 来源 | 校验行为 |
| --- | --- | --- |
| 层号 | `manifest.inscope_layers` 优先，否则按 `layer_semantics` 换算 | 越界/重复/非整数拒绝 |
| token 位置 | `selected_positions` | 与 profile 不一致 → 启动失败 |
| 归一化 | `feature_normalization` + `input_mean/std` | 未支持的方式拒绝 |
| 铺平顺序 | `flatten_order` | 逐层校验 `[n_pos, hidden]` |
| 阈值 | checkpoint | 必须有限且在 (0,1] |
| 权重完整性 | `manifest.sha256` | 不匹配拒绝加载 |

**多风险扩展**：`INSCOPE_PROBES=risk=/path[,risk=/path]` 为每个风险加载独立 checkpoint，**层可以不同**；启动期要求所有探针的位置声明与 profile 一致。

### D. 下游对服务（coolwatch）— 本次对齐重点

coolwatch 侧真实代码事实（不是推测）：

| 位置 | 行为 | 对 inscope-service 的硬要求 |
| --- | --- | --- |
| `backend/watchers/probe_bank.py:44-64` | `ProbeBankGuard.moderate()` 逐项校验 | 见下表 |
| `backend/app.py:399-427` | 组装对外响应 | 读顶层 `input_tokens`/`per_risk`/`per_entry`/`triggered_entries`/`bank_version`；`latency_ms` 必须是 mapping |
| `backend/probe_risks.py` | 风险代号 | 只接受 `harmful`/`prompt_leakage`/`ipi` |
| `frontend/src/app.js:79` | 三行固定风险 | `harmful`/`prompt_leakage`/`ipi` |
| `frontend/src/app.js:2890-2905` | 状态渲染 | `disabled_risks` → “已禁用”；缺 verdict → “未检测” |

`ProbeBankGuard` 的硬校验逐条对应：

| 校验 | 我们的字段 | 结论 |
| --- | --- | --- |
| `detector_model` 非空 | `detector_model`（`INSCOPE_DETECTOR_MODEL`，默认 `Qwen3-8B-FP8`） | **不需要配对**：由服务端自述，coolwatch 只透传展示 |
| `per_risk` 覆盖调用方启用的风险 | `per_risk` / `enabled_risks` | 交集语义，见 §4.2 |
| 返回的 `enabled_risks` 包含调用方所需 | 同上 | 服务多返回不算错 |
| `flagged` 是 `bool` | `per_risk[r].flagged` | 已满足 |
| `score`/`threshold` 有限且在 [0,1] | 同名字段 | 已满足 |
| `bank_version` 非空 | `bank_version` | 已满足 |
| `input_tokens > 0` | 顶层 `input_tokens` | 已满足 |
| `latency_ms` 可解包 | 多风险模式为 `{"total": ms}` | 已满足；**单风险模式仍是数字** |
| 失败时读 `status_code` | 400/413/429/504 原样透传，其余归 503 | 已满足（503/504 语义吻合） |

**仍需注意**：coolwatch 的 `moderate()` 原先使用 `httpx.AsyncClient` 但**不发送 `Authorization` 头**，
而本服务除 `/health/live` 外全部要求 Bearer，联调必然 401。该问题已于本轮修复，见 §5 P0-1。

---

## 4. 多风险（Probe Bank）设计

### 4.1 数据流

```text
请求 messages
  → Composer 编排（单一 profile，所有探针共享）
  → capture_plan(): 层并集 ∪ 位置并集
  → backend.capture / capture_batch（一次 /generate，按绝对位置分组）
  → {layer: [n_pos, hidden]}
  → 各探针按自己的 capture_layers 切片、flatten、打分
  → per_risk / per_entry / triggered_entries / overall
```

### 4.2 关键决策与理由

| 决策 | 理由 |
| --- | --- |
| 层取并集、一次采集 | 避免每加一个风险就多一次前向；不同 checkpoint 层不同也能共享 |
| 位置取并集 | 后端按绝对位置分组采集，各组仍并发；单探针时并集退化为自身 |
| 未加载的风险进 `disabled_risks` | **绝不用别的 checkpoint 编造分数**；前端会显示“已禁用” |
| 阈值不共享 | 每个 checkpoint 自带阈值，混合会让判定失真 |
| profile 不参与选风险 | 编排策略与风险选择是两个正交维度，避免语义耦合 |
| batch 也输出 bank 字段 | 与单条保持同一套字段，减少调用方分支 |

### 4.3 已识别的设计债

1. **`/detect` 单条路径语义变化**：多风险模式改走 `pipeline.collect()`（`capture`），批量走 `score_prompts`（`capture_batch`）。两条路径共享 `score_probes_all`，但**入口不同**，测试需要同时注入两者。
2. **`latency_ms` 双形态**：单风险是数字、多风险是 mapping。SDK 已兼容，但这仍是调用方最容易踩的坑；`docs/api-support.md` 已提示。
3. **`profile` 字段硬编码为 `"balanced"`**：coolwatch 不校验该值，但它不反映任何真实策略，属于占位。
4. **多风险模式下 `risk`/`label` 仍只描述主探针**：顶层字段语义 = 主探针，`per_risk` 才是全景。
5. **`/audit`、`/analyze` 不支持多风险**：目前仅 `/detect` 输出 bank 视图。这是有意的范围裁剪，但需要显式声明。

---

## 5. 当前需求（按优先级）

### P0 — 联调阻塞项

1. ~~**鉴权对齐**~~ —— **已完成**。采用方案 (a)：coolwatch `ProbeBankGuard` 新增 `PROBE_BANK_API_KEY`，
   对 `/detect` 与 `/info` 均发送 `Authorization: Bearer <key>`；未配置时**失败关闭**
   （返回 “PROBE_BANK_API_KEY is not configured”），不会静默发出匿名请求。服务端鉴权未放宽。
   同时修正 `get_model_info()`：原请求 `/bank`（本服务不存在），改为 `/info`。
2. **`detector_model` 解耦**（已完成）。原本要求 `detector_model` 等于 coolwatch 的
   `CUSTOMER_AGENT_SHADOW_MODEL`，这是错的：那是 coolwatch 调推理服务时用的模型名，
   属于客户自己的部署，与探针服务加载哪份权重无关。现在改为只校验**非空**，
   `detector_model` 由服务端自述并原样透传（coolwatch 的 `/detect` 响应也改用服务端回报值）。
   因此 **coolwatch 对接本服务只需要 URL 和 key，不需要知道任何模型名**。
3. **风险集合**（已改为交集语义）。服务与调用方各自独立配置，互不要求相等：
   - 服务端 `INSCOPE_PROBES` 决定**能**提供哪些风险；
   - coolwatch `PROBE_BANK_ENABLED_RISKS` 决定**要**哪些，适配器在 `/detect` 请求里带 `enabled_risks`；
   - 服务返回该子集，其余进 `disabled_risks`；适配器再按自己的启用集裁剪一次，保证下游 UI 只看到自己关心的风险；
   - 请求了服务**未加载**的风险 → 服务 `422` → 适配器报错（失败关闭，不当作阴性）。

   这样服务可以多加载（例如提前接好 harmful 探针），coolwatch 可以按需逐个开启，无需两边同时改配置。

### P1 — 功能缺口

4. `/info` 增加 `risks`（已加载风险 → 层/阈值/sha），便于 coolwatch `/api/guards/activation_probe/info` 展示。
5. `bank_version` 语义：当前是全组 SHA 前缀。若 coolwatch 需要“每次重启变化”，需增加启动序号。
6. batch 的 `per_risk`：现在只取第一行，需确认业务是否要求逐行视图。

### P2 — 待定

7. `/audit`、`/analyze` 是否也需要 `per_risk`。
8. 多风险模式的真实引擎验证（需要 3 份真实 checkpoint，目前只有 IPI 一份）。

---

## 6. 部署参数对照

| 侧 | 变量 | 值要求 |
| --- | --- | --- |
| coolwatch | `ACTIVATION_PROBE_BACKEND` | 设为 `probe_bank` 才走本服务 |
| coolwatch | `PROBE_BANK_BASE_URL` | 默认 `http://127.0.0.1:8302`，应指向本服务 |
| coolwatch | `PROBE_BANK_API_KEY` | **必填**，必须等于服务端 `INSCOPE_API_KEY` |
| coolwatch | `PROBE_BANK_ENABLED_RISKS` | 要哪些风险；服务只需**包含**这些，可以更多 |
| coolwatch | `CUSTOMER_AGENT_SHADOW_MODEL` | **与本服务无关**，只是它调推理服务用的模型名 |
| inscope | `INSCOPE_API_KEY` | 与 `PROBE_BANK_API_KEY` 相同 |
| inscope | `INSCOPE_DETECTOR_MODEL` | 与 `CUSTOMER_AGENT_SHADOW_MODEL` 相同 |
| inscope | `INSCOPE_PROBES` | `risk=/path`，风险代号限三个；可多于调用方所需 |
| inscope | `INSCOPE_COMPOSER_PROFILE` | 必须与探针 `probe_id`/位置匹配 |

### 已验证的联调证据

`coolwatch/evaluation/tests/test_probe_bank_e2e.py` 用**真实的** `create_app` 与**真实的**
`ProbeBankGuard` 对跑（httpx ASGITransport，无需 GPU，仅 stub tokenizer/probe/backend），覆盖四点：

1. 无 `Authorization` → 401（证明鉴权确实生效）；
2. 带正确 Bearer → `ProbeBankGuard` 全项校验通过，`risky` 判定正确，`latency_ms` 可解包，`/info` 可用；
3. 服务多加载 `harmful`、coolwatch 只要 `ipi` → 取交集，返回 `enabled_risks=["ipi"]`，`disabled_risks` 含 `harmful`；
4. 请求服务未加载的风险 → 失败关闭，`BankAssessment.error` 非空。

该测试证明**线上契约、鉴权与风险选择链路**正确，不证明检出率、阈值语义或 GPU 数值一致性。

---

## 7. 文档分工（避免再次漂移）

| 文档 | 只回答 | 不该出现 |
| --- | --- | --- |
| [README](../README.md) | 安装、启动、配置、快速验证 | 未测量的性能数字、设计论证 |
| 本文 | 现状、边界、对接点、需求 | 操作步骤细节 |
| [api-support.md](api-support.md) | 接口语义与错误处理 | 内部实现细节 |
| [architecture.md](architecture.md) | 代码在哪、调用链 | 需求与规划 |
| [performance.md](performance.md) | 历史测量与不可推广的理由 | 推断性结论 |
| [HANDOFF.md](../HANDOFF.md) | 一次性交接记录 | 与现状不符的**当前**描述 |

`HANDOFF.md` 描述的是 2026-10-23 的交接快照，其中“59 passed”“未切换到本仓库”等已经过时；
它以历史记录身份保留，当前事实以本文为准。

---

## 8. 变更规则

- 改请求/响应字段：先改 [schemas.py](../inscope_contract/schemas.py)，再改路由、SDK、本文对接点表。
- 改编排或打分：改 [scoring.py](../inscope_server/scoring.py) / [context_profile.py](../inscope_server/context_profile.py)，不在路由复制。
- 改批量运输：改 [backend.py](../inscope_server/backend.py)，mock 与真实引擎检查必须同时更新。
- 新增风险：同时更新 `KNOWN_RISKS`（服务端）、coolwatch `RISK_NAMES`、前端 `PROBE_RISK_LABELS`，三处必须一致。
- 不写未经测量的性能结论；不把“跑通”写成“验收”。
