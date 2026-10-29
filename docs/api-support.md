# 多接口支持与契约边界

> 现状、上下游对接点与待办需求见 [统一设计文档](design.md)。

## 对外接口和内部引擎接口不要混淆

调用方访问同一个 InScope Server 地址上的多个业务路径；不需要每个功能单独启动服务。
一个实例共享一个已加载探针、一个匹配 profile 和一个后端客户端。
多业务接口不等于同时加载多个探针，也不要求引擎暴露 detect/audit/analyze。

```text
调用方 → 同一个 InScope Server
          ├─ POST /detect   当前上下文评分（单条/批量）
          ├─ POST /audit    轨迹逐步审计
          └─ POST /analyze  指定 observation 内相对归因
                         ↓
                  共享编排与探针评分
                         ↓
              sglang-inscope /model_info + /generate
```

## 支持表

| 路径 | 行为 | 同步 / 异步 SDK |
| --- | --- | --- |
| POST /detect | 兼容单条消息列表；二维 messages 为批量，逐项隔离错误 | `detect` / `detect_batch` |
| POST /analyze | 切分、组合评分与相对区域定位 | `analyze` |
| POST /audit | 串行逐检查点回放，输出时间线与首次检出下标 | `audit` |
| GET /info | 返回当前探针和编排配置 | `info` |
| GET /health/live | 仅进程存活，不需鉴权 | 无专用方法 |
| GET /health/ready | 使用单条采集路径做真实打分；不能证明所有批量与审计场景就绪 | `ready` |

业务接口、info 和 ready 需要 Bearer key。实现见 [app.py](<../inscope_server/app.py>)，SDK 见 [client.py](<../inscope/client.py>)。

## 共享契约与错误语义

[schemas.py](<../inscope_contract/schemas.py>) 是请求与检测响应的共同来源：

- `DetectionResult` 表示成功结果：有限的 logit/probability/threshold、合法标签、token 数与追踪元数据。probability 和 threshold 在 [0,1]，label 必须与阈值比较一致。
- `DetectionBatchResult` 表示有序批量结果，包含 `schema="inscope.detect_batch.v1"`、`count`、`results` 和请求元数据；`count` 必须等于结果条数。
- `results` 每项是 `DetectionResult` 或 `DetectionItemError`；错误项有 `error`、`detail`、`status_code`，不是带零分的成功项。
- 单条失败使用 HTTP `413`（输入超限）、`422`（输入不合法）、`503`（后端/打分失败）、`504`（超时）。鉴权失败为 `401`，共享请求并发额度耗尽为 `429`。
- 批量在整批请求结构与上限校验后，对各会话分别规范化和校验；某一项无效不阻止其他项评分。已完成的批量响应为 HTTP 200，逐项检查 `status_code`，不能将失败项当作 benign。整批 body 超限、条数超限、非法外层结构、鉴权、准入或整体 deadline 失败仍可返回非 200。

OpenAPI 声明 detect 的单条/批量 requestBody 和 `DetectionResponse` 联合响应；audit/analyze requestBody 引用共享模型。audit/analyze 的完整响应模型仍在演进，不能据此承诺已能生成全部强类型客户端。

两种 SDK 均提供 detect/detect_batch/analyze/audit/info/ready。`detect` 返回兼容型 `DetectResult`（继承共享模型并保留旧服务元数据默认值），`detect_batch` 返回共享 `DetectionBatchResult`；`flagged` 是结果上的便利属性，不是独立 wire 字段。analyze/audit/info 返回 JSON 对象，ready 返回布尔值。传输、非 200 和畸形成功响应不会被当作检测阴性。analyze/audit 分别有 `analyze_timeout`（默认 900 秒）和 `audit_timeout`（默认 1800 秒），也可逐调用传入 `timeout`；这是客户端 HTTP 超时，不替代服务端总 deadline。

## 资源、超时与身份检查

- detect（含整批）、audit、analyze 和 ready 共享 `INSCOPE_MAX_INFLIGHT` 请求额度，每个外部请求占一份；audit 的内部检查点不会重复占用请求槽。
- `INSCOPE_BACKEND_MAX_INFLIGHT` 另行限制真实引擎 HTTP 在途请求，不等于请求数、桶大小或 GPU batch 大小。
- body 流式大小限制与读取均在业务请求 deadline 内；detect/ready 使用 request timeout，audit/analyze 使用各自总 timeout。引擎单次请求仍受 request timeout 限制。
- `INSCOPE_MAX_BATCH_ITEMS=128` 是 detect 整批条数上限，`INSCOPE_DETECT_BATCH=16` 是每桶上限；analyze 每桶默认 8。桶串行发送，不承诺引擎 GPU 同次 forward。
- 每次真实引擎 `/generate` 前读取 `/model_info` 核验模型身份，不使用永久 `_verified` 缓存。但检查与采集是两个请求，存在远端切换的竞态，不保证原子性。

配置明细见 [README](<../README.md#配置参考>)，实现边界见 [代码导览](<architecture.md>)。

## 业务语义

### detect：现在有没有达到探针阈值？

输入 messages/tools，返回分数、阈值及 label。若业务需要 exists，可显式约定与阈值判定的关系，不静默换字段。`attack` / `benign` 和固定 risk 是当前接口的历史命名，并不让任意探针自动具有相同业务语义。

对于 `fyh.probe_checkpoint.v3` observation-block attention 探针，调用方可传 `"explain": true`。服务在**同一次**特征采集和 attention 打分中追加 `blocks`、`compiled_context`、`capture_positions`、`coordinate_system: "compiled_messages"` 与 `importance_semantics`。`blocks[].attention_weight` 与 `blocks[].attention_contribution` 是当前已编译上下文中的 attention 证据；固定语义文本为 `cached cumulative-prefix evidence, not IPI probabilities`，不得将它们当作 block 独立的 IPI 概率或跨请求可比较的风险分数。`message_index` 属于编译后消息坐标；要定位回调用请求，必须使用 `source_message_indices`。空 anchor 也在 `blocks` 内并参与 attention，不能过滤后再解释归一化权重。

配置 `INSCOPE_PROBES` 后，同一次特征采集会为所有已加载探针打分，响应追加 `per_risk`/`per_entry`/`triggered_entries`/`enabled_risks`/`disabled_risks`/`overall`/`profile`/`detector_model`/`bank_version`，且 `latency_ms` 变为含 `total` 的映射。顶层 `probability`/`logit`/`label` 仍来自主探针，语义不变。未注册的风险只出现在 `disabled_risks`，不会得到伪造分数。该视图目前只作用于 `/detect`，`/audit`、`/analyze` 仍按主探针工作；observation-block 探针不能与多风险 Probe Bank 同实例加载。

### audit：首次在哪一步检出？

第 i 个检查点只用 `messages[:i+1]`，绝不带入未来消息；检查点串行评分，按轨迹顺序返回 `steps`、`first_flagged_index`、`flagged_indices`、`failed_at`。没有 audit concurrency 配置。

三种状态必须区分：
- `clear`：所有检查点都跑完且都低于阈值；
- `incomplete`：有检查点没跑或失败，**不是阴性**；
- `flagged`：完整评分后至少一步越过阈值。若后续检查点失败，即使已有检出，整体仍为 incomplete，已检出下标仍保留。

首次检出表示“截至该步的累计上下文已越过阈值”，不代表攻击在该步发生或导致后续行为。实际模型输入仍遵守所选 profile：A6 保留截至该步的全部历史与 CoT；A5 简化规则会选窗。128 条消息约束、token 上限和 profile 规则须共同检查，超限不静默截断。

### analyze：所选 observation 中哪些片段相对影响分数？

保留 observation_index、粒度、模式、预算；返回可能不连续的 regions、原文偏移、levels 和不确定状态。无需先调用 detect/audit。定位是相对归因，不是片段绝对恶意性的判决。

对 observation-block attention 探针，`/analyze` 保持原有删/留片段的反事实机制，兼容调用但不是该探针的首选解释路径：删段会改变 block 数量和 attention softmax 的候选集合；删空时只余 empty anchor，形成训练未覆盖的基线。因此可将结果用作片段的**相对排序**，不可将 phi 的绝对值解释成可比较的风险或因果贡献。需要块级解释时，优先使用 `/detect` 的 `explain: true`，它不需要额外前向。

单单元 auto 使用可处理该输入的 exact 路径；数值并列不能武断选出唯一高/低区间，非有限探针分数必须报错，不能伪装成未检出。真实引擎批量协议已跑通，但历史性能快照不是当前身份校验、并发或准确性验收，见 [实测性能](<performance.md>)。
