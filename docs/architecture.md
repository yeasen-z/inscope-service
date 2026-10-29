# 代码导览与实现边界

> 现状、上下游对接点与待办需求见 [统一设计文档](design.md)；本文只回答“代码在哪、调用链怎么走”。

## 从哪里开始读

三个顶层包保持原样，不为了目录整齐破坏 SDK 导入路径：

| 区域 | 职责 | 入口 |
| --- | --- | --- |
| `inscope` | 轻量 HTTP 客户端，不加载模型 | [client.py](<../inscope/client.py>) |
| `inscope_contract` | 请求/响应、默认值、模型和层号契约 | [schemas.py](<../inscope_contract/schemas.py>)、[layers.py](<../inscope_contract/layers.py>) |
| `inscope_server` | 生命周期、编排、采集、评分与定位 | [app.py](<../inscope_server/app.py>) |

服务端按职责阅读：

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| [app.py](<../inscope_server/app.py>) | 启停依赖、鉴权、请求校验、路由和响应封装 | 探针数学、分桶算法 |
| [scoring.py](<../inscope_server/scoring.py>) | `ScoringPipeline`：编排、长度检查、采集与共享评分 | 定位区间选择、传输序列化 |
| [batching.py](<../inscope_server/batching.py>) | 字符长度粗分桶、批次上限、恢复原输入顺序 | tokenizer、HTTP、GPU 调度 |
| [analyze.py](<../inscope_server/analyze.py>) | observation 下标校验、定位入口与旧导入兼容 | 重写上游定位算法 |
| [localization/api.py](<../inscope_server/localization/api.py>) | 组合枚举、去重、层级细化、结果组织 | 探针加载、网络请求 |
| [localization/attribution.py](<../inscope_server/localization/attribution.py>) | Shapley 和交互项计算 | 输入编排 |
| [localization/localization.py](<../inscope_server/localization/localization.py>) | 相对贡献比较与区间选择 | 绝对风险阈值 |
| [backend.py](<../inscope_server/backend.py>) | HTTP 引擎交互、特征响应校验 | ZMQ 客户端、定位算法 |
| [probe.py](<../inscope_server/probe.py>) | checkpoint 解析、特征铺平与评分 | 聊天模板 |
| [context_profile.py](<../inscope_server/context_profile.py>) | profile 注册及探针绑定 | 自动猜训练规则 |
| [composer.py](<../inscope_server/composer.py>)、[rendering.py](<../inscope_server/rendering.py>) | 上下文选窗与模型模板渲染 | HTTP 路由 |
| [normalization.py](<../inscope_server/normalization.py>) | 保守的请求形状规范化 | 猜测缺失历史和 tool-call ID |
| [config.py](<../inscope_server/config.py>)、[cli.py](<../inscope_server/cli.py>) | 部署配置、预检和启动 | 训练 |

## 共享入口与三条调用链

所有业务请求先通过共享请求级 `max_inflight` 准入，detect（含整批）、audit、analyze 和 ready 使用同一个额度。body 的流式大小限制、读取、校验和业务处理位于对应请求 deadline 内；审计/分析使用独立总超时，真实引擎单次请求仍受 request timeout 约束。另一个 `backend_max_inflight` 限制实际引擎 HTTP 在途数，不能与外部请求数、每桶大小或 GPU 并发混同。

单条 detect 错误返回 HTTP 413/422/503/504。批量先校验外层和 `max_batch_items`，再逐会话 normalize + 校验，保留原顺序并隔离失败；失败项包含 `status_code`，不是零分阴性。成功响应使用共享 `DetectionResult` / `DetectionBatchResult`，OpenAPI 的 `DetectionResponse` 为二者联合；audit/analyze 仅共享请求模型，完整响应模型仍在演进。

```text
/detect
  → 规范化与校验
  → ScoringPipeline.capture
      → prepare（Composer → input_ids）
      → backend.capture → score_features
  → 分数、阈值、标签

/audit
  → 规范化与校验
  → 按检查点串行回放 messages[:i+1]
      → ScoringPipeline.capture（单条采集，实际公共 token 前缀可能复用缓存）
  → 时间线、首次检出下标与 semantics

/analyze
  → 规范化与校验
  → analyze → localization.analyze_observation
      → 一轮去重后的反事实文本
      → batching.make_batched_scorer
          → ScoringPipeline.score_prompts
              → prepare → backend.capture_batch → score_features
      → 相对归因、区域选择，必要时继续下一轮
  → regions、levels、cost 与不确定状态
```

三个接口共享编排和 `score_features`，但采集分别走单条/批量方法。
不要说 analyze 调用了 detect 路由，也不要说它们都经过 `score_ids`。
`/detect` 单条在多风险模式下改走 `ScoringPipeline.collect`（内部 `backend.capture`），批量仍走 `score_prompts`（`backend.capture_batch`），两者共享 `score_probes_all`。

`observation_blocks` v3 是一个例外分支：`prepare` 使用 `compile_observation_blocks` 生成每个请求自己的采集位置与 block plan；单条与 readiness 经 `capture`/`score_ids`，批量和 analyze 经 `score_prompts`。不同请求的位置集合不同，无法共享批量 `feature_capture` spec，所以该分支逐条 capture；`explain: true` 在已有 attention 结果上附加 blocks，不再发起额外前向。block 的 `message_index` 属于编译后坐标，`source_message_indices` 才是回指原请求消息的映射。

打分流水线暂时保留 FastAPI 错误映射，尚不是与传输完全解耦的领域层。

## 分析究竟如何工作

1. 只修改指定 observation 的 content，其余消息与 tools 保留。
2. 按 paragraph/line/sentence 切出原文区间；偏移是 Python Unicode codepoint，左闭右开。
3. 每组有 k 个单元，就枚举 2^k 种保留/删除组合，不是每次仅删一段。单元外文字和分隔符保留。
4. 同一分析中，相同修改后文本只打分一次。当前缓存键不是最终 token 序列。
5. 用每份文本的 logit 算精确组内 Shapley，并比较相对影响；不是绝对安全/危险判决。
6. exact 枚举全部选定单元；hierarchical 先比较粗组，再细化被选中区域。后续层依赖前层结果，不能一次全部提前提交。
7. 根组超预算报错；子组预算不足时保留粗区域并标记 `budget_limited`，不是一律抛错。
8. 单单元 auto 走 exact；数值并列用容差判断，不强行选出唯一高/低区域。探针归一化参数、权重和输出的非有限数值不能静默进入定位或变成“未检出”。

同一轮可批量评分，不同轮有依赖。`cost.score_batches` 是算法调用评分器的轮数，不是 HTTP 次数。
当前评分器不返回 `prompt_index`，所以 `model_requests` 为 null，不能据此宣称已检测最终 token 输入坍缩。
公共前缀有多长由裁剪、模板和删除位置决定，没有固定 90% 的命中保证。

对于 observation-block v3，保留/删除一个片段不仅改变文本，也改变动态 block 计划和 attention softmax 的候选项；全部删空后只留下 empty anchor。因此该分支的 `/analyze` 结果只可用于片段**相对排序**，不能把 phi 绝对值解释为可比较的风险分数或因果贡献。要展示块级证据，使用 `/detect` 的 `explain: true`；其 `attention_weight` / `attention_contribution` 是同一次打分的累积前缀证据，而非 IPI 概率。

## HTTP 引擎与现场 ZMQ 服务是两回事

本仓库后端使用 `/model_info` 与 `/generate`。djs 的 `featurize.feature_server` 使用 ZMQ 二进制消息；
它可以将多条结果合并回包，但与它对齐不证明本仓库的 HTTP 批量路径正确；ZMQ 合并回包也不必然等于同一次 GPU forward。
本仓库的批量协议是独立对齐并验证的，不依赖该 ZMQ 服务。

### HTTP 批量协议（已验证）

`capture_batch` 发送单个对象，`input_ids` 为二维列表，`feature_capture` 为该批共用的一个 spec，
与引擎 `GenerateReqInput` 对齐。由于 spec 整批共享，负向位置会按各 prompt 自身长度解析成不同的绝对下标，
因此实现按“解析后的位置”分组、串行发送各组，最后按输入顺序拼回，避免拿 A 的位置读 B 的序列。评分层的长度桶也串行执行；detect 每桶默认 16 条、整批最多默认 128 条，analyze 每桶默认 8 条，这些都不是 GPU batch 大小。

已在真实引擎（sglang-inscope + Qwen3-8B-FP8）上验证：两条等长 prompt 合并为一次请求，
返回 2 份记录、各 3 层 × 4 位置、`bfloat16`、形状 `(4, 4096)`，两份数值不同。该结果验证 HTTP 批量收发与记录区分，不证明请求在 GPU 上同次 forward，也不证明不存在引擎内部串行。
协议改动须同时更新 mock 与真实引擎检查，不能只改 mock 让结果变绿。

### 身份检查、配置与仍存边界

- **身份检查不是原子操作。** 每个真实 `/generate` 前 GET `/model_info` 核验，不再永久缓存 `_verified`；但两个请求间仍可能发生远端模型切换，不能承诺检查/采集原子性。该额外往返的成本不在历史性能快照中。
- **请求与后端额度分开。** 所有业务入口及 ready 共享请求并发限制；真实后端 HTTP 另有全局额度。analyze 批量后端失败映射 503，deadline 超时映射 504，不转为阴性。
- **配置启动校验。** `analyze_batch`、`detect_batch`、`max_batch_items`、`backend_max_inflight` 必须为正整数；默认分别为 8、16、128、16。默认 body 上限为 2097152 字节。
- **契约覆盖并非全部强类型。** detect 使用严谨共享结果模型并声明 OpenAPI；audit/analyze 共享请求模型但完整响应模型仍在演进。同步/异步 SDK 均有 detect/detect_batch/analyze/audit/info/ready。
- **profile 规则仍须显式匹配。** A6 保留截至当前检查点的全部历史与 CoT；A5 简化规则会选窗。不能把所有 profile 说成最近 N 个动作块，也不能承诺任意权重自动适配。

### 尚未验收的部分

三个业务接口均在真实引擎上跑通并有一轮性能测量（见 [实测性能](<performance.md>)）。observation-block v3 另完成了一次真实单条 prefill 的训练侧对拍：编译文本、采集位置、block 元数据相同，logit 与 attention explain 通道的最大差为 `0`。但以下仍未完成：并发下的表现（测量均为单请求串行）、v3 的多请求 batch/radix-cache/chunked-prefill 组合、v3 下 audit/analyze 的逐位训练侧对拍，以及独立的检出率/误报率复现。跑通不等于生产就绪。

## 维护规则

- 变更请求字段：先看 contract，再看规范化、路由与 SDK。
- 变更编排/打分：改 scoring / profile / probe，不在路由复制第二套规则。
- 变更批量运输：改 backend；分桶只决定输入分组，不猜引擎调度行为。
- 定位内核保持上游副本，服务差异放适配层。
- 注释解释不变量和约束，不写未经测量的加速数字；实验记录与实现契约分开。
- CPU 测试保护重构，不替代真实引擎协议与 GPU 验收。
