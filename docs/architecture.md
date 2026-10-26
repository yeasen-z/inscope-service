# 代码导览与实现边界

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

## 两条调用链

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
      → ScoringPipeline.capture（单条采集，相邻窗口命中引擎前缀缓存）
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
打分流水线暂时保留 FastAPI 错误映射，尚不是与传输完全解耦的领域层。

## 分析究竟如何工作

1. 只修改指定 observation 的 content，其余消息与 tools 保留。
2. 按 paragraph/line/sentence 切出原文区间；偏移是 Python Unicode codepoint，左闭右开。
3. 每组有 k 个单元，就枚举 2^k 种保留/删除组合，不是每次仅删一段。单元外文字和分隔符保留。
4. 同一分析中，相同修改后文本只打分一次。当前缓存键不是最终 token 序列。
5. 用每份文本的 logit 算精确组内 Shapley，并比较相对影响；不是绝对安全/危险判决。
6. exact 枚举全部选定单元；hierarchical 先比较粗组，再细化被选中区域。后续层依赖前层结果，不能一次全部提前提交。
7. 根组超预算报错；子组预算不足时保留粗区域并标记 `budget_limited`，不是一律抛错。

同一轮可批量评分，不同轮有依赖。`cost.score_batches` 是算法调用评分器的轮数，不是 HTTP 次数。
当前评分器不返回 `prompt_index`，所以 `model_requests` 为 null，不能据此宣称已检测最终 token 输入坍缩。
公共前缀有多长由裁剪、模板和删除位置决定，没有固定 90% 的命中保证。

## HTTP 引擎与现场 ZMQ 服务是两回事

本仓库后端使用 `/model_info` 与 `/generate`。djs 的 `featurize.feature_server` 使用 ZMQ 二进制消息；
它可以将多条结果合并回包，但与它对齐不证明本仓库的 HTTP 批量路径正确；ZMQ 合并回包也不必然等于同一次 GPU forward。
本仓库的批量协议是独立对齐并验证的，不依赖该 ZMQ 服务。

### HTTP 批量协议（已验证）

`capture_batch` 发送单个对象，`input_ids` 为二维列表，`feature_capture` 为该批共用的一个 spec，
与引擎 `GenerateReqInput` 对齐。由于 spec 整批共享，负向位置会按各 prompt 自身长度解析成不同的绝对下标，
因此实现按“解析后的位置”分组、逐组并发，最后按输入顺序拼回，避免拿 A 的位置读 B 的序列。

已在真实引擎（sglang-inscope + Qwen3-8B-FP8）上验证：两条等长 prompt 合并为一次请求，
返回 2 份记录、各 3 层 × 4 位置、`bfloat16`、形状 `(4, 4096)`，两份数值不同（未串行）。
协议改动须同时更新 mock 与真实引擎检查，不能只改 mock 让结果变绿。

### 已知限制（仍在）

- **模型身份缓存不随远端重启失效。** `_verified` 缓存至本实例关闭，HTTP 重连不会重新校验身份。
- **分析资源与异常边界不完整。** analyze 尚无请求级 inflight 限制，读取请求体在分析 deadline 外；批量 `BackendError` 尚未像 detect 一样映射到 503。
- **批次配置边界。** `analyze_batch` 的正数检查在评分器构造时，不在 Settings 启动校验时。
- **OpenAPI 未声明请求体。** detect/analyze 手动读取 Request 并做 Pydantic 校验，运行时安全，但 OpenAPI 看不出字段；`DetectionResult` 等旧模型未绑定为 response_model，不能当作当前响应契约。
- **能力边界。** SDK 已提供 detect/audit/info/ready，尚无 analyze 方法。profile 仍须显式选择并与 checkpoint 匹配，不能承诺任意权重自动适配。

### 尚未验收的部分

三个业务接口均在真实引擎上跑通并有一轮性能测量（见 [实测性能](<performance.md>)），
但以下仍未完成：并发下的表现（测量均为单请求串行）、GPU 采集数值与训练侧缓存的一致性、
以及检出率/误报率评估。跑通不等于生产就绪。

## 维护规则

- 变更请求字段：先看 contract，再看规范化、路由与 SDK。
- 变更编排/打分：改 scoring / profile / probe，不在路由复制第二套规则。
- 变更批量运输：改 backend；分桶只决定输入分组，不猜引擎调度行为。
- 定位内核保持上游副本，服务差异放适配层。
- 注释解释不变量和约束，不写未经测量的加速数字；实验记录与实现契约分开。
- CPU 测试保护重构，不替代真实引擎协议与 GPU 验收。
