# 多接口支持现状与补齐顺序

## 对外接口和内部引擎接口不要混淆

调用方访问同一个 InScope Server 地址上的多个业务路径；不需要每个功能单独启动服务。
当前一个服务实例共享一个已加载探针、一个匹配 profile 和一个后端客户端。
多业务接口不等于同时加载多个探针，也不要求引擎暴露 detect/audit/analyze。

```text
调用方 → 同一个 InScope Server
          ├─ POST /detect   当前上下文评分（单条/批量）
          ├─ POST /audit    轨迹逐步审计
          └─ POST /analyze  指定 observation 内相对归因
                         ↓
                  共享编排与探针评分
                         ↓
                 sglang-inscope /generate
```

## 当前支持表

| 路径 | 当前状态 | SDK |
| --- | --- | --- |
| POST /detect | 已注册；默认接收批量消息并逐条隔离错误，兼容原单条格式 | 同步/异步均有 detect |
| POST /analyze | 已注册；切分、组合评分与相对区域定位；真实引擎批量已验证 | 尚无 analyze 方法，可直接调用 HTTP |
| POST /audit | 已注册；逐检查点回放，输出时间线与首次检出下标 | 同步/异步均有 audit |
| GET /info | 返回当前探针和编排配置；没有完整功能支持列表 | 同步/异步均有 info |
| GET /health/live | 仅进程存活，不需鉴权 | 无专用方法 |
| GET /health/ready | 使用单条采集路径做真实打分；不能证明 analyze 批量就绪 | 同步/异步均有 ready |

业务接口、info 和 ready 需要 Bearer key。接口实现见 [app.py](<../inscope_server/app.py>)，SDK 见 [client.py](<../inscope/client.py>)。

## 核对发现

- audit 已补齐：路由、`AuditRequest` 契约、同步/异步 SDK。检查点默认取 tool/user 观察，只回放截至该步的上下文。
- detect/analyze 手动读取 Request 并做 Pydantic 校验，运行时有校验，但 OpenAPI 未声明 requestBody；200 响应 schema 也是空对象。自动生成客户端或看 Swagger 时，无法获得完整字段说明。
- [schemas.py](<../inscope_contract/schemas.py>) 中 DetectionResult 仍要求 score、threshold_logit、flagged 等旧字段；当前 detect 返回 probability、label 等。该模型未绑定为路由 response_model，不能把它当当前真实响应契约。
- analyze 有 AnalyzeRequest，没有相应的正式响应模型。SDK 尚未覆盖 analyze。
- analyze 的请求级并发隔离与请求体读取的 deadline 边界仍未闭环；HTTP 批量协议已在真实引擎上验证通过。详见 [代码导览](<architecture.md>)。

## 业务语义（已实现部分的边界说明）

### detect：现在有没有达到探针阈值？

输入 messages/tools；保持现有分数、阈值及 label。若演示端需要 exists，可显式约定其与阈值判定的关系，不删除已有 label 或静默换字段。

### audit：首次在哪一步检出？（已实现）

沿轨迹检查点逐步回放，第 i 步只用 `messages[:i+1]`，绝不带入未来消息。
每个检查点独立评分，并发受 `inscope_audit_concurrency` 约束但仍按轨迹顺序报告。

返回：`steps` 时间线、`first_flagged_index`、`flagged_indices`、`failed_at`。

三种"没检出"必须区分：
- `clear`：所有检查点都跑完且都低于阈值；
- `incomplete`：有检查点没跑或失败——**不是阴性**；
- `flagged`：至少一步越过阈值。

`first_flagged_index` 表示"截至该步的累计上下文已越过阈值"，不代表攻击在该步发生，
也不代表该步导致后续行为。响应里的 `semantics` 字段把这个边界写给调用方。

上线前仍需确认：超长轨迹的 token 上限策略（当前受 `INSCOPE_MAX_INPUT_TOKENS` 与 128 条消息约束）、
以及探针 composer 的选窗是否会裁掉早期检查点。

### analyze：所选 observation 中哪些片段相对影响分数？

保持现有 observation_index、粒度、模式、预算；返回不连续 regions、原文偏移、levels 和不确定状态。
不必强制先调用 detect/audit；已有 observation_index 的调用方可以直接分析。
定位是相对归因，不是片段绝对恶意性的判决。

## 推荐补齐顺序

1. 修正模型身份校验失效策略、异常与并发边界；不要通过切到现场 ZMQ 回避 HTTP 契约问题。
2. 统一实际响应与 contract，补 OpenAPI 请求/响应描述；保持流式 body 大小限制和现有客户端兼容。
3. 加固 audit 的超长轨迹策略，复用 ScoringPipeline，不复制另一套探针逻辑。
4. 补同步/异步 SDK 的 analyze/audit 及相应超时说明；可在 info 增加明确的 capabilities，避免调用方猜测功能是否存在。
5. 分别验收三个业务入口。ready 单条打分通过不能替代批量或轨迹审计验收。

本页区分已实现能力与仍待补的边界；未列出的能力不代表已支持。
