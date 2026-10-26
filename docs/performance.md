# 实测性能：一次真实环境测量

本文记录在 **301-A06（8×H200）** 上对 `inscope-service` 的端到端测量。
所有数字来自真实引擎与本机服务，不是模拟。

## 环境

| 项 | 值 |
|---|---|
| 机器 | 301-A06，8×H200 |
| 引擎 | sglang-inscope `8a2998b`，Qwen3-8B-FP8，单卡（GPU 6） |
| 引擎参数 | `--mem-fraction-static 0.75 --chunked-prefill-size 32768 --cuda-graph-bs-prefill 8192 16384 32768` |
| 探针 | `best_layers_08_16_24.pt`，层 9/17/25，位置 -4..-1，threshold 0.9 |
| profile | `a6_nemotron_risk_faced_v1` |
| 网络 | 同机回环（服务与引擎均在本机） |

## 一次 `/detect` 的开销拆解

| 环节 | 耗时 | 说明 |
|---|---|---|
| tokenize + chat template | 0.14 ms | 可忽略 |
| 引擎 `/generate` 往返 | **88 ms** | 固定开销，见下 |
| 特征采集（3 层 × 4 位置） | +24 ms | 与采集规模相关 |
| 服务层（鉴权/JSON/编排） | ~0 ms | 连接复用后基本消失 |
| **合计（复用连接）** | **77 ms** | 部分被批处理摊销 |

### 关键发现：88ms 与输入长度无关

| 输入 | 只生成（不采集） |
|---|---|
| 16 token | 92.88 ms |
| 200 token | 91.47 ms |

16 个 token 和 200 个 token 的引擎耗时几乎一样，说明这段时间主要花在调度与往返，不是前向计算。

> 推断：这是引擎侧固定调度开销，服务层无法消除。要降低只能改引擎调度策略。

## 连接复用的收益（真实 SDK 实测）

| 方式 | median |
|---|---|
| 每次新建连接（改造前 SDK 行为） | 146–126 ms |
| **复用连接（当前 SDK）** | **77.39 ms** |

降幅约 **47%**。实现方式：
- SDK `Client` 内部持有单个 `httpx.Client`，`close()` / `with` 释放
- 服务端 `timeout_keep_alive` 从默认 5 秒放宽到 **75 秒**（`INSCOPE_KEEPALIVE` 可调）

## 各接口实测

### `/detect`

| 场景 | median | min | max |
|---|---|---|---|
| 短（约 30 token） | 77.4 ms | 76.6 | 78.0 |
| 批量 4 条 | 239.2 ms 总计 | — | — |
| 批量 4 条（每条均摊） | **59.8 ms** | — | — |

批量比单条更快，说明批处理生效。

### `/audit`（串行推进，命中前缀缓存）

| 场景 | median | 检查点数 |
|---|---|---|
| 3 个工具返回 | 434.9 ms | 4 |
| 6 个工具返回 | 715.4 ms | 7 |

边际成本约 **93 ms/检查点**。串行推进让相邻窗口构成前缀链，引擎的 radix cache 复用已算过的前缀：

| 窗口 | 总 token | cached | 实际计算 |
|---|---|---|---|
| 1 | 33 | 0 | 33 |
| 2 | 53 | 33 | 20 |
| 3 | 73 | 53 | 20 |
| 4 | 93 | 73 | 20 |

**每步只算新增的 20 个 token**，这就是逐条回放并不慢的原因。

### `/analyze`（自适应模式）

| 段数 | 模式 | 组数 | 组合数 | 层数 | median | 状态 |
|---|---|---|---|---|---|---|
| 3 | exact | 3 | 8 | 1 | 677 ms | localized |
| 6 | exact | 6 | 64 | 1 | 3004 ms | localized |
| 10 | hierarchical | 6 | 76 | 4 | 3357 ms | localized |
| 16 | hierarchical | 6 | 100 | 6 | 4111 ms | localized |

## 模式自适应规则

exact 需要枚举 2^n 个子集，默认预算 256 意味着 n≥9 必然超支。

| 单元数 | 模式 | 每层组数 | 理由 |
|---|---|---|---|
| ≤8 | exact | min(n, 6) | 一次全枚举，最精确 |
| ≥9 | hierarchical | 6 | 2⁶=64，256 预算可展开 4 层 |

**为什么是 6 而不是 8**：8 组 = 256 组合，一次吃光预算，第二层无法展开，
退化成"只做一次粗分"。实测 16 段在 8 组时只展开 1 层、耗时 11.2 秒；
改成 6 组后展开 6 层、耗时 4.1 秒。

## 测量的局限

务必连同以下限制一起引用这些数字：

1. **单卡、无并发**：测量时只有单个请求在跑。并发下批处理会让单条均摊更低，
   但本表没有覆盖。
2. **同机回环**：服务与引擎在同一台机器，跨机调用的网络开销未计入。
3. **`--mem-fraction-static 0.75`**：显存配置会影响可并发的批大小。
4. **CUDA graph 前后不可直接比较**：两组数据在不同时间、不同系统状态下采集，
   不构成受控实验。本文的对比均为同一配置下的相对比较。
5. **`model_requests` 恒为 null**：评分器不返回 `prompt_index`，算法层拿不到该计数。
6. **样本量小**：每组 3–20 次，中位数为主，未做严格置信区间。

## 复现方式

```bash
# 1. 起引擎（单卡，只录 3 个 prefill 档位，避免默认 106 档的长时间录制）
CUDA_VISIBLE_DEVICES=6 SGLANG_ENABLE_FEATURE_CAPTURE=1 \
PATH=/opt/miniconda3/envs/inscope/bin:$PATH \
python -m sglang.launch_server \
  --model-path /data/djs/models/Qwen3-8B-FP8 \
  --host 127.0.0.1 --port 30000 --tp-size 1 \
  --mem-fraction-static 0.75 --chunked-prefill-size 32768 \
  --cuda-graph-bs-prefill 8192 16384 32768

# 2. 起服务
INSCOPE_MODEL_PATH=/data/djs/models/Qwen3-8B-FP8 \
INSCOPE_PROBE_CHECKPOINT=/data/zmosg/checkpoints/best_layers_08_16_24.pt \
INSCOPE_COMPOSER_PROFILE=a6_nemotron_risk_faced_v1 \
INSCOPE_API_KEY=<key> INSCOPE_ENGINE_URL=http://127.0.0.1:30000 \
python -m inscope_server.cli --host 127.0.0.1 --port 8302
```

> 首次启动会做一次 DeepGEMM JIT 编译（每种 GEMM shape 一轮），约 10 分钟。
> 想让后续启动复用，可先跑 `python -m sglang.compile_deep_gemm`。
