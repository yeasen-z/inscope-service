"""有界分桶与同序打分适配，不依赖 HTTP、tokenizer 或探针。

当前分析以消息 content 字符数粗分桶，不是 token 数或显存预算。
每桶串行提交；桶数取决于输入长度与 max_batch，不能保证固定往返数。
分桶只是调度策略，性能收益须由真实引擎实测。
"""
import asyncio
import inspect


async def _maybe_await(value):
    """打分函数可以是同步或 async 的：服务内是 async，测试替身多为同步。"""
    if inspect.isawaitable(value):
        return await value
    return value


def bucket_by_length(items, *, length_of=len, tolerance=0.2, max_batch=8):
    """按长度分桶：桶内 (最长 - 最短) / 最长 <= tolerance，且不超过 max_batch。

    返回若干桶；每项为 (原始下标, 输入)。按长度升序排列，同长时保留原序。
    """
    if not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be within [0, 1]")
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    indexed = list(enumerate(items))
    indexed.sort(key=lambda pair: (length_of(pair[1]), pair[0]))
    buckets, current = [], []
    for original_index, item in indexed:
        if current:
            longest = max(length_of(x) for _, x in current)
            shortest = min(length_of(x) for _, x in current)
            candidate = max(longest, length_of(item))
            fits = (len(current) < max_batch and candidate > 0
                    and (candidate - min(shortest, length_of(item))) / candidate <= tolerance)
            if not fits:
                buckets.append(current)
                current = []
        current.append((original_index, item))
    if current:
        buckets.append(current)
    return buckets


def make_score_batch(score_one, *, concurrency=8):
    """逐条打分的兼容路径：同步或 async 的 ``score_one`` 包成批量接口。

    当调用方只能一条一条打分时使用；服务内实际走 ``make_batched_scorer``。
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    sem = asyncio.Semaphore(concurrency)

    async def score_batch(requests):
        async def one(request):
            async with sem:
                return await _maybe_await(score_one(request))

        return await asyncio.gather(*(one(r) for r in requests))

    return score_batch


def make_batched_scorer(score_prompts, *, length_of="messages", max_batch=8, tolerance=0.2):
    """批量打分器：按长度分桶，每桶一次请求送进引擎。

    ``score_prompts(bucket) -> list[{"logit": ...}]``：接收一整桶请求，返回
    与桶内顺序一致的分数。token 长度由调用方在桶内自行决定（它才有
    tokenizer），这里只按字符数等可用的廉价指标做粗分桶。
    """
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    if length_of != "messages":
        raise ValueError("only message-content length is supported for bucketing")

    def _size(request):
        # 分桶只需要"谁明显更长"这个粗粒度信号；真正的 token 长度由
        # score_prompts 内部的 tokenizer 决定，这里不用它排序。
        return sum(len(str(m.get("content") or ""))
                   for m in request.get("messages", ()))

    async def score_batch(requests):
        buckets = bucket_by_length(requests, length_of=_size,
                                   tolerance=tolerance, max_batch=max_batch)
        ordered = [None] * len(requests)
        for bucket in buckets:
            rows = await _maybe_await(score_prompts([req for _, req in bucket]))
            if len(rows) != len(bucket):
                raise ValueError("scorer returned %d rows for %d prompts"
                                 % (len(rows), len(bucket)))
            # 用原始下标还原顺序；相同对象重复出现也不会丢失。
            for (original_index, _), row in zip(bucket, rows):
                ordered[original_index] = row
        return ordered

    return score_batch
