from .composer import ComposerConfig, compose_prompt


def render_ids(tokenizer, messages, tools=None, config=None):
    """在线入口：返回必须原样送进 engine 的 token ids。

    规则全部在 composer 里（选窗、discard CoT、merge、question prefix）；
    这里只保持原有调用形状，避免热路径出现第二套消息处理。
    """
    return list(compose_prompt(tokenizer, messages, tools, config or ComposerConfig()).token_ids)
