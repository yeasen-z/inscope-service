"""Composer adapters. Never re-render an already compiled training prompt."""
from .contracts import CompiledPrompt


class CompiledComposer:
    """Explicit boundary for trusted compiled contexts, preserving full tools/CoT.

    Natural-message composers can implement the same interface. They must be
    versioned and validated against training before registering with a detector.
    No current checkpoint is implicitly bound to a legacy A6/v3 composer.
    """
    def __init__(self, composer_id, max_tokens=40959):
        self.composer_id = composer_id
        self.max_tokens = max_tokens

    def compose(self, request):
        if request.get('composer_id') != self.composer_id:
            raise ValueError('Request composer does not match detector manifest')
        ids = request.get('input_ids')
        if not isinstance(ids,list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('input_ids must be nonempty nonnegative integers')
        if len(ids) > self.max_tokens:
            raise ValueError('Prompt exceeds configured context; no truncation')
        names = ('pre_position','observation_tail_position','post_position','suffix_position')
        bounds = [request.get(k) for k in names]
        if any(type(x) is not int for x in bounds):
            raise ValueError('Explicit pre/tail/post/suffix boundaries required')
        a,b,c,d = bounds
        if not 0 <= a < b < c < d < len(ids):
            raise ValueError('Invalid compiled context boundaries')
        return CompiledPrompt(tuple(ids),a,b,c,d,self.composer_id,
            metadata=dict(output=request.get("output","tokens"),enabled_detectors=request.get("enabled_detectors")))
