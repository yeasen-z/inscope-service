"""GT-independent positions. Detector demand is merged before one capture."""
from .contracts import CapturePlan


class AllPositionPlanner:
    def plan(self,prompt,layers):
        if not layers or tuple(sorted(set(layers))) != tuple(layers):
            raise ValueError('Capture layers must be ordered unique boundary IDs')
        region = tuple(range(prompt.pre,prompt.tail+1))
        # Position0 forces the existing native full-prefill contract, even if
        # the selected observation begins after a reusable cached prefix.
        selected = tuple(sorted({0,*region,prompt.post,prompt.suffix}))
        return CapturePlan(tuple(layers),selected,region,(prompt.post,prompt.suffix))
