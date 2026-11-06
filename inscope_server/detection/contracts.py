"""Immutable data crossing composer/planner/capture/detector boundaries."""
from dataclasses import dataclass, field
from typing import Protocol, Any


@dataclass(frozen=True)
class CompiledPrompt:
    input_ids: tuple[int, ...]
    pre: int
    tail: int
    post: int
    suffix: int
    composer_id: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CapturePlan:
    layers: tuple[int, ...]
    positions: tuple[int, ...]
    region: tuple[int, ...]
    anchors: tuple[int, int]
    full_prefill: bool = True


class Composer(Protocol):
    def compose(self, request: dict) -> CompiledPrompt: ...


class PositionPlanner(Protocol):
    def plan(self, prompt: CompiledPrompt, layers: tuple[int, ...]) -> CapturePlan: ...


class FeatureBackend(Protocol):
    async def capture(self, prompt: CompiledPrompt, plan: CapturePlan) -> Any: ...
    async def close(self): ...


class DetectorBank(Protocol):
    layers: list[int]
    dim: int
    def predict(self, batch: list) -> dict: ...


class PostProcessor(Protocol):
    def process(self, scores: dict, prompt: CompiledPrompt, plan: CapturePlan) -> dict: ...
