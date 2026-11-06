"""Compiled-context API: token IDs already carry the exact composer/tools policy."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt


class CompiledDetectionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    composer_id: str=Field(min_length=1,max_length=256)
    input_ids: list[StrictInt]=Field(min_length=1,max_length=1048576)
    pre_position: StrictInt
    observation_tail_position: StrictInt
    post_position: StrictInt
    suffix_position: StrictInt
    output: Literal["tokens","scores"]="tokens"
    enabled_detectors: list[str] | None=Field(default=None,max_length=128)


class DetectorOutput(BaseModel):
    layer: int
    logit: float
    logit_threshold: float
    flagged: bool
    region_logit: float
    detector_logit: float | None
    probability: float | None=None
    token_logits: list[float] | None=None
    token_positions: list[int] | None=None
    token_spans: list[list[int]] | None=None
    threshold_source: str | None=None


class CompiledDetectionResponse(BaseModel):
    composer_id: str
    input_tokens: int
    detectors: dict[str,DetectorOutput]
    latency_ms: dict[str,float]
    transfer: dict[str,str]
