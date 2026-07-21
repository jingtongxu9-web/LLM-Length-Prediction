from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypeAlias

MetadataValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class TracePoint:
    """Signals observed at one decode step."""

    step: int
    entropy: float
    eos_probability: float
    remaining_length: int
    entropy_mean: float | None = None
    entropy_slope: float | None = None
    token_id: int | None = None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.remaining_length < 0:
            raise ValueError("remaining_length must be non-negative")
        if not math.isfinite(self.entropy) or self.entropy < 0:
            raise ValueError("entropy must be finite and non-negative")
        if not 0.0 <= self.eos_probability <= 1.0:
            raise ValueError("eos_probability must be in [0, 1]")
        if self.token_id is not None and self.token_id < 0:
            raise ValueError("token_id must be non-negative")


@dataclass
class GenerationTrace:
    """One complete sampled generation and its instrumentation."""

    prompt_id: str
    task: str
    prompt_tokens: int
    output_tokens: int
    temperature: float
    seed: int
    stop_reason: str
    points: list[TracePoint] = field(default_factory=list)
    model_name: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    generated_text: str = ""
    prefill_hidden_states: dict[int, list[float]] = field(default_factory=dict)
    duration_ms: float | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id is required")
        if self.prompt_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not self.stop_reason:
            raise ValueError("stop_reason is required")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

        steps = [point.step for point in self.points]
        if steps != sorted(set(steps)):
            raise ValueError("trace point steps must be unique and sorted")
        for point in self.points:
            if point.step > self.output_tokens:
                raise ValueError("trace point cannot occur after the output ends")
            if point.remaining_length != self.output_tokens - point.step:
                raise ValueError("remaining_length must equal output_tokens - step")

        for layer, values in self.prefill_hidden_states.items():
            if layer < 0 or not values:
                raise ValueError("prefill hidden states require non-negative layers and values")
            if not all(math.isfinite(value) for value in values):
                raise ValueError("prefill hidden states must contain finite values")
