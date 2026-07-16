from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TracePoint:
    """Signals observed at one decode step."""

    step: int
    entropy: float
    eos_probability: float
    remaining_length: int
    entropy_mean: float | None = None
    entropy_slope: float | None = None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.remaining_length < 0:
            raise ValueError("remaining_length must be non-negative")
        if not 0.0 <= self.eos_probability <= 1.0:
            raise ValueError("eos_probability must be in [0, 1]")


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

    def validate(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id is required")
        if self.prompt_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if any(point.step > self.output_tokens for point in self.points):
            raise ValueError("trace point cannot occur after the output ends")
