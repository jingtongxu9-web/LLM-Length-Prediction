from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PrefillSignals:
    layer: int
    last_token_hidden_state: Sequence[float]


@dataclass(frozen=True)
class DecodeSignals:
    step: int
    entropy: float
    eos_probability: float
    hidden_state: Sequence[float] | None = None


class SignalCollector(Protocol):
    """Adapter boundary for Hugging Face, vLLM, or another runtime."""

    def capture_prefill(self, prompt: str, layer: int) -> PrefillSignals: ...

    def generate(self, prompt: str) -> tuple[str, list[DecodeSignals]]: ...
