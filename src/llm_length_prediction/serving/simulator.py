from __future__ import annotations

from dataclasses import dataclass


def length_bucket(predicted_remaining: float, boundaries: tuple[int, ...]) -> int:
    """Return a stable bucket index for prediction-aware batching."""

    if predicted_remaining < 0:
        raise ValueError("predicted_remaining must be non-negative")
    if tuple(sorted(boundaries)) != boundaries:
        raise ValueError("boundaries must be sorted")
    for index, boundary in enumerate(boundaries):
        if predicted_remaining <= boundary:
            return index
    return len(boundaries)


@dataclass(frozen=True)
class ServingMetrics:
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_s: float
    padding_waste_tokens: int
    kv_cache_peak_bytes: int
    oom_count: int
