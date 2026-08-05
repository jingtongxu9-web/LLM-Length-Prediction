"""Versioned unified trace for the isolated ALPS+PLP Hybrid v3 experiment."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.data.schema import MetadataValue

HYBRID_TRACE_SCHEMA_VERSION = 1


@dataclass
class HybridV3Trace:
    prompt_id: str
    task: str
    prompt_tokens: int
    output_tokens: int
    temperature: float
    seed: int
    stop_reason: str
    prior_feature: np.ndarray
    prompt_feature: np.ndarray
    decode_hidden_states: np.ndarray
    steps: np.ndarray
    remaining_lengths: np.ndarray
    token_ids: np.ndarray
    generated_token_ids: np.ndarray
    entropies: np.ndarray
    entropy_means: np.ndarray
    entropy_slopes: np.ndarray
    eos_probabilities: np.ndarray
    model_name: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt_id or self.prompt_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("prompt identity and positive token counts are required")
        if self.temperature < 0 or not self.stop_reason:
            raise ValueError("invalid temperature or stop_reason")
        if self.stop_reason not in {"eos", "max_new_tokens"}:
            raise ValueError("unsupported stop_reason")
        if self.duration_ms is not None and (
            self.duration_ms < 0 or not math.isfinite(self.duration_ms)
        ):
            raise ValueError("duration_ms must be finite and non-negative")
        prior = np.asarray(self.prior_feature)
        prompt = np.asarray(self.prompt_feature)
        decode = np.asarray(self.decode_hidden_states)
        steps = np.asarray(self.steps)
        remaining = np.asarray(self.remaining_lengths)
        token_ids = np.asarray(self.token_ids)
        generated = np.asarray(self.generated_token_ids)
        signals = tuple(
            np.asarray(values)
            for values in (
                self.entropies,
                self.entropy_means,
                self.entropy_slopes,
                self.eos_probabilities,
            )
        )
        if prior.ndim != 1 or prompt.ndim != 1 or prior.shape != prompt.shape:
            raise ValueError("prior_feature and prompt_feature must share one hidden size")
        if decode.ndim != 2 or decode.shape[1:] != prompt.shape:
            raise ValueError("decode_hidden_states has the wrong hidden size")
        point_count = len(decode)
        if point_count == 0:
            raise ValueError("at least one progressive point is required")
        aligned = (steps, remaining, token_ids, *signals)
        if any(values.shape != (point_count,) for values in aligned):
            raise ValueError("all per-point arrays must align")
        if generated.shape != (self.output_tokens,):
            raise ValueError("generated_token_ids must contain the complete output")
        if not np.all(np.isfinite(prior)) or not np.all(np.isfinite(prompt)):
            raise ValueError("prompt-side features must be finite")
        if not np.all(np.isfinite(decode)) or any(
            not np.all(np.isfinite(values)) for values in signals
        ):
            raise ValueError("decode features must be finite")
        if np.any(signals[0] < 0) or np.any(signals[1] < 0):
            raise ValueError("entropy features must be non-negative")
        if np.any(signals[3] < 0) or np.any(signals[3] > 1):
            raise ValueError("EOS probabilities must lie in [0, 1]")
        if not np.all(np.diff(steps) > 0) or int(steps[0]) != 1:
            raise ValueError("steps must start at one and strictly increase")
        if int(steps[-1]) != self.output_tokens:
            raise ValueError("the terminal generation step must be saved")
        expected = self.output_tokens - steps.astype(np.int64)
        if np.any(remaining < 0) or not np.array_equal(remaining.astype(np.int64), expected):
            raise ValueError("remaining_lengths must equal output_tokens - steps")
        if np.any(token_ids < 0) or np.any(generated < 0):
            raise ValueError("token IDs must be non-negative")
        if not np.array_equal(token_ids, generated[steps.astype(np.int64) - 1]):
            raise ValueError("saved token IDs do not match the complete output")


def _metadata(trace: HybridV3Trace) -> dict[str, Any]:
    return {
        "schema_version": HYBRID_TRACE_SCHEMA_VERSION,
        "prompt_id": trace.prompt_id,
        "task": trace.task,
        "prompt_tokens": trace.prompt_tokens,
        "output_tokens": trace.output_tokens,
        "temperature": trace.temperature,
        "seed": trace.seed,
        "stop_reason": trace.stop_reason,
        "model_name": trace.model_name,
        "model_revision": trace.model_revision,
        "tokenizer_revision": trace.tokenizer_revision,
        "duration_ms": trace.duration_ms,
        "metadata": trace.metadata,
    }


def write_hybrid_trace(path: str | Path, trace: HybridV3Trace) -> Path:
    trace.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(_metadata(trace), ensure_ascii=False, separators=(",", ":"))
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(metadata_json),
            prior_feature=np.asarray(trace.prior_feature, dtype=np.float32),
            prompt_feature=np.asarray(trace.prompt_feature, dtype=np.float32),
            decode_hidden_states=np.asarray(trace.decode_hidden_states, dtype=np.float32),
            steps=np.asarray(trace.steps, dtype=np.int32),
            remaining_lengths=np.asarray(trace.remaining_lengths, dtype=np.int32),
            token_ids=np.asarray(trace.token_ids, dtype=np.int32),
            generated_token_ids=np.asarray(trace.generated_token_ids, dtype=np.int32),
            entropies=np.asarray(trace.entropies, dtype=np.float32),
            entropy_means=np.asarray(trace.entropy_means, dtype=np.float32),
            entropy_slopes=np.asarray(trace.entropy_slopes, dtype=np.float32),
            eos_probabilities=np.asarray(trace.eos_probabilities, dtype=np.float32),
        )
    return output


def read_hybrid_trace(path: str | Path) -> HybridV3Trace:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "metadata_json",
            "prior_feature",
            "prompt_feature",
            "decode_hidden_states",
            "steps",
            "remaining_lengths",
            "token_ids",
            "generated_token_ids",
            "entropies",
            "entropy_means",
            "entropy_slopes",
            "eos_probabilities",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Hybrid trace is missing arrays: {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.pop("schema_version", None) != HYBRID_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported Hybrid trace schema_version")
        trace = HybridV3Trace(
            **metadata,
            prior_feature=payload["prior_feature"].astype(np.float32, copy=False),
            prompt_feature=payload["prompt_feature"].astype(np.float32, copy=False),
            decode_hidden_states=payload["decode_hidden_states"].astype(np.float32, copy=False),
            steps=payload["steps"].astype(np.int32, copy=False),
            remaining_lengths=payload["remaining_lengths"].astype(np.int32, copy=False),
            token_ids=payload["token_ids"].astype(np.int32, copy=False),
            generated_token_ids=payload["generated_token_ids"].astype(np.int32, copy=False),
            entropies=payload["entropies"].astype(np.float32, copy=False),
            entropy_means=payload["entropy_means"].astype(np.float32, copy=False),
            entropy_slopes=payload["entropy_slopes"].astype(np.float32, copy=False),
            eos_probabilities=payload["eos_probabilities"].astype(np.float32, copy=False),
        )
    trace.validate()
    return trace


def hybrid_trace_path(root: str | Path, record: dict[str, Any], seed: int) -> Path:
    return Path(root) / record["split"] / record["prompt_id"] / f"seed_{seed}.npz"
