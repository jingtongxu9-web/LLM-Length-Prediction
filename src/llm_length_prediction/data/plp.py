from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.data.schema import MetadataValue

PLP_TRACE_SCHEMA_VERSION = 2


@dataclass
class PLPHiddenStateTrace:
    """One rollout with the hidden-state inputs required by paper-style PLP."""

    prompt_id: str
    task: str
    prompt_tokens: int
    output_tokens: int
    temperature: float
    seed: int
    stop_reason: str
    prompt_feature: np.ndarray
    decode_hidden_states: np.ndarray
    steps: np.ndarray
    remaining_lengths: np.ndarray
    token_ids: np.ndarray
    generated_token_ids: np.ndarray
    model_name: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id is required")
        if self.prompt_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("prompt_tokens and output_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not self.stop_reason:
            raise ValueError("stop_reason is required")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

        prompt_feature = np.asarray(self.prompt_feature)
        decode = np.asarray(self.decode_hidden_states)
        steps = np.asarray(self.steps)
        remaining = np.asarray(self.remaining_lengths)
        token_ids = np.asarray(self.token_ids)
        generated_token_ids = np.asarray(self.generated_token_ids)
        if prompt_feature.ndim != 1 or prompt_feature.size == 0:
            raise ValueError("prompt_feature must be a non-empty vector")
        if decode.ndim != 2 or decode.shape[1] != prompt_feature.size:
            raise ValueError("decode_hidden_states must have shape (points, hidden_size)")
        point_count = decode.shape[0]
        if point_count == 0:
            raise ValueError("at least one PLP point is required")
        if steps.shape != (point_count,):
            raise ValueError("steps must align with decode_hidden_states")
        if remaining.shape != (point_count,):
            raise ValueError("remaining_lengths must align with decode_hidden_states")
        if token_ids.shape != (point_count,):
            raise ValueError("token_ids must align with decode_hidden_states")
        if generated_token_ids.shape != (self.output_tokens,):
            raise ValueError("generated_token_ids must contain the complete generated sequence")
        if not np.all(np.isfinite(prompt_feature)) or not np.all(np.isfinite(decode)):
            raise ValueError("hidden-state features must be finite")
        if not np.all(np.diff(steps) > 0):
            raise ValueError("steps must be strictly increasing")
        if int(steps[0]) < 1 or int(steps[-1]) > self.output_tokens:
            raise ValueError("steps must fall within the generated output")
        if np.any(remaining < 0):
            raise ValueError("remaining lengths must be non-negative")
        expected = self.output_tokens - steps.astype(np.int64)
        if not np.array_equal(remaining.astype(np.int64), expected):
            raise ValueError("remaining_lengths must equal output_tokens - steps")
        if np.any(token_ids < 0):
            raise ValueError("token_ids must be non-negative")
        if np.any(generated_token_ids < 0):
            raise ValueError("generated_token_ids must be non-negative")
        if not np.array_equal(token_ids, generated_token_ids[steps.astype(np.int64) - 1]):
            raise ValueError("saved token_ids must match generated_token_ids at saved steps")
        if self.duration_ms is not None and not math.isfinite(self.duration_ms):
            raise ValueError("duration_ms must be finite")


def _metadata_payload(trace: PLPHiddenStateTrace) -> dict[str, Any]:
    return {
        "schema_version": PLP_TRACE_SCHEMA_VERSION,
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


def write_plp_trace(path: str | Path, trace: PLPHiddenStateTrace) -> Path:
    """Store one PLP trace as a compressed, pickle-free NPZ file."""

    trace.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(
        _metadata_payload(trace), ensure_ascii=False, separators=(",", ":")
    )
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(metadata_json),
            prompt_feature=np.asarray(trace.prompt_feature, dtype=np.float32),
            decode_hidden_states=np.asarray(trace.decode_hidden_states, dtype=np.float32),
            steps=np.asarray(trace.steps, dtype=np.int32),
            remaining_lengths=np.asarray(trace.remaining_lengths, dtype=np.int32),
            token_ids=np.asarray(trace.token_ids, dtype=np.int32),
            generated_token_ids=np.asarray(trace.generated_token_ids, dtype=np.int32),
        )
    return output


def read_plp_trace(path: str | Path) -> PLPHiddenStateTrace:
    """Load and validate one pickle-free PLP NPZ trace."""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "metadata_json",
            "prompt_feature",
            "decode_hidden_states",
            "steps",
            "remaining_lengths",
            "token_ids",
            "generated_token_ids",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"PLP trace is missing arrays: {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.pop("schema_version", None) != PLP_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported PLP trace schema_version")
        trace = PLPHiddenStateTrace(
            **metadata,
            prompt_feature=payload["prompt_feature"].astype(np.float32, copy=False),
            decode_hidden_states=payload["decode_hidden_states"].astype(
                np.float32, copy=False
            ),
            steps=payload["steps"].astype(np.int32, copy=False),
            remaining_lengths=payload["remaining_lengths"].astype(np.int32, copy=False),
            token_ids=payload["token_ids"].astype(np.int32, copy=False),
            generated_token_ids=payload["generated_token_ids"].astype(
                np.int32, copy=False
            ),
        )
    trace.validate()
    return trace


def plp_trace_path(
    trace_root: str | Path, record: dict[str, Any], seed: int
) -> Path:
    return Path(trace_root) / record["split"] / record["prompt_id"] / f"seed_{seed}.npz"
