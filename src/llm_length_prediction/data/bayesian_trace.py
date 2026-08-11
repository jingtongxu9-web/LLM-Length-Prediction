"""Pickle-free unified trace schema for Bayesian Sequential collection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.data.sequential import (
    SequentialRawTrace,
    scheduled_update_steps,
)

BAYESIAN_TRACE_SCHEMA_NAME = "bayesian-sequential-unified-trace"
BAYESIAN_TRACE_SCHEMA_VERSION = 1


@dataclass
class BayesianTraceV1:
    """One rollout containing every frozen baseline and Bayesian evidence input."""

    prompt_id: str
    prompt_family_id: str
    task: str
    intended_length: str
    split: str
    prompt_tokens: int
    observed_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    stop_reason: str
    eos_token_ids: tuple[int, ...]
    prior_feature: np.ndarray
    prompt_feature: np.ndarray
    initial_decode_hidden_state: np.ndarray
    decode_hidden_states: np.ndarray
    saved_steps: np.ndarray
    generated_token_ids: np.ndarray
    token_entropies: np.ndarray
    token_eos_probabilities: np.ndarray
    model_name: str
    model_revision: str
    tokenizer_revision: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_censored(self) -> bool:
        return self.stop_reason == "max_new_tokens"

    @property
    def terminal_observed(self) -> bool:
        return self.stop_reason == "eos"

    @property
    def hidden_size(self) -> int:
        return int(np.asarray(self.prior_feature).size)

    def validate(self, *, stride: int = 5) -> None:
        required_text = {
            "prompt_id": self.prompt_id,
            "prompt_family_id": self.prompt_family_id,
            "task": self.task,
            "intended_length": self.intended_length,
            "split": self.split,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
        }
        missing = [name for name, value in required_text.items() if not value]
        if missing:
            raise ValueError(f"Bayesian trace is missing required text fields: {missing}")
        if self.stop_reason not in {"eos", "max_new_tokens"}:
            raise ValueError("stop_reason must be eos or max_new_tokens")
        if self.prompt_tokens <= 0 or self.observed_tokens <= 0 or self.max_new_tokens <= 0:
            raise ValueError("prompt, observed, and maximum token counts must be positive")
        if self.observed_tokens > self.max_new_tokens:
            raise ValueError("observed_tokens cannot exceed max_new_tokens")
        if self.is_censored and self.observed_tokens != self.max_new_tokens:
            raise ValueError("max_new_tokens stop must reach the generation cap")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Bayesian collection requires a positive finite temperature")
        if not math.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        if not math.isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("duration_ms must be finite and non-negative")
        if any(token_id < 0 for token_id in self.eos_token_ids):
            raise ValueError("EOS token IDs cannot be negative")

        prior = np.asarray(self.prior_feature)
        prompt = np.asarray(self.prompt_feature)
        initial = np.asarray(self.initial_decode_hidden_state)
        decode = np.asarray(self.decode_hidden_states)
        steps = np.asarray(self.saved_steps)
        generated = np.asarray(self.generated_token_ids)
        entropies = np.asarray(self.token_entropies)
        eos_probabilities = np.asarray(self.token_eos_probabilities)
        if prior.ndim != 1 or prior.size == 0:
            raise ValueError("prior_feature must be a non-empty vector")
        if prompt.shape != prior.shape or initial.shape != prior.shape:
            raise ValueError("prompt-side hidden states must use one hidden size")
        expected_steps = scheduled_update_steps(
            self.observed_tokens,
            terminal_observed=self.terminal_observed,
            stride=stride,
        )
        if steps.shape != expected_steps.shape or not np.array_equal(steps, expected_steps):
            raise ValueError("saved_steps violate the frozen first/stride/terminal schedule")
        if decode.shape != (len(steps), prior.size):
            raise ValueError("decode_hidden_states must align with saved steps and hidden size")
        hidden_arrays = (prior, prompt, initial, decode)
        if any(np.any(~np.isfinite(array)) for array in hidden_arrays):
            raise ValueError("hidden-state arrays must be finite")
        if generated.shape != (self.observed_tokens,):
            raise ValueError("generated_token_ids must cover every observed token")
        if np.any(generated < 0):
            raise ValueError("generated token IDs cannot be negative")
        if entropies.shape != generated.shape or eos_probabilities.shape != generated.shape:
            raise ValueError("raw entropy and EOS arrays must cover every observed token")
        if np.any(~np.isfinite(entropies)) or np.any(entropies < 0):
            raise ValueError("token entropies must be finite and non-negative")
        if (
            np.any(~np.isfinite(eos_probabilities))
            or np.any(eos_probabilities < 0)
            or np.any(eos_probabilities > 1)
        ):
            raise ValueError("token EOS probabilities must lie in [0, 1]")
        if self.terminal_observed:
            if not self.eos_token_ids:
                raise ValueError("EOS termination requires at least one EOS token ID")
            if int(generated[-1]) not in self.eos_token_ids:
                raise ValueError("the final generated token does not match an EOS token ID")
        try:
            json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be finite JSON data") from error


def _metadata_payload(trace: BayesianTraceV1) -> dict[str, Any]:
    return {
        "schema_name": BAYESIAN_TRACE_SCHEMA_NAME,
        "schema_version": BAYESIAN_TRACE_SCHEMA_VERSION,
        "prompt_id": trace.prompt_id,
        "prompt_family_id": trace.prompt_family_id,
        "task": trace.task,
        "intended_length": trace.intended_length,
        "split": trace.split,
        "prompt_tokens": trace.prompt_tokens,
        "observed_tokens": trace.observed_tokens,
        "max_new_tokens": trace.max_new_tokens,
        "temperature": trace.temperature,
        "top_p": trace.top_p,
        "seed": trace.seed,
        "stop_reason": trace.stop_reason,
        "eos_token_ids": list(trace.eos_token_ids),
        "model_name": trace.model_name,
        "model_revision": trace.model_revision,
        "tokenizer_revision": trace.tokenizer_revision,
        "duration_ms": trace.duration_ms,
        "metadata": trace.metadata,
    }


def write_bayesian_trace(path: str | Path, trace: BayesianTraceV1) -> Path:
    """Write one atomic-ready compressed NPZ without Python pickle payloads."""

    trace.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(
        _metadata_payload(trace),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(metadata_json),
            prior_feature=np.asarray(trace.prior_feature, dtype=np.float32),
            prompt_feature=np.asarray(trace.prompt_feature, dtype=np.float32),
            initial_decode_hidden_state=np.asarray(
                trace.initial_decode_hidden_state,
                dtype=np.float32,
            ),
            decode_hidden_states=np.asarray(trace.decode_hidden_states, dtype=np.float32),
            saved_steps=np.asarray(trace.saved_steps, dtype=np.int32),
            generated_token_ids=np.asarray(trace.generated_token_ids, dtype=np.int32),
            token_entropies=np.asarray(trace.token_entropies, dtype=np.float32),
            token_eos_probabilities=np.asarray(
                trace.token_eos_probabilities,
                dtype=np.float32,
            ),
        )
    return output


def read_bayesian_trace(path: str | Path) -> BayesianTraceV1:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "metadata_json",
            "prior_feature",
            "prompt_feature",
            "initial_decode_hidden_state",
            "decode_hidden_states",
            "saved_steps",
            "generated_token_ids",
            "token_entropies",
            "token_eos_probabilities",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Bayesian trace is missing arrays: {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.pop("schema_name", None) != BAYESIAN_TRACE_SCHEMA_NAME:
            raise ValueError("unsupported Bayesian trace schema_name")
        if metadata.pop("schema_version", None) != BAYESIAN_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported Bayesian trace schema_version")
        metadata["eos_token_ids"] = tuple(int(value) for value in metadata["eos_token_ids"])
        trace = BayesianTraceV1(
            **metadata,
            prior_feature=payload["prior_feature"].astype(np.float32, copy=False),
            prompt_feature=payload["prompt_feature"].astype(np.float32, copy=False),
            initial_decode_hidden_state=payload["initial_decode_hidden_state"].astype(
                np.float32,
                copy=False,
            ),
            decode_hidden_states=payload["decode_hidden_states"].astype(
                np.float32,
                copy=False,
            ),
            saved_steps=payload["saved_steps"].astype(np.int32, copy=False),
            generated_token_ids=payload["generated_token_ids"].astype(
                np.int32,
                copy=False,
            ),
            token_entropies=payload["token_entropies"].astype(np.float32, copy=False),
            token_eos_probabilities=payload["token_eos_probabilities"].astype(
                np.float32,
                copy=False,
            ),
        )
    trace.validate()
    return trace


def temperature_path_component(temperature: float) -> str:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    return f"temperature_{temperature:.3f}".replace(".", "p")


def bayesian_trace_path(
    root: str | Path,
    *,
    split: str,
    prompt_id: str,
    temperature: float,
    seed: int,
) -> Path:
    return (
        Path(root)
        / split
        / prompt_id
        / temperature_path_component(temperature)
        / f"seed_{seed}.npz"
    )


def sequential_raw_trace_from_collected(
    trace: BayesianTraceV1,
    *,
    prior_mu: float,
    prior_log_variance: float,
    prior_mean_total_tokens: float,
) -> SequentialRawTrace:
    """Attach leakage-safe fold-specific ALPS summaries to one collected trace."""

    trace.validate()
    return SequentialRawTrace(
        prompt_id=trace.prompt_id,
        prompt_family_id=trace.prompt_family_id,
        task=trace.task,
        intended_length=trace.intended_length,
        temperature=trace.temperature,
        seed=trace.seed,
        stop_reason=trace.stop_reason,
        observed_tokens=trace.observed_tokens,
        max_new_tokens=trace.max_new_tokens,
        prior_mu=prior_mu,
        prior_log_variance=prior_log_variance,
        prior_mean_total_tokens=prior_mean_total_tokens,
        token_entropies=trace.token_entropies,
        token_eos_probabilities=trace.token_eos_probabilities,
        saved_steps=trace.saved_steps,
        initial_decode_hidden_state=trace.initial_decode_hidden_state,
        decode_hidden_states=trace.decode_hidden_states,
    )
