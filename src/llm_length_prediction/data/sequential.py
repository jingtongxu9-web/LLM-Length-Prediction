"""Leakage-safe sequence objects for Bayesian remaining-length inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from llm_length_prediction.models.bayesian_filter import (
    shifted_lognormal_integer_prior,
)

SCALAR_EVIDENCE_FEATURE_NAMES = (
    "mu_log1p_total",
    "oof_calibrated_log_variance",
    "prior_mean_total_tokens",
    "step",
    "delta_step",
    "step_fraction_of_max_new_tokens",
    "last_entropy",
    "block_entropy_mean",
    "block_entropy_slope",
    "last_eos_probability",
    "block_max_eos_probability",
    "delta_entropy",
    "delta_eos_probability",
    "terminal_observed",
)

CANDIDATE_FEATURE_NAMES = (
    "remaining_tokens",
    "log1p_remaining_tokens",
    "remaining_fraction_of_max_new_tokens",
    "candidate_total_tokens",
    "overflow_indicator",
)


def scheduled_update_steps(
    observed_tokens: int,
    *,
    terminal_observed: bool,
    stride: int = 5,
) -> np.ndarray:
    """Return steps 1, stride, 2*stride, ... and the true terminal token."""

    if observed_tokens <= 0 or stride <= 1:
        raise ValueError("observed_tokens must be positive and stride must exceed one")
    steps = {1, *range(stride, observed_tokens + 1, stride)}
    if terminal_observed:
        steps.add(observed_tokens)
    return np.asarray(sorted(steps), dtype=np.int32)


def _block_slope(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    positions = np.arange(values.size, dtype=np.float64)
    centered = positions - positions.mean()
    denominator = float(np.square(centered).sum())
    return float(np.sum(centered * (values - values.mean())) / denominator)


@dataclass(frozen=True)
class SequentialRawTrace:
    """One rollout with per-token signals needed to form non-overlapping evidence."""

    prompt_id: str
    prompt_family_id: str
    task: str
    intended_length: str
    temperature: float
    seed: int
    stop_reason: str
    observed_tokens: int
    max_new_tokens: int
    prior_mu: float
    prior_log_variance: float
    prior_mean_total_tokens: float
    token_entropies: np.ndarray
    token_eos_probabilities: np.ndarray
    saved_steps: np.ndarray
    initial_decode_hidden_state: np.ndarray | None = None
    decode_hidden_states: np.ndarray | None = None

    @property
    def is_censored(self) -> bool:
        return self.stop_reason == "max_new_tokens"

    @property
    def terminal_observed(self) -> bool:
        return self.stop_reason == "eos"

    def validate(self, *, stride: int = 5) -> None:
        if not self.prompt_id or not self.prompt_family_id:
            raise ValueError("prompt_id and prompt_family_id are required")
        if not self.task or not self.intended_length:
            raise ValueError("task and intended_length are required")
        if self.stop_reason not in {"eos", "max_new_tokens"}:
            raise ValueError("stop_reason must be eos or max_new_tokens")
        if self.observed_tokens <= 0 or self.max_new_tokens <= 0:
            raise ValueError("token counts must be positive")
        if self.observed_tokens > self.max_new_tokens:
            raise ValueError("observed_tokens cannot exceed max_new_tokens")
        if self.is_censored and self.observed_tokens != self.max_new_tokens:
            raise ValueError("a max_new_tokens stop must reach the configured cap")
        scalars = (
            self.temperature,
            self.prior_mu,
            self.prior_log_variance,
            self.prior_mean_total_tokens,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("temperature and prior summaries must be finite")
        if self.temperature < 0 or self.prior_log_variance < 0:
            raise ValueError("temperature and prior variance cannot be negative")
        if self.prior_mean_total_tokens < 0:
            raise ValueError("prior mean total tokens cannot be negative")

        entropies = np.asarray(self.token_entropies)
        eos = np.asarray(self.token_eos_probabilities)
        steps = np.asarray(self.saved_steps)
        if entropies.shape != (self.observed_tokens,) or eos.shape != entropies.shape:
            raise ValueError("per-token signals must cover every observed token")
        if np.any(~np.isfinite(entropies)) or np.any(entropies < 0):
            raise ValueError("entropies must be finite and non-negative")
        if np.any(~np.isfinite(eos)) or np.any(eos < 0) or np.any(eos > 1):
            raise ValueError("EOS probabilities must lie in [0, 1]")
        expected_steps = scheduled_update_steps(
            self.observed_tokens,
            terminal_observed=self.terminal_observed,
            stride=stride,
        )
        if steps.shape != expected_steps.shape or not np.array_equal(steps, expected_steps):
            raise ValueError("saved_steps do not match the frozen update schedule")

        if (self.initial_decode_hidden_state is None) != (
            self.decode_hidden_states is None
        ):
            raise ValueError("initial and saved decode hidden states must be provided together")
        if self.decode_hidden_states is not None:
            initial = np.asarray(self.initial_decode_hidden_state)
            decode = np.asarray(self.decode_hidden_states)
            if initial.ndim != 1 or initial.size == 0:
                raise ValueError("initial_decode_hidden_state must be a non-empty vector")
            if decode.shape != (len(expected_steps), initial.size):
                raise ValueError("decode_hidden_states must align with saved steps")
            if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(decode)):
                raise ValueError("decode hidden states must be finite")


@dataclass(frozen=True)
class SequentialEvidenceStep:
    step: int
    delta: int
    scalar_features: np.ndarray
    candidate_features: np.ndarray
    true_remaining: int | None
    censored_after_remaining: int | None
    terminal_observed: bool
    hidden_delta: np.ndarray | None = None

    def validate(self) -> None:
        scalar = np.asarray(self.scalar_features)
        candidates = np.asarray(self.candidate_features)
        if self.step <= 0 or self.delta <= 0:
            raise ValueError("step and delta must be positive")
        if scalar.shape != (len(SCALAR_EVIDENCE_FEATURE_NAMES),):
            raise ValueError("scalar evidence has the wrong dimension")
        if candidates.ndim != 2 or candidates.shape[1] != len(CANDIDATE_FEATURE_NAMES):
            raise ValueError("candidate features have the wrong shape")
        if np.any(~np.isfinite(scalar)) or np.any(~np.isfinite(candidates)):
            raise ValueError("evidence and candidate features must be finite")
        if (self.true_remaining is None) == (self.censored_after_remaining is None):
            raise ValueError("each step needs exactly one exact or right-censored target")
        if self.true_remaining is not None and self.true_remaining < 0:
            raise ValueError("true remaining length cannot be negative")
        if self.censored_after_remaining is not None and self.censored_after_remaining < 0:
            raise ValueError("censoring threshold cannot be negative")
        if self.hidden_delta is not None:
            hidden = np.asarray(self.hidden_delta)
            if hidden.ndim != 1 or hidden.size == 0 or np.any(~np.isfinite(hidden)):
                raise ValueError("hidden_delta must be a finite non-empty vector")


@dataclass(frozen=True)
class BayesianSequence:
    prompt_id: str
    prompt_family_id: str
    seed: int
    max_new_tokens: int
    initial_log_probabilities: np.ndarray
    steps: tuple[SequentialEvidenceStep, ...]
    prior_lower_tail_mass: float
    prior_upper_tail_mass: float
    has_overflow: bool = True

    @property
    def sequence_id(self) -> tuple[str, int]:
        return self.prompt_id, self.seed

    def validate(self) -> None:
        initial = np.asarray(self.initial_log_probabilities)
        expected = self.max_new_tokens + 2
        if initial.shape != (expected,) or np.any(np.isnan(initial)):
            raise ValueError("initial posterior must contain exact states plus overflow")
        if not self.steps:
            raise ValueError("a Bayesian sequence needs at least one evidence step")
        previous = 0
        expected_candidate_count = expected
        hidden_size: int | None = None
        for evidence_step in self.steps:
            evidence_step.validate()
            if evidence_step.step - previous != evidence_step.delta:
                raise ValueError("step deltas must connect adjacent updates")
            expected_candidate_count -= evidence_step.delta
            if len(evidence_step.candidate_features) != expected_candidate_count:
                raise ValueError("candidate support does not match the countdown transition")
            if evidence_step.hidden_delta is not None:
                current_size = len(evidence_step.hidden_delta)
                if hidden_size is None:
                    hidden_size = current_size
                elif hidden_size != current_size:
                    raise ValueError("hidden delta dimensions must remain constant")
            previous = evidence_step.step


def candidate_feature_matrix(
    *,
    step: int,
    exact_max_remaining: int,
    max_new_tokens: int,
    include_overflow: bool = True,
) -> np.ndarray:
    if step < 0 or exact_max_remaining < 0 or max_new_tokens <= 0:
        raise ValueError("invalid candidate-support bounds")
    remaining = np.arange(exact_max_remaining + 1, dtype=np.float64)
    overflow = np.zeros_like(remaining)
    if include_overflow:
        remaining = np.concatenate(
            (remaining, np.asarray([exact_max_remaining + 1.0], dtype=np.float64))
        )
        overflow = np.concatenate((overflow, np.asarray([1.0], dtype=np.float64)))
    return np.column_stack(
        (
            remaining,
            np.log1p(remaining),
            remaining / max_new_tokens,
            step + remaining,
            overflow,
        )
    )


def build_bayesian_sequence(
    trace: SequentialRawTrace,
    *,
    stride: int = 5,
) -> BayesianSequence:
    """Build non-overlapping evidence blocks and an explicit censored target."""

    trace.validate(stride=stride)
    prior = shifted_lognormal_integer_prior(
        trace.prior_mu,
        trace.prior_log_variance,
        max_total_tokens=trace.max_new_tokens,
    )
    evidence_steps: list[SequentialEvidenceStep] = []
    previous_step = 0
    previous_entropy: float | None = None
    previous_eos: float | None = None
    previous_hidden = trace.initial_decode_hidden_state
    for point_index, raw_step in enumerate(trace.saved_steps):
        step = int(raw_step)
        block_entropy = np.asarray(
            trace.token_entropies[previous_step:step],
            dtype=np.float64,
        )
        block_eos = np.asarray(
            trace.token_eos_probabilities[previous_step:step],
            dtype=np.float64,
        )
        last_entropy = float(block_entropy[-1])
        last_eos = float(block_eos[-1])
        terminal = trace.terminal_observed and step == trace.observed_tokens
        scalar_features = np.asarray(
            (
                trace.prior_mu,
                trace.prior_log_variance,
                trace.prior_mean_total_tokens,
                float(step),
                float(step - previous_step),
                step / trace.max_new_tokens,
                last_entropy,
                float(block_entropy.mean()),
                _block_slope(block_entropy),
                last_eos,
                float(block_eos.max()),
                0.0 if previous_entropy is None else last_entropy - previous_entropy,
                0.0 if previous_eos is None else last_eos - previous_eos,
                float(terminal),
            ),
            dtype=np.float64,
        )
        hidden_delta = None
        if trace.decode_hidden_states is not None:
            current_hidden = np.asarray(
                trace.decode_hidden_states[point_index],
                dtype=np.float64,
            )
            hidden_delta = current_hidden - np.asarray(previous_hidden, dtype=np.float64)
            previous_hidden = current_hidden
        exact_max_remaining = trace.max_new_tokens - step
        evidence_steps.append(
            SequentialEvidenceStep(
                step=step,
                delta=step - previous_step,
                scalar_features=scalar_features,
                candidate_features=candidate_feature_matrix(
                    step=step,
                    exact_max_remaining=exact_max_remaining,
                    max_new_tokens=trace.max_new_tokens,
                ),
                true_remaining=(trace.observed_tokens - step if not trace.is_censored else None),
                censored_after_remaining=(
                    trace.observed_tokens - step if trace.is_censored else None
                ),
                terminal_observed=terminal,
                hidden_delta=hidden_delta,
            )
        )
        previous_step = step
        previous_entropy = last_entropy
        previous_eos = last_eos

    sequence = BayesianSequence(
        prompt_id=trace.prompt_id,
        prompt_family_id=trace.prompt_family_id,
        seed=trace.seed,
        max_new_tokens=trace.max_new_tokens,
        initial_log_probabilities=prior.log_probabilities,
        steps=tuple(evidence_steps),
        prior_lower_tail_mass=prior.lower_tail_mass,
        prior_upper_tail_mass=prior.upper_tail_mass,
    )
    sequence.validate()
    return sequence


def build_synthetic_sequence(
    *,
    output_tokens: int,
    max_new_tokens: int,
    prior_mu: float | None = None,
    prior_log_variance: float = 0.25,
    hidden_size: int | None = None,
    censored: bool = False,
) -> BayesianSequence:
    """Create a deterministic toy rollout for numerical and directionality tests."""

    if output_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("synthetic token counts must be positive")
    if censored:
        observed_tokens = max_new_tokens
    elif output_tokens > max_new_tokens:
        raise ValueError("an uncensored synthetic output must fit inside max_new_tokens")
    else:
        observed_tokens = output_tokens
    resolved_mu = math.log1p(output_tokens) if prior_mu is None else prior_mu
    positions = np.arange(observed_tokens, dtype=np.float64)
    progress = (positions + 1.0) / observed_tokens
    entropies = np.maximum(0.05, 3.0 - 2.0 * progress)
    eos_probabilities = np.clip(0.01 + 0.98 * np.square(progress), 0.0, 1.0)
    steps = scheduled_update_steps(
        observed_tokens,
        terminal_observed=not censored,
    )
    initial_hidden = None
    decode_hidden = None
    if hidden_size is not None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        initial_hidden = np.zeros(hidden_size, dtype=np.float64)
        base = np.linspace(0.25, 1.0, hidden_size, dtype=np.float64)
        decode_hidden = np.stack([base * (step / observed_tokens) for step in steps])
    raw = SequentialRawTrace(
        prompt_id="synthetic-prompt",
        prompt_family_id="synthetic-family",
        task="synthetic",
        intended_length="synthetic",
        temperature=0.7,
        seed=42,
        stop_reason="max_new_tokens" if censored else "eos",
        observed_tokens=observed_tokens,
        max_new_tokens=max_new_tokens,
        prior_mu=resolved_mu,
        prior_log_variance=prior_log_variance,
        prior_mean_total_tokens=max(
            0.0,
            math.expm1(resolved_mu + 0.5 * prior_log_variance),
        ),
        token_entropies=entropies,
        token_eos_probabilities=eos_probabilities,
        saved_steps=steps,
        initial_decode_hidden_state=initial_hidden,
        decode_hidden_states=decode_hidden,
    )
    return build_bayesian_sequence(raw)
