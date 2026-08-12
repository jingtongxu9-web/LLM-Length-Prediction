"""Shared likelihood-ratio scorers and rollout-balanced sequence training."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.checkpoints import atomic_torch_save, load_torch_checkpoint
from llm_length_prediction.data.sequential import (
    CANDIDATE_FEATURE_NAMES,
    SCALAR_EVIDENCE_FEATURE_NAMES,
    BayesianSequence,
)

SCALAR_METHOD_ID = "bayesian_entropy_scalar_v1"
HIDDEN_DELTA_METHOD_ID = "bayesian_entropy_hidden_delta_v1"
BAYESIAN_METHOD_IDS = (SCALAR_METHOD_ID, HIDDEN_DELTA_METHOD_ID)
BAYESIAN_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScorerStandardization:
    evidence_mean: np.ndarray
    evidence_scale: np.ndarray
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray

    def validate(self) -> None:
        expected = (
            (self.evidence_mean, len(SCALAR_EVIDENCE_FEATURE_NAMES), "evidence_mean"),
            (self.evidence_scale, len(SCALAR_EVIDENCE_FEATURE_NAMES), "evidence_scale"),
            (self.candidate_mean, len(CANDIDATE_FEATURE_NAMES), "candidate_mean"),
            (self.candidate_scale, len(CANDIDATE_FEATURE_NAMES), "candidate_scale"),
        )
        for values, size, name in expected:
            array = np.asarray(values)
            if array.shape != (size,) or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} has the wrong shape or non-finite values")
        if np.any(np.asarray(self.evidence_scale) <= 0):
            raise ValueError("evidence_scale must be positive")
        if np.any(np.asarray(self.candidate_scale) <= 0):
            raise ValueError("candidate_scale must be positive")

    def to_dict(self) -> dict[str, list[float]]:
        self.validate()
        return {
            "evidence_mean": np.asarray(self.evidence_mean).tolist(),
            "evidence_scale": np.asarray(self.evidence_scale).tolist(),
            "candidate_mean": np.asarray(self.candidate_mean).tolist(),
            "candidate_scale": np.asarray(self.candidate_scale).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScorerStandardization:
        standardization = cls(
            evidence_mean=np.asarray(payload["evidence_mean"], dtype=np.float32),
            evidence_scale=np.asarray(payload["evidence_scale"], dtype=np.float32),
            candidate_mean=np.asarray(payload["candidate_mean"], dtype=np.float32),
            candidate_scale=np.asarray(payload["candidate_scale"], dtype=np.float32),
        )
        standardization.validate()
        return standardization


def fit_scorer_standardization(
    sequences: Sequence[BayesianSequence],
) -> ScorerStandardization:
    """Fit rollout-balanced scalers without materializing all candidate rows."""

    if not sequences:
        raise ValueError("at least one sequence is required")
    evidence_sum = np.zeros(len(SCALAR_EVIDENCE_FEATURE_NAMES), dtype=np.float64)
    evidence_square_sum = np.zeros_like(evidence_sum)
    candidate_sum = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float64)
    candidate_square_sum = np.zeros_like(candidate_sum)
    evidence_weight = 0.0
    candidate_weight = 0.0
    sequence_weight = 1.0 / len(sequences)
    for sequence in sequences:
        sequence.validate()
        step_weight = sequence_weight / len(sequence.steps)
        for step in sequence.steps:
            evidence = np.asarray(step.scalar_features, dtype=np.float64)
            evidence_sum += step_weight * evidence
            evidence_square_sum += step_weight * np.square(evidence)
            evidence_weight += step_weight
            step_sum, step_square_sum = candidate_feature_moments(step)
            row_weight = step_weight / step.candidate_count
            candidate_sum += row_weight * step_sum
            candidate_square_sum += row_weight * step_square_sum
            candidate_weight += row_weight * step.candidate_count
    evidence_mean = evidence_sum / evidence_weight
    candidate_mean = candidate_sum / candidate_weight
    evidence_variance = evidence_square_sum / evidence_weight - np.square(evidence_mean)
    candidate_variance = candidate_square_sum / candidate_weight - np.square(candidate_mean)
    standardization = ScorerStandardization(
        evidence_mean=evidence_mean.astype(np.float32),
        evidence_scale=np.maximum(np.sqrt(np.maximum(evidence_variance, 0.0)), 1e-6).astype(
            np.float32
        ),
        candidate_mean=candidate_mean.astype(np.float32),
        candidate_scale=np.maximum(
            np.sqrt(np.maximum(candidate_variance, 0.0)),
            1e-6,
        ).astype(np.float32),
    )
    standardization.validate()
    return standardization


def candidate_feature_moments(step: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate column moments without retaining a support matrix."""

    remaining = np.arange(step.exact_max_remaining + 1, dtype=np.float64)
    overflow = np.zeros_like(remaining)
    if step.has_overflow:
        remaining = np.concatenate((remaining, [step.exact_max_remaining + 1.0]))
        overflow = np.concatenate((overflow, [1.0]))
    columns = (
        remaining,
        np.log1p(remaining),
        remaining / step.max_new_tokens,
        step.step + remaining,
        overflow,
    )
    return (
        np.asarray([values.sum() for values in columns], dtype=np.float64),
        np.asarray([np.square(values).sum() for values in columns], dtype=np.float64),
    )


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Bayesian scorer training requires the project's model dependencies"
        ) from error
    return torch, nn


def build_likelihood_ratio_scorer(
    method_id: str,
    *,
    standardization: ScorerStandardization,
    hidden_input_dim: int | None = None,
    hidden_projection_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    projection_seed: int = 42,
) -> Any:
    """Build a shared scorer over evidence and candidate remaining lengths."""

    if method_id not in BAYESIAN_METHOD_IDS:
        raise ValueError(f"unsupported Bayesian method_id: {method_id}")
    standardization.validate()
    uses_hidden = method_id == HIDDEN_DELTA_METHOD_ID
    if uses_hidden and (hidden_input_dim is None or hidden_input_dim <= 0):
        raise ValueError("hidden-delta scorer requires a positive hidden_input_dim")
    if not uses_hidden and hidden_input_dim is not None:
        raise ValueError("scalar scorer cannot receive hidden_input_dim")
    if hidden_projection_dim <= 0 or hidden_dim <= 0:
        raise ValueError("hidden dimensions must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    torch, nn = _torch_modules()

    evidence_mean = torch.as_tensor(standardization.evidence_mean, dtype=torch.float32)
    evidence_scale = torch.as_tensor(standardization.evidence_scale, dtype=torch.float32)
    candidate_mean = torch.as_tensor(standardization.candidate_mean, dtype=torch.float32)
    candidate_scale = torch.as_tensor(standardization.candidate_scale, dtype=torch.float32)
    projection = None
    if uses_hidden:
        generator = np.random.default_rng(projection_seed)
        projection = generator.normal(
            0.0,
            1.0 / math.sqrt(float(hidden_input_dim)),
            size=(int(hidden_input_dim), hidden_projection_dim),
        ).astype(np.float32)

    class SharedLikelihoodRatioScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("evidence_mean", evidence_mean)
            self.register_buffer("evidence_scale", evidence_scale)
            self.register_buffer("candidate_mean", candidate_mean)
            self.register_buffer("candidate_scale", candidate_scale)
            if projection is not None:
                self.register_buffer("hidden_projection", torch.from_numpy(projection))
                self.hidden_normalization = nn.LayerNorm(
                    hidden_projection_dim,
                    elementwise_affine=False,
                )
            else:
                self.register_buffer("hidden_projection", None)
                self.hidden_normalization = None
            evidence_dim = len(SCALAR_EVIDENCE_FEATURE_NAMES)
            if uses_hidden:
                evidence_dim += hidden_projection_dim
            input_dim = evidence_dim + len(CANDIDATE_FEATURE_NAMES)
            second_hidden = max(16, hidden_dim // 2)
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, second_hidden),
                nn.GELU(),
                nn.Linear(second_hidden, 1),
            )

        def forward(
            self,
            evidence_features: Any,
            candidate_features: Any,
            hidden_delta: Any | None = None,
        ) -> Any:
            if evidence_features.ndim != 2:
                raise ValueError("evidence_features must have shape (batch, evidence_dim)")
            if candidate_features.ndim != 3:
                raise ValueError("candidate_features must have shape (batch, support, dim)")
            if candidate_features.shape[0] != evidence_features.shape[0]:
                raise ValueError("evidence and candidates must have the same batch size")
            evidence = (evidence_features - self.evidence_mean) / self.evidence_scale
            if self.hidden_projection is not None:
                if hidden_delta is None or hidden_delta.ndim != 2:
                    raise ValueError("hidden-delta scorer requires batched hidden_delta")
                hidden = hidden_delta @ self.hidden_projection
                hidden = self.hidden_normalization(hidden)
                evidence = torch.cat((evidence, hidden), dim=-1)
            elif hidden_delta is not None:
                raise ValueError("scalar scorer does not accept hidden_delta")
            candidates = (candidate_features - self.candidate_mean) / self.candidate_scale
            expanded = evidence[:, None, :].expand(-1, candidates.shape[1], -1)
            network_input = torch.cat((expanded, candidates), dim=-1)
            scores = self.network(network_input).squeeze(-1)
            return scores - scores.mean(dim=-1, keepdim=True)

    model = SharedLikelihoodRatioScorer()
    model.bayesian_spec = {
        "method_id": method_id,
        "standardization": standardization.to_dict(),
        "hidden_input_dim": hidden_input_dim,
        "hidden_projection_dim": hidden_projection_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "projection_seed": projection_seed,
    }
    return model


def _torch_transition(log_posterior: Any, delta: int, *, has_overflow: bool) -> Any:
    torch, _ = _torch_modules()
    finite_size = log_posterior.numel() - int(has_overflow)
    if delta <= 0 or finite_size <= delta:
        raise ValueError("invalid Bayesian transition delta")
    exact = log_posterior[delta:finite_size]
    shifted = torch.cat((exact, log_posterior[-1:])) if has_overflow else exact
    return shifted - torch.logsumexp(shifted, dim=0)


def _step_nll(log_posterior: Any, step: Any, *, has_overflow: bool) -> Any:
    torch, _ = _torch_modules()
    finite_size = log_posterior.numel() - int(has_overflow)
    if step.true_remaining is not None:
        if step.true_remaining >= finite_size:
            raise ValueError("exact target lies outside finite posterior support")
        return -log_posterior[step.true_remaining]
    threshold = int(step.censored_after_remaining)
    eligible = log_posterior[threshold + 1 : finite_size]
    if has_overflow:
        eligible = torch.cat((eligible, log_posterior[-1:]))
    if eligible.numel() == 0:
        raise ValueError("right-censored target has no survival support")
    return -torch.logsumexp(eligible, dim=0)


def bayesian_sequence_loss(
    scorer: Any,
    sequences: Sequence[BayesianSequence],
    *,
    terminal_bce_weight: float = 0.1,
    stability_weight: float = 0.01,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Differentiate through every update with rollout-balanced sequence batching.

    Sequences at the same saved step share one scorer call. This is mathematically
    identical to processing rollouts independently, but it avoids millions of tiny
    GPU launches during the full Stage-5 OOF run.
    """

    if not sequences:
        raise ValueError("at least one Bayesian sequence is required")
    if terminal_bce_weight < 0 or stability_weight < 0:
        raise ValueError("loss weights cannot be negative")
    torch, _ = _torch_modules()
    for sequence in sequences:
        sequence.validate()
    method_id = getattr(scorer, "bayesian_spec", {}).get("method_id")
    if method_id not in BAYESIAN_METHOD_IDS:
        raise ValueError("scorer is missing a recognized Bayesian method_id")
    uses_hidden = method_id == HIDDEN_DELTA_METHOD_ID
    log_posteriors = [
        torch.as_tensor(
            sequence.initial_log_probabilities,
            dtype=torch.float32,
            device=device,
        )
        for sequence in sequences
    ]
    zero = log_posteriors[0].new_zeros(())
    nll_sums = [zero for _ in sequences]
    terminal_sums = [zero for _ in sequences]
    stability_sums = [zero for _ in sequences]
    positions = [0 for _ in sequences]
    while True:
        groups: dict[tuple[int, int, int, bool], list[int]] = defaultdict(list)
        for index, sequence in enumerate(sequences):
            if positions[index] < len(sequence.steps):
                step = sequence.steps[positions[index]]
                groups[
                    (step.step, step.delta, step.candidate_count, sequence.has_overflow)
                ].append(index)
        if not groups:
            break
        for (_, delta, _, has_overflow), indices in groups.items():
            previous = torch.stack([log_posteriors[index] for index in indices])
            finite_size = previous.shape[1] - int(has_overflow)
            exact = previous[:, delta:finite_size]
            shifted = (
                torch.cat((exact, previous[:, -1:]), dim=1)
                if has_overflow
                else exact
            )
            log_predictive = shifted - torch.logsumexp(shifted, dim=1, keepdim=True)
            steps = [sequences[index].steps[positions[index]] for index in indices]
            evidence = torch.as_tensor(
                np.stack([step.scalar_features for step in steps]),
                dtype=torch.float32,
                device=device,
            )
            one_candidate_matrix = torch.as_tensor(
                steps[0].candidate_features[None, :, :],
                dtype=torch.float32,
                device=device,
            )
            candidates = one_candidate_matrix.expand(len(indices), -1, -1)
            hidden = None
            if uses_hidden:
                if any(step.hidden_delta is None for step in steps):
                    raise ValueError("hidden-delta scorer requires hidden evidence")
                hidden = torch.as_tensor(
                    np.stack([step.hidden_delta for step in steps]),
                    dtype=torch.float32,
                    device=device,
                )
            scores = scorer(evidence, candidates, hidden)
            posterior = torch.log_softmax(log_predictive + scores, dim=1)
            for row, (index, step) in enumerate(zip(indices, steps, strict=True)):
                log_posteriors[index] = posterior[row]
                nll_sums[index] = nll_sums[index] + _step_nll(
                    posterior[row], step, has_overflow=has_overflow
                )
                probability_zero = torch.exp(posterior[row, 0]).clamp(
                    1e-7, 1.0 - 1e-7
                )
                terminal_target = posterior.new_tensor(float(step.terminal_observed))
                terminal_sums[index] = terminal_sums[index] + (
                    -terminal_target * torch.log(probability_zero)
                    - (1.0 - terminal_target) * torch.log1p(-probability_zero)
                )
                stability_sums[index] = stability_sums[index] + 0.5 * torch.abs(
                    torch.exp(posterior[row]) - torch.exp(log_predictive[row])
                ).sum()
                positions[index] += 1
    counts = [len(sequence.steps) for sequence in sequences]
    sequence_nlls = [value / count for value, count in zip(nll_sums, counts, strict=True)]
    sequence_terminals = [
        value / count for value, count in zip(terminal_sums, counts, strict=True)
    ]
    sequence_stabilities = [
        value / count for value, count in zip(stability_sums, counts, strict=True)
    ]
    sequence_totals = [
        nll + terminal_bce_weight * terminal + stability_weight * stability
        for nll, terminal, stability in zip(
            sequence_nlls, sequence_terminals, sequence_stabilities, strict=True
        )
    ]
    components = {
        "posterior_nll": torch.stack(sequence_nlls).mean(),
        "terminal_bce": torch.stack(sequence_terminals).mean(),
        "posterior_total_variation": torch.stack(sequence_stabilities).mean(),
    }
    return torch.stack(sequence_totals).mean(), components


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def fit_bayesian_scorer(
    sequences: Sequence[BayesianSequence],
    *,
    method_id: str,
    hidden_projection_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    terminal_bce_weight: float = 0.1,
    stability_weight: float = 0.01,
    sequence_batch_size: int | None = None,
    seed: int = 42,
    device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    """Fit a scorer with deterministic full-sequence AdamW updates."""

    if not sequences or epochs <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid Bayesian scorer training inputs")
    if sequence_batch_size is not None and sequence_batch_size <= 0:
        raise ValueError("sequence_batch_size must be positive when provided")
    for sequence in sequences:
        sequence.validate()
    hidden_input_dim = None
    if method_id == HIDDEN_DELTA_METHOD_ID:
        first_hidden = sequences[0].steps[0].hidden_delta
        if first_hidden is None:
            raise ValueError("hidden-delta training requires hidden features")
        hidden_input_dim = len(first_hidden)
        if any(
            step.hidden_delta is None or len(step.hidden_delta) != hidden_input_dim
            for sequence in sequences
            for step in sequence.steps
        ):
            raise ValueError("all hidden deltas must share one dimension")
    elif method_id == SCALAR_METHOD_ID:
        # Unified traces retain hidden deltas for both candidates. The scalar
        # scorer ignores them explicitly rather than copying every long sequence.
        pass
    else:
        raise ValueError(f"unsupported Bayesian method_id: {method_id}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch, _ = _torch_modules()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    resolved_device = _resolve_device(torch, device)
    standardization = fit_scorer_standardization(sequences)
    scorer = build_likelihood_ratio_scorer(
        method_id,
        standardization=standardization,
        hidden_input_dim=hidden_input_dim,
        hidden_projection_dim=hidden_projection_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        projection_seed=seed,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )
    epoch_losses = []
    effective_batch_size = min(sequence_batch_size or len(sequences), len(sequences))
    for epoch in range(epochs):
        scorer.train()
        generator = np.random.default_rng(seed + epoch)
        order = generator.permutation(len(sequences))
        weighted_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), effective_batch_size):
            indices = order[start : start + effective_batch_size]
            batch = [sequences[int(index)] for index in indices]
            loss, _ = bayesian_sequence_loss(
                scorer,
                batch,
                terminal_bce_weight=terminal_bce_weight,
                stability_weight=stability_weight,
                device=resolved_device,
            )
            # Accumulating rollout-weighted gradients preserves the original
            # full-training-set objective and one AdamW update per epoch while
            # releasing each mini-batch computation graph immediately.
            (loss * (len(batch) / len(sequences))).backward()
            weighted_loss += float(loss.detach().cpu().item()) * len(batch)
        optimizer.step()
        epoch_losses.append(weighted_loss / len(sequences))
    scorer.eval()
    component_sums = {
        "posterior_nll": 0.0,
        "terminal_bce": 0.0,
        "posterior_total_variation": 0.0,
    }
    final_loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(sequences), effective_batch_size):
            batch = sequences[start : start + effective_batch_size]
            batch_loss, batch_components = bayesian_sequence_loss(
                scorer,
                batch,
                terminal_bce_weight=terminal_bce_weight,
                stability_weight=stability_weight,
                device=resolved_device,
            )
            final_loss_sum += float(batch_loss.cpu().item()) * len(batch)
            for name, value in batch_components.items():
                component_sums[name] += float(value.cpu().item()) * len(batch)
    report = {
        "framework": "pytorch",
        "torch_version": str(torch.__version__),
        "method_id": method_id,
        "device": resolved_device,
        "seed": seed,
        "epochs": epochs,
        "optimizer_steps": epochs,
        "gradient_accumulation": "rollout_weighted_full_dataset_per_epoch",
        "sequence_batch_size": effective_batch_size,
        "rollout_count": len(sequences),
        "epoch_losses": epoch_losses,
        "final_loss": final_loss_sum / len(sequences),
        **{name: value / len(sequences) for name, value in component_sums.items()},
    }
    return scorer, report


def make_bayesian_checkpoint(
    scorer: Any,
    *,
    contract_sha256: str,
    training_report: Mapping[str, Any],
) -> dict[str, Any]:
    spec = getattr(scorer, "bayesian_spec", None)
    if not isinstance(spec, dict) or spec.get("method_id") not in BAYESIAN_METHOD_IDS:
        raise ValueError("scorer is missing a valid Bayesian specification")
    if len(contract_sha256) != 64:
        raise ValueError("contract_sha256 must be a 64-character digest")
    return {
        "schema_version": BAYESIAN_CHECKPOINT_SCHEMA_VERSION,
        "model_type": "shared_incremental_likelihood_ratio_scorer",
        "method_id": spec["method_id"],
        "contract_sha256": contract_sha256,
        "scorer_spec": spec,
        "state_dict": scorer.state_dict(),
        "training_report": dict(training_report),
    }


def validate_bayesian_checkpoint(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != BAYESIAN_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Bayesian checkpoint schema_version")
    if payload.get("model_type") != "shared_incremental_likelihood_ratio_scorer":
        raise ValueError("unsupported Bayesian checkpoint model_type")
    if payload.get("method_id") not in BAYESIAN_METHOD_IDS:
        raise ValueError("unsupported Bayesian checkpoint method_id")
    digest = payload.get("contract_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Bayesian checkpoint is missing the contract digest")
    spec = payload.get("scorer_spec")
    if not isinstance(spec, Mapping) or spec.get("method_id") != payload.get("method_id"):
        raise ValueError("Bayesian checkpoint scorer_spec is inconsistent")
    if not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("Bayesian checkpoint is missing state_dict")


def restore_bayesian_scorer(payload: Mapping[str, Any], *, device: str = "cpu") -> Any:
    validate_bayesian_checkpoint(payload)
    spec = payload["scorer_spec"]
    standardization = ScorerStandardization.from_dict(spec["standardization"])
    scorer = build_likelihood_ratio_scorer(
        str(spec["method_id"]),
        standardization=standardization,
        hidden_input_dim=(
            None if spec["hidden_input_dim"] is None else int(spec["hidden_input_dim"])
        ),
        hidden_projection_dim=int(spec["hidden_projection_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        dropout=float(spec["dropout"]),
        projection_seed=int(spec["projection_seed"]),
    )
    scorer.load_state_dict(payload["state_dict"])
    return scorer.to(device)


def save_bayesian_checkpoint(payload: dict[str, Any], path: str | Path) -> Path:
    validate_bayesian_checkpoint(payload)
    output = Path(path)
    atomic_torch_save(payload, output)
    return output


def load_bayesian_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = load_torch_checkpoint(Path(path))
    validate_bayesian_checkpoint(payload)
    return payload
