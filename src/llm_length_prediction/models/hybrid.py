"""Leakage-safe ALPS prior and progressive heads for Hybrid v3."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from llm_length_prediction.data.hybrid import HybridV3Trace


@dataclass(frozen=True)
class HybridSample:
    prompt_id: str
    prompt_family_id: str
    task: str
    intended_length: str
    seed: int
    step: int
    output_tokens: int
    remaining_tokens: int
    prior_feature: np.ndarray
    prompt_feature: np.ndarray
    decode_feature: np.ndarray
    dynamic_features: tuple[float, float, float, float, float]
    sequence_weight: float

    @property
    def trace_key(self) -> tuple[str, int]:
        return self.prompt_id, self.seed

    @property
    def plp_features(self) -> np.ndarray:
        return np.concatenate((self.prompt_feature, self.decode_feature)).astype(
            np.float32, copy=False
        )


def build_hybrid_samples(
    trace: HybridV3Trace,
    *,
    prompt_family_id: str,
    intended_length: str,
) -> list[HybridSample]:
    trace.validate()
    weight = 1.0 / len(trace.steps)
    return [
        HybridSample(
            prompt_id=trace.prompt_id,
            prompt_family_id=prompt_family_id,
            task=trace.task,
            intended_length=intended_length,
            seed=trace.seed,
            step=int(step),
            output_tokens=trace.output_tokens,
            remaining_tokens=int(remaining),
            prior_feature=trace.prior_feature,
            prompt_feature=trace.prompt_feature,
            decode_feature=decode,
            dynamic_features=(
                float(step),
                float(trace.entropies[index]),
                float(trace.entropy_means[index]),
                float(trace.entropy_slopes[index]),
                float(trace.eos_probabilities[index]),
            ),
            sequence_weight=weight,
        )
        for index, (step, remaining, decode) in enumerate(
            zip(
                trace.steps,
                trace.remaining_lengths,
                trace.decode_hidden_states,
                strict=True,
            )
        )
    ]


@dataclass(frozen=True)
class WeightedLogRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float
    residual_variance: float
    target: str

    def predict_mu(self, features: np.ndarray) -> np.ndarray:
        return self.bias + ((features - self.feature_mean) / self.feature_scale) @ self.weights

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        return np.maximum(
            0.0,
            np.expm1(self.predict_mu(features) + 0.5 * self.residual_variance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_type": "standardized_weighted_ridge_shifted_lognormal",
            "target": self.target,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "residual_variance": self.residual_variance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WeightedLogRidge:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported Hybrid Ridge schema")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            residual_variance=float(payload["residual_variance"]),
            target=str(payload["target"]),
        )


def fit_log1p_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    target_name: str,
) -> WeightedLogRidge:
    if features.ndim != 2 or targets.shape != (len(features),):
        raise ValueError("Ridge features and targets are not aligned")
    if weights.shape != targets.shape or np.any(weights <= 0) or np.any(targets < 0):
        raise ValueError("Ridge weights/targets are invalid")
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("Hybrid v3 requires scikit-learn") from error
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features, sample_weight=weights)
    transformed = np.log1p(targets.astype(np.float64))
    ridge = Ridge(alpha=alpha, solver="lsqr")
    ridge.fit(scaled, transformed, sample_weight=weights)
    residuals = transformed - ridge.predict(scaled)
    variance = float(np.sum(weights * residuals**2) / weights.sum())
    return WeightedLogRidge(
        feature_mean=np.asarray(scaler.mean_, dtype=np.float64),
        feature_scale=np.asarray(scaler.scale_, dtype=np.float64),
        weights=np.asarray(ridge.coef_, dtype=np.float64),
        bias=float(ridge.intercept_),
        residual_variance=variance,
        target=target_name,
    )


def _trace_representatives(samples: Sequence[HybridSample]) -> list[HybridSample]:
    representatives: dict[tuple[str, int], HybridSample] = {}
    for sample in samples:
        representatives.setdefault(sample.trace_key, sample)
    return [representatives[key] for key in sorted(representatives)]


def fit_alps_prior(samples: Sequence[HybridSample], *, alpha: float = 1.0) -> WeightedLogRidge:
    representatives = _trace_representatives(samples)
    return fit_log1p_ridge(
        np.stack([sample.prior_feature for sample in representatives]),
        np.asarray([sample.output_tokens for sample in representatives], dtype=np.float64),
        np.ones(len(representatives), dtype=np.float64),
        alpha=alpha,
        target_name="log1p_output_tokens",
    )


def alps_prior_summaries(model: WeightedLogRidge, samples: Sequence[HybridSample]) -> np.ndarray:
    matrix = np.stack([sample.prior_feature for sample in samples])
    mu = model.predict_mu(matrix)
    variance = np.full(len(samples), model.residual_variance, dtype=np.float64)
    mean_total = np.maximum(0.0, np.expm1(mu + 0.5 * variance))
    median_total = np.maximum(0.0, np.expm1(mu))
    steps = np.asarray([sample.step for sample in samples], dtype=np.float64)
    return np.column_stack(
        (
            mu,
            variance,
            mean_total,
            np.maximum(mean_total - steps, 0.0),
            np.maximum(median_total - steps, 0.0),
        )
    ).astype(np.float32)


def cross_fitted_prior_summaries(
    samples: Sequence[HybridSample],
    family_folds: Mapping[str, int],
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    fold_values = sorted(set(family_folds.values()))
    output = np.full((len(samples), 5), np.nan, dtype=np.float32)
    for fold in fold_values:
        train = [sample for sample in samples if family_folds[sample.prompt_family_id] != fold]
        indices = [
            index
            for index, sample in enumerate(samples)
            if family_folds[sample.prompt_family_id] == fold
        ]
        model = fit_alps_prior(train, alpha=alpha)
        output[indices] = alps_prior_summaries(model, [samples[index] for index in indices])
    if np.any(~np.isfinite(output)):
        raise RuntimeError("cross-fitted ALPS prior summaries are incomplete")
    return output


@dataclass(frozen=True)
class SummaryScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SummaryScaler:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
        )


def fit_summary_scaler(values: np.ndarray, weights: np.ndarray) -> SummaryScaler:
    normalized = weights / weights.sum()
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum((values - mean) ** 2 * normalized[:, None], axis=0)
    return SummaryScaler(
        mean=mean.astype(np.float32),
        scale=np.maximum(np.sqrt(variance), 1e-6).astype(np.float32),
    )


def hybrid_feature_matrix(
    samples: Sequence[HybridSample],
    prior_summaries: np.ndarray,
    *,
    scaler: SummaryScaler,
) -> np.ndarray:
    if prior_summaries.shape != (len(samples), 5):
        raise ValueError("prior summaries do not align with Hybrid samples")
    plp = np.stack([sample.plp_features for sample in samples]).astype(np.float32)
    return np.concatenate((plp, scaler.transform(prior_summaries)), axis=1)


def target_range(
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    percentiles: tuple[float, float],
    positive_only: bool,
    weighted: bool,
) -> tuple[float, float]:
    values = targets.astype(np.float64)
    sample_weights = weights.astype(np.float64)
    if positive_only:
        mask = values > 0
        values = values[mask]
        sample_weights = sample_weights[mask]
    if not len(values):
        raise ValueError("target range requires at least one eligible target")
    if weighted:
        order = np.argsort(values, kind="stable")
        values = values[order]
        sample_weights = sample_weights[order]
        cumulative = np.cumsum(sample_weights) - 0.5 * sample_weights
        cumulative /= sample_weights.sum()
        lower, upper = np.interp(
            np.asarray(percentiles) / 100.0,
            cumulative,
            values,
            left=values[0],
            right=values[-1],
        )
    else:
        lower, upper = np.percentile(values, percentiles)
    lower = max(0.0, float(lower))
    upper = float(upper)
    return (lower, upper if upper > lower else lower + 1.0)


def bin_centers(
    value_range: tuple[float, float], num_bins: int, *, terminal_zero: bool
) -> np.ndarray:
    lower, upper = value_range
    positive_bins = num_bins - 1 if terminal_zero else num_bins
    if num_bins <= 1 or lower < 0 or not upper > lower or positive_bins <= 0:
        raise ValueError("invalid length-bin configuration")
    width = (upper - lower) / positive_bins
    positive = lower + (np.arange(positive_bins, dtype=np.float32) + 0.5) * width
    return (
        np.concatenate((np.asarray([0.0], dtype=np.float32), positive))
        if terminal_zero
        else positive
    )


def soft_labels(
    targets: np.ndarray,
    value_range: tuple[float, float],
    num_bins: int,
    *,
    terminal_zero: bool,
) -> np.ndarray:
    lower, upper = value_range
    positive_bins = num_bins - 1 if terminal_zero else num_bins
    width = (upper - lower) / positive_bins
    clipped = np.clip(targets, lower, upper)
    bins = np.floor((clipped - lower) / width).astype(np.int64)
    bins = np.clip(bins, 0, positive_bins - 1)
    if terminal_zero:
        bins += 1
        bins[targets == 0] = 0
    logits = -np.abs(np.arange(num_bins)[None, :] - bins[:, None]).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    if terminal_zero:
        terminal = targets == 0
        probabilities[~terminal, 0] = 0.0
        probabilities[terminal] = 0.0
        probabilities[terminal, 0] = 1.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def build_progressive_head(
    input_dim: int,
    *,
    hidden_dim: int,
    num_bins: int,
    value_range: tuple[float, float],
    terminal_zero: bool,
    dropout: float,
) -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("Hybrid progressive heads require PyTorch") from error
    centers = torch.from_numpy(bin_centers(value_range, num_bins, terminal_zero=terminal_zero))

    class ProgressiveHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_bins),
            )
            self.register_buffer("bin_centers", centers)

        def forward(self, features: Any) -> tuple[Any, Any]:
            logits = self.network(features)
            probabilities = torch.softmax(logits, dim=-1)
            return logits, (probabilities * self.bin_centers).sum(dim=-1)

    return ProgressiveHead()


def fit_progressive_head(
    samples: Sequence[HybridSample],
    features: np.ndarray,
    *,
    hidden_dim: int,
    num_bins: int,
    percentiles: tuple[float, float],
    terminal_zero: bool,
    weighted_range: bool,
    lambda_ce: float,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    if features.shape[0] != len(samples) or features.ndim != 2:
        raise ValueError("progressive feature matrix does not align")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("Hybrid progressive training requires PyTorch") from error
    targets = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float32)
    weights = np.asarray([sample.sequence_weight for sample in samples], dtype=np.float32)
    weights *= len(weights) / weights.sum()
    value_range = target_range(
        targets,
        weights,
        percentiles=percentiles,
        positive_only=terminal_zero,
        weighted=weighted_range,
    )
    labels = soft_labels(targets, value_range, num_bins, terminal_zero=terminal_zero)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved == "auto":
        resolved = "cpu"
    head = build_progressive_head(
        features.shape[1],
        hidden_dim=hidden_dim,
        num_bins=num_bins,
        value_range=value_range,
        terminal_zero=terminal_zero,
        dropout=dropout,
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=False
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features.astype(np.float32, copy=False)),
            torch.from_numpy(targets),
            torch.from_numpy(labels),
            torch.from_numpy(weights),
        ),
        batch_size=min(batch_size, len(samples)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    history = []
    for _ in range(epochs):
        head.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for batch_features, batch_targets, batch_labels, batch_weights in loader:
            batch_features = batch_features.to(resolved)
            batch_targets = batch_targets.to(resolved)
            batch_labels = batch_labels.to(resolved)
            batch_weights = batch_weights.to(resolved)
            optimizer.zero_grad(set_to_none=True)
            logits, predictions = head(batch_features)
            ce = -(batch_labels * functional.log_softmax(logits, dim=-1)).sum(dim=-1)
            mse = (predictions - batch_targets).square()
            point_loss = lambda_ce * ce + (1.0 - lambda_ce) * mse
            loss = (batch_weights * point_loss).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Hybrid training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += float((batch_weights * point_loss.detach()).sum().item())
            weight_sum += float(batch_weights.sum().item())
        history.append(loss_sum / weight_sum)
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    return head, {
        "device": resolved,
        "input_dim": features.shape[1],
        "hidden_dim": hidden_dim,
        "num_bins": num_bins,
        "target_range": list(value_range),
        "terminal_zero_bin": terminal_zero,
        "weighted_target_range": weighted_range,
        "trainable_parameter_count": parameter_count,
        "epoch_losses": history,
        "final_loss": history[-1],
    }


def predict_progressive_head(
    head: Any,
    features: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Hybrid inference requires PyTorch") from error
    head.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            _, predictions = head(torch.from_numpy(features[start : start + batch_size]).to(device))
            outputs.append(predictions.cpu().numpy())
    return np.concatenate(outputs)
