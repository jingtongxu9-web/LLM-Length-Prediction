"""Conservative ALPS baseline plus progress-gated PLP correction (Hybrid v2.1)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.checkpoints import atomic_torch_save, load_torch_checkpoint
from llm_length_prediction.models.hybrid import (
    HybridSample,
    WeightedLogRidge,
    alps_prior_summaries,
    fit_alps_prior,
)
from llm_length_prediction.models.hybrid_versions import (
    ControlScaler,
    fit_control_scaler,
    residual_control_matrix,
    residual_feature_matrix,
)

METHOD_ID = "alps_plp_gated_residual_v2_1"
SCALAR_RIDGE_ID = "alps_scalar_residual_ridge"


def _weights(samples: Sequence[HybridSample]) -> np.ndarray:
    return np.asarray([sample.sequence_weight for sample in samples], dtype=np.float32)


def progress_values(samples: Sequence[HybridSample], base_countdown: np.ndarray) -> np.ndarray:
    if base_countdown.shape != (len(samples),):
        raise ValueError("ALPS countdown does not align with samples")
    steps = np.asarray([sample.step for sample in samples], dtype=np.float32)
    return np.clip(steps / np.maximum(steps + base_countdown, 1.0), 0.0, 1.0).astype(
        np.float32
    )


def correction_bounds(
    base_countdown: np.ndarray, *, fraction: float, minimum_tokens: float
) -> np.ndarray:
    if fraction <= 0 or minimum_tokens <= 0 or np.any(base_countdown < 0):
        raise ValueError("invalid gated-correction bounds")
    return np.maximum(minimum_tokens, fraction * (base_countdown + 1.0)).astype(np.float32)


@dataclass(frozen=True)
class WeightedLinearRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.bias + ((features - self.feature_mean) / self.feature_scale) @ self.weights


def fit_scalar_residual_ridge(
    samples: Sequence[HybridSample],
    prior_summaries: np.ndarray,
    *,
    alpha: float,
) -> WeightedLinearRidge:
    """Diagnose whether six cheap controls can predict the signed ALPS error."""

    if prior_summaries.shape != (len(samples), 5):
        raise ValueError("scalar residual prior summaries do not align")
    if alpha < 0:
        raise ValueError("scalar residual Ridge alpha must be non-negative")
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("scalar residual Ridge requires scikit-learn") from error
    features = residual_control_matrix(samples, prior_summaries).astype(np.float64)
    targets = np.asarray(
        [sample.remaining_tokens for sample in samples], dtype=np.float64
    ) - prior_summaries[:, 3].astype(np.float64)
    weights = _weights(samples).astype(np.float64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features, sample_weight=weights)
    ridge = Ridge(alpha=alpha, solver="lsqr")
    ridge.fit(scaled, targets, sample_weight=weights)
    return WeightedLinearRidge(
        feature_mean=np.asarray(scaler.mean_, dtype=np.float64),
        feature_scale=np.asarray(scaler.scale_, dtype=np.float64),
        weights=np.asarray(ridge.coef_, dtype=np.float64),
        bias=float(ridge.intercept_),
    )


def predict_scalar_residual_ridge(
    model: WeightedLinearRidge,
    samples: Sequence[HybridSample],
    prior_summaries: np.ndarray,
) -> np.ndarray:
    controls = residual_control_matrix(samples, prior_summaries).astype(np.float64)
    return np.maximum(prior_summaries[:, 3] + model.predict(controls), 0.0)


def build_gated_residual_head(
    input_dim: int,
    *,
    hidden_dim: int,
    dropout: float,
    gate_initial_bias: float,
) -> Any:
    """Use a small correction backbone and a separate six-control terminal classifier."""

    if input_dim <= 6:
        raise ValueError("gated residual input must contain PLP state plus six controls")
    try:
        from torch import nn
    except ImportError as error:
        raise RuntimeError("gated residual training requires PyTorch") from error

    class GatedResidualHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.raw_correction = nn.Linear(hidden_dim, 1)
            self.gate = nn.Linear(hidden_dim, 1)
            self.terminal = nn.Linear(6, 1)
            nn.init.zeros_(self.raw_correction.weight)
            nn.init.zeros_(self.raw_correction.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_initial_bias)
            nn.init.zeros_(self.terminal.weight)
            nn.init.constant_(self.terminal.bias, -4.0)

        def forward(
            self,
            features: Any,
            progress: Any,
            correction_bound: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            import torch

            encoded = self.backbone(features)
            bounded = correction_bound * torch.tanh(
                self.raw_correction(encoded).squeeze(-1)
            )
            gate_confidence = torch.sigmoid(self.gate(encoded).squeeze(-1))
            gate = progress * gate_confidence
            applied = gate * bounded
            terminal_logit = self.terminal(features[:, -6:]).squeeze(-1)
            return applied, gate, gate_confidence, bounded, terminal_logit

    return GatedResidualHead()


def fit_gated_residual_head(
    samples: Sequence[HybridSample],
    features: np.ndarray,
    base_countdown: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    gate_initial_bias: float,
    correction_bound_fraction: float,
    minimum_correction_bound_tokens: float,
    correction_penalty: float,
    gate_penalty: float,
    terminal_loss_weight: float,
    terminal_threshold: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    if features.ndim != 2 or features.shape[0] != len(samples):
        raise ValueError("gated residual feature matrix does not align")
    if any(value < 0 for value in (correction_penalty, gate_penalty, terminal_loss_weight)):
        raise ValueError("gated residual loss weights must be non-negative")
    if not 0.0 < terminal_threshold < 1.0:
        raise ValueError("terminal threshold must be between zero and one")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("gated residual training requires PyTorch") from error

    targets = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float32)
    weights = _weights(samples)
    weights *= len(weights) / weights.sum()
    terminal = (targets == 0).astype(np.float32)
    nonterminal = terminal == 0
    if not np.any(nonterminal) or not np.any(~nonterminal):
        raise ValueError("gated residual training requires terminal and non-terminal points")
    progress = progress_values(samples, base_countdown)
    bounds = correction_bounds(
        base_countdown,
        fraction=correction_bound_fraction,
        minimum_tokens=minimum_correction_bound_tokens,
    )
    residual = targets - base_countdown.astype(np.float32)
    residual_scale = float(
        np.sqrt(
            np.sum(weights[nonterminal] * residual[nonterminal] ** 2)
            / weights[nonterminal].sum()
        )
    )
    residual_scale = max(residual_scale, 1.0)
    terminal_positive_weight = float(
        weights[nonterminal].sum() / weights[~nonterminal].sum()
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved == "auto":
        resolved = "cpu"
    head = build_gated_residual_head(
        features.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        gate_initial_bias=gate_initial_bias,
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=False
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features.astype(np.float32, copy=False)),
            torch.from_numpy(base_countdown.astype(np.float32, copy=False)),
            torch.from_numpy(targets),
            torch.from_numpy(progress),
            torch.from_numpy(bounds),
            torch.from_numpy(terminal),
            torch.from_numpy(weights),
        ),
        batch_size=min(batch_size, len(samples)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    pos_weight = torch.tensor(
        terminal_positive_weight, dtype=torch.float32, device=resolved
    )
    history: list[dict[str, float]] = []
    for _ in range(epochs):
        head.train()
        totals = {"loss": 0.0, "prediction": 0.0, "correction": 0.0, "gate": 0.0}
        weight_sum = 0.0
        for batch in loader:
            (
                batch_features,
                batch_base,
                batch_targets,
                batch_progress,
                batch_bounds,
                batch_terminal,
                batch_weights,
            ) = [value.to(resolved) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            applied, gate, _, _, terminal_logits = head(
                batch_features, batch_progress, batch_bounds
            )
            predicted = torch.clamp(batch_base + applied, min=0.0)
            eligible = batch_terminal == 0
            if bool(eligible.any().item()):
                standardized_error = (
                    predicted[eligible] - batch_targets[eligible]
                ) / residual_scale
                prediction_loss = functional.smooth_l1_loss(
                    standardized_error,
                    torch.zeros_like(standardized_error),
                    reduction="none",
                )
                eligible_weights = batch_weights[eligible]
                prediction_loss = (
                    eligible_weights * prediction_loss
                ).sum() / eligible_weights.sum()
                correction_loss = (
                    eligible_weights * (applied[eligible] / residual_scale).square()
                ).sum() / eligible_weights.sum()
                gate_loss = (
                    eligible_weights * gate[eligible]
                ).sum() / eligible_weights.sum()
            else:
                prediction_loss = applied.sum() * 0.0
                correction_loss = applied.sum() * 0.0
                gate_loss = gate.sum() * 0.0
            terminal_loss = functional.binary_cross_entropy_with_logits(
                terminal_logits,
                batch_terminal,
                reduction="none",
                pos_weight=pos_weight,
            )
            terminal_loss = (batch_weights * terminal_loss).sum() / batch_weights.sum()
            loss = (
                prediction_loss
                + correction_penalty * correction_loss
                + gate_penalty * gate_loss
                + terminal_loss_weight * terminal_loss
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("gated residual training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            batch_weight = float(batch_weights.sum().item())
            totals["loss"] += float(loss.detach().item()) * batch_weight
            totals["prediction"] += float(prediction_loss.detach().item()) * batch_weight
            totals["correction"] += float(correction_loss.detach().item()) * batch_weight
            totals["gate"] += float(gate_loss.detach().item()) * batch_weight
            weight_sum += batch_weight
        history.append({name: value / weight_sum for name, value in totals.items()})
    return head, {
        "method_id": METHOD_ID,
        "device": resolved,
        "input_dim": features.shape[1],
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "gate_initial_bias": gate_initial_bias,
        "correction_bound_fraction": correction_bound_fraction,
        "minimum_correction_bound_tokens": minimum_correction_bound_tokens,
        "correction_penalty": correction_penalty,
        "gate_penalty": gate_penalty,
        "terminal_loss_weight": terminal_loss_weight,
        "terminal_positive_weight": terminal_positive_weight,
        "terminal_threshold": terminal_threshold,
        "residual_scale_tokens": residual_scale,
        "trainable_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "epoch_history": history,
        "final_loss": history[-1]["loss"],
    }


def predict_gated_residual_head(
    head: Any,
    features: np.ndarray,
    base_countdown: np.ndarray,
    progress: np.ndarray,
    *,
    correction_bound_fraction: float,
    minimum_correction_bound_tokens: float,
    terminal_threshold: float,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("gated residual inference requires PyTorch") from error
    bounds = correction_bounds(
        base_countdown,
        fraction=correction_bound_fraction,
        minimum_tokens=minimum_correction_bound_tokens,
    )
    head.eval()
    applied_values: list[np.ndarray] = []
    gate_values: list[np.ndarray] = []
    gate_confidence_values: list[np.ndarray] = []
    bounded_values: list[np.ndarray] = []
    terminal_values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            applied, gate, gate_confidence, bounded, terminal_logit = head(
                torch.from_numpy(features[start : start + batch_size]).to(device),
                torch.from_numpy(progress[start : start + batch_size]).to(device),
                torch.from_numpy(bounds[start : start + batch_size]).to(device),
            )
            applied_values.append(applied.cpu().numpy())
            gate_values.append(gate.cpu().numpy())
            gate_confidence_values.append(gate_confidence.cpu().numpy())
            bounded_values.append(bounded.cpu().numpy())
            terminal_values.append(torch.sigmoid(terminal_logit).cpu().numpy())
    applied = np.concatenate(applied_values)
    gate = np.concatenate(gate_values)
    gate_confidence = np.concatenate(gate_confidence_values)
    bounded = np.concatenate(bounded_values)
    terminal_probability = np.concatenate(terminal_values)
    prediction = np.maximum(base_countdown + applied, 0.0)
    prediction[terminal_probability >= terminal_threshold] = 0.0
    return prediction, {
        "applied_correction": applied,
        "gate": gate,
        "gate_confidence": gate_confidence,
        "bounded_correction": bounded,
        "terminal_probability": terminal_probability,
    }


@dataclass
class FittedGatedResidual:
    alps_prior: WeightedLogRidge
    head: Any
    metadata: dict[str, Any]
    control_scaler: ControlScaler


def fit_gated_residual(
    samples: Sequence[HybridSample],
    cross_fitted_prior: np.ndarray,
    *,
    method_config: dict[str, Any],
    protocol: dict[str, Any],
    device: str,
) -> FittedGatedResidual:
    if not samples or cross_fitted_prior.shape != (len(samples), 5):
        raise ValueError("gated residual training inputs are empty or misaligned")
    settings = protocol["method"]
    training = protocol["training"]
    alps_prior = fit_alps_prior(
        samples, alpha=float(method_config["stacking"]["alps_ridge_alpha"])
    )
    controls = residual_control_matrix(samples, cross_fitted_prior)
    scaler = fit_control_scaler(controls, _weights(samples))
    features = residual_feature_matrix(samples, cross_fitted_prior, scaler=scaler)
    head, metadata = fit_gated_residual_head(
        samples,
        features,
        cross_fitted_prior[:, 3],
        hidden_dim=int(settings["hidden_dim"]),
        dropout=float(settings["dropout"]),
        gate_initial_bias=float(settings["gate_initial_bias"]),
        correction_bound_fraction=float(settings["correction_bound_fraction"]),
        minimum_correction_bound_tokens=float(
            settings["minimum_correction_bound_tokens"]
        ),
        correction_penalty=float(settings["correction_penalty"]),
        gate_penalty=float(settings["gate_penalty"]),
        terminal_loss_weight=float(settings["terminal_loss_weight"]),
        terminal_threshold=float(settings["terminal_threshold"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        seed=int(training["seed"]),
        device=device,
    )
    return FittedGatedResidual(
        alps_prior=alps_prior, head=head, metadata=metadata, control_scaler=scaler
    )


def predict_gated_residual(
    fitted: FittedGatedResidual,
    samples: Sequence[HybridSample],
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prior = alps_prior_summaries(fitted.alps_prior, samples)
    features = residual_feature_matrix(samples, prior, scaler=fitted.control_scaler)
    progress = progress_values(samples, prior[:, 3])
    return predict_gated_residual_head(
        fitted.head,
        features,
        prior[:, 3],
        progress,
        correction_bound_fraction=float(
            fitted.metadata["correction_bound_fraction"]
        ),
        minimum_correction_bound_tokens=float(
            fitted.metadata["minimum_correction_bound_tokens"]
        ),
        terminal_threshold=float(fitted.metadata["terminal_threshold"]),
        batch_size=batch_size,
        device=str(fitted.metadata["device"]),
    )


def save_gated_residual(fitted: FittedGatedResidual, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_name = "alps_prior.json"
    checkpoint_name = f"{METHOD_ID}.pt"
    (output_dir / prior_name).write_text(
        json.dumps(fitted.alps_prior.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    atomic_torch_save(
        {
            "schema_version": 1,
            "state_dict": fitted.head.state_dict(),
            "metadata": fitted.metadata,
            "control_scaler": fitted.control_scaler.to_dict(),
        },
        output_dir / checkpoint_name,
    )
    return [prior_name, checkpoint_name]


def load_gated_residual(output_dir: Path) -> FittedGatedResidual:
    prior = WeightedLogRidge.from_dict(
        json.loads((output_dir / "alps_prior.json").read_text(encoding="utf-8"))
    )
    payload = load_torch_checkpoint(output_dir / f"{METHOD_ID}.pt")
    metadata = {**payload["metadata"], "device": "cpu"}
    head = build_gated_residual_head(
        int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        dropout=0.0,
        gate_initial_bias=float(metadata["gate_initial_bias"]),
    )
    head.load_state_dict(payload["state_dict"])
    return FittedGatedResidual(
        alps_prior=prior,
        head=head,
        metadata=metadata,
        control_scaler=ControlScaler.from_dict(payload["control_scaler"]),
    )
