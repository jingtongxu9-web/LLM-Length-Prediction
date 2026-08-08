"""Explicit ALPS+PLP Hybrid v1 (concatenation) and v2 (residual correction)."""

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
    SummaryScaler,
    WeightedLogRidge,
    alps_prior_summaries,
    build_progressive_head,
    fit_alps_prior,
    fit_progressive_head,
    fit_summary_scaler,
    hybrid_feature_matrix,
    predict_progressive_head,
)

METHOD_IDS = (
    "alps_countdown",
    "plp_terminal_zero_v3",
    "alps_plp_concat_v1",
    "alps_plp_residual_v2",
)


def _weights(samples: Sequence[HybridSample]) -> np.ndarray:
    return np.asarray([sample.sequence_weight for sample in samples], dtype=np.float32)


def _plp_matrix(samples: Sequence[HybridSample]) -> np.ndarray:
    return np.stack([sample.plp_features for sample in samples]).astype(np.float32)


@dataclass(frozen=True)
class ControlScaler:
    """Rollout-balanced scaler for v2's small dynamic control vector."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlScaler:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
        )


def residual_control_matrix(
    samples: Sequence[HybridSample], prior_summaries: np.ndarray
) -> np.ndarray:
    """Return [ALPS countdown, step, entropy, mean, slope, EOS probability]."""

    if prior_summaries.shape != (len(samples), 5):
        raise ValueError("prior summaries do not align with Hybrid samples")
    dynamic = np.asarray([sample.dynamic_features for sample in samples], dtype=np.float32)
    return np.column_stack((prior_summaries[:, 3], dynamic)).astype(np.float32)


def fit_control_scaler(values: np.ndarray, weights: np.ndarray) -> ControlScaler:
    if values.ndim != 2 or weights.shape != (len(values),) or np.any(weights <= 0):
        raise ValueError("control values and weights are invalid")
    normalized = weights.astype(np.float64) / weights.sum()
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum((values - mean) ** 2 * normalized[:, None], axis=0)
    return ControlScaler(
        mean=mean.astype(np.float32),
        scale=np.maximum(np.sqrt(variance), 1e-6).astype(np.float32),
    )


def residual_feature_matrix(
    samples: Sequence[HybridSample],
    prior_summaries: np.ndarray,
    *,
    scaler: ControlScaler,
) -> np.ndarray:
    controls = residual_control_matrix(samples, prior_summaries)
    return np.concatenate((_plp_matrix(samples), scaler.transform(controls)), axis=1)


def build_residual_head(input_dim: int, *, hidden_dim: int, dropout: float) -> Any:
    """Build a correction/terminal head whose initial correction is exactly zero."""

    try:
        from torch import nn
    except ImportError as error:
        raise RuntimeError("Hybrid residual training requires PyTorch") from error

    class ResidualHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.correction = nn.Linear(hidden_dim, 1)
            self.terminal = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)
            nn.init.zeros_(self.terminal.weight)
            nn.init.constant_(self.terminal.bias, -4.0)

        def forward(self, features: Any) -> tuple[Any, Any]:
            encoded = self.backbone(features)
            return self.correction(encoded).squeeze(-1), self.terminal(encoded).squeeze(-1)

    return ResidualHead()


def fit_residual_head(
    samples: Sequence[HybridSample],
    features: np.ndarray,
    base_countdown: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
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
        raise ValueError("residual feature matrix does not align")
    if base_countdown.shape != (len(samples),):
        raise ValueError("ALPS countdown does not align")
    if not 0.0 <= terminal_loss_weight <= 1.0 or not 0.0 < terminal_threshold < 1.0:
        raise ValueError("invalid residual terminal settings")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("Hybrid residual training requires PyTorch") from error

    targets = np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float32)
    weights = _weights(samples)
    weights *= len(weights) / weights.sum()
    terminal = (targets == 0).astype(np.float32)
    nonterminal = terminal == 0
    if not np.any(nonterminal) or not np.any(~nonterminal):
        raise ValueError("residual training requires terminal and non-terminal points")
    residual = targets - base_countdown.astype(np.float32)
    residual_scale = float(
        np.sqrt(
            np.sum(weights[nonterminal] * residual[nonterminal] ** 2)
            / weights[nonterminal].sum()
        )
    )
    residual_scale = max(residual_scale, 1.0)
    standardized_residual = residual / residual_scale
    positive_weight = float(weights[nonterminal].sum() / weights[~nonterminal].sum())

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved == "auto":
        resolved = "cpu"
    head = build_residual_head(
        features.shape[1], hidden_dim=hidden_dim, dropout=dropout
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=False
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features.astype(np.float32, copy=False)),
            torch.from_numpy(standardized_residual),
            torch.from_numpy(terminal),
            torch.from_numpy(weights),
        ),
        batch_size=min(batch_size, len(samples)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    history: list[float] = []
    pos_weight = torch.tensor(positive_weight, dtype=torch.float32, device=resolved)
    for _ in range(epochs):
        head.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for batch_features, batch_residual, batch_terminal, batch_weights in loader:
            batch_features = batch_features.to(resolved)
            batch_residual = batch_residual.to(resolved)
            batch_terminal = batch_terminal.to(resolved)
            batch_weights = batch_weights.to(resolved)
            optimizer.zero_grad(set_to_none=True)
            predicted_residual, terminal_logits = head(batch_features)
            eligible = batch_terminal == 0
            if bool(eligible.any().item()):
                residual_loss = functional.smooth_l1_loss(
                    predicted_residual[eligible], batch_residual[eligible], reduction="none"
                )
                residual_loss = (
                    (batch_weights[eligible] * residual_loss).sum()
                    / batch_weights[eligible].sum()
                )
            else:
                residual_loss = predicted_residual.sum() * 0.0
            terminal_loss = functional.binary_cross_entropy_with_logits(
                terminal_logits,
                batch_terminal,
                reduction="none",
                pos_weight=pos_weight,
            )
            terminal_loss = (batch_weights * terminal_loss).sum() / batch_weights.sum()
            loss = (
                (1.0 - terminal_loss_weight) * residual_loss
                + terminal_loss_weight * terminal_loss
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Hybrid residual training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * float(batch_weights.sum().item())
            weight_sum += float(batch_weights.sum().item())
        history.append(loss_sum / weight_sum)
    return head, {
        "device": resolved,
        "input_dim": features.shape[1],
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "residual_scale_tokens": residual_scale,
        "terminal_loss_weight": terminal_loss_weight,
        "terminal_positive_weight": positive_weight,
        "terminal_threshold": terminal_threshold,
        "trainable_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "epoch_losses": history,
        "final_loss": history[-1],
    }


def predict_residual_head(
    head: Any,
    features: np.ndarray,
    base_countdown: np.ndarray,
    *,
    residual_scale: float,
    terminal_threshold: float,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Hybrid residual inference requires PyTorch") from error
    head.eval()
    corrections: list[np.ndarray] = []
    terminal_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            correction, terminal_logit = head(
                torch.from_numpy(features[start : start + batch_size]).to(device)
            )
            corrections.append((correction * residual_scale).cpu().numpy())
            terminal_probabilities.append(torch.sigmoid(terminal_logit).cpu().numpy())
    correction_values = np.concatenate(corrections)
    terminal_values = np.concatenate(terminal_probabilities)
    prediction = np.maximum(base_countdown + correction_values, 0.0)
    prediction[terminal_values >= terminal_threshold] = 0.0
    return prediction, correction_values, terminal_values


@dataclass
class FittedHybridVersions:
    alps_prior: WeightedLogRidge
    plp_head: Any
    plp_metadata: dict[str, Any]
    concat_head: Any
    concat_metadata: dict[str, Any]
    concat_scaler: SummaryScaler
    residual_head: Any
    residual_metadata: dict[str, Any]
    residual_scaler: ControlScaler
    reports: dict[str, Any]


def _progressive_common(config: dict[str, Any], device: str) -> dict[str, Any]:
    head = config["progressive_head"]
    training = config["training"]
    return {
        "num_bins": int(head["num_bins"]),
        "percentiles": tuple(float(value) for value in head["target_range_percentiles"]),
        "lambda_ce": float(head["lambda_ce"]),
        "dropout": float(head["dropout"]),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "seed": int(training["seed"]),
        "device": device,
    }


def fit_hybrid_versions(
    samples: Sequence[HybridSample],
    cross_fitted_prior: np.ndarray,
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
    device: str,
) -> FittedHybridVersions:
    if not samples or cross_fitted_prior.shape != (len(samples), 5):
        raise ValueError("Hybrid version training inputs are empty or misaligned")
    weights = _weights(samples)
    alps_prior = fit_alps_prior(samples, alpha=float(config["stacking"]["alps_ridge_alpha"]))
    common = _progressive_common(config, device)
    plp_settings = protocol["methods"]["plp_terminal_zero_v3"]
    plp_head, plp_metadata = fit_progressive_head(
        samples,
        _plp_matrix(samples),
        hidden_dim=int(plp_settings["hidden_dim"]),
        terminal_zero=True,
        weighted_range=False,
        **common,
    )

    concat_scaler = fit_summary_scaler(cross_fitted_prior, weights)
    concat_features = hybrid_feature_matrix(
        samples, cross_fitted_prior, scaler=concat_scaler
    )
    concat_settings = protocol["methods"]["alps_plp_concat_v1"]
    concat_head, concat_metadata = fit_progressive_head(
        samples,
        concat_features,
        hidden_dim=int(concat_settings["hidden_dim"]),
        terminal_zero=True,
        weighted_range=True,
        **common,
    )

    residual_controls = residual_control_matrix(samples, cross_fitted_prior)
    residual_scaler = fit_control_scaler(residual_controls, weights)
    residual_features = residual_feature_matrix(
        samples, cross_fitted_prior, scaler=residual_scaler
    )
    residual_settings = protocol["methods"]["alps_plp_residual_v2"]
    residual_head, residual_metadata = fit_residual_head(
        samples,
        residual_features,
        cross_fitted_prior[:, 3],
        hidden_dim=int(residual_settings["hidden_dim"]),
        dropout=float(common["dropout"]),
        terminal_loss_weight=float(residual_settings["terminal_loss_weight"]),
        terminal_threshold=float(residual_settings["terminal_threshold"]),
        epochs=int(common["epochs"]),
        batch_size=int(common["batch_size"]),
        learning_rate=float(common["learning_rate"]),
        weight_decay=float(common["weight_decay"]),
        seed=int(common["seed"]),
        device=device,
    )
    reports = {
        "plp_terminal_zero_v3": plp_metadata,
        "alps_plp_concat_v1": concat_metadata,
        "alps_plp_residual_v2": residual_metadata,
    }
    return FittedHybridVersions(
        alps_prior=alps_prior,
        plp_head=plp_head,
        plp_metadata=plp_metadata,
        concat_head=concat_head,
        concat_metadata=concat_metadata,
        concat_scaler=concat_scaler,
        residual_head=residual_head,
        residual_metadata=residual_metadata,
        residual_scaler=residual_scaler,
        reports=reports,
    )


def predict_hybrid_versions(
    fitted: FittedHybridVersions,
    samples: Sequence[HybridSample],
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prior = alps_prior_summaries(fitted.alps_prior, samples)
    plp = _plp_matrix(samples)
    concat = hybrid_feature_matrix(samples, prior, scaler=fitted.concat_scaler)
    residual = residual_feature_matrix(samples, prior, scaler=fitted.residual_scaler)
    residual_prediction, correction, terminal_probability = predict_residual_head(
        fitted.residual_head,
        residual,
        prior[:, 3],
        residual_scale=float(fitted.residual_metadata["residual_scale_tokens"]),
        terminal_threshold=float(fitted.residual_metadata["terminal_threshold"]),
        batch_size=batch_size,
        device=str(fitted.residual_metadata["device"]),
    )
    predictions = {
        "alps_countdown": prior[:, 3].astype(np.float64),
        "plp_terminal_zero_v3": predict_progressive_head(
            fitted.plp_head,
            plp,
            batch_size=batch_size,
            device=str(fitted.plp_metadata["device"]),
        ),
        "alps_plp_concat_v1": predict_progressive_head(
            fitted.concat_head,
            concat,
            batch_size=batch_size,
            device=str(fitted.concat_metadata["device"]),
        ),
        "alps_plp_residual_v2": residual_prediction,
    }
    if tuple(predictions) != METHOD_IDS:
        raise RuntimeError("Hybrid version method order changed")
    diagnostics = {
        "alps_countdown": prior[:, 3].astype(np.float64),
        "residual_correction": correction.astype(np.float64),
        "terminal_probability": terminal_probability.astype(np.float64),
    }
    return predictions, diagnostics


def save_hybrid_versions(fitted: FittedHybridVersions, output_dir: Path) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alps_prior.json").write_text(
        json.dumps(fitted.alps_prior.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoints = {
        "plp_terminal_zero_v3.pt": {
            "state_dict": fitted.plp_head.state_dict(),
            "metadata": fitted.plp_metadata,
        },
        "alps_plp_concat_v1.pt": {
            "state_dict": fitted.concat_head.state_dict(),
            "metadata": fitted.concat_metadata,
            "concat_scaler": fitted.concat_scaler.to_dict(),
        },
        "alps_plp_residual_v2.pt": {
            "state_dict": fitted.residual_head.state_dict(),
            "metadata": fitted.residual_metadata,
            "residual_scaler": fitted.residual_scaler.to_dict(),
        },
    }
    for name, payload in checkpoints.items():
        atomic_torch_save({"schema_version": 1, **payload}, output_dir / name)
    return {
        "alps_countdown": ["alps_prior.json"],
        "plp_terminal_zero_v3": ["plp_terminal_zero_v3.pt"],
        "alps_plp_concat_v1": ["alps_prior.json", "alps_plp_concat_v1.pt"],
        "alps_plp_residual_v2": ["alps_prior.json", "alps_plp_residual_v2.pt"],
    }


def load_hybrid_versions(output_dir: Path) -> FittedHybridVersions:
    alps = WeightedLogRidge.from_dict(
        json.loads((output_dir / "alps_prior.json").read_text(encoding="utf-8"))
    )
    plp_payload = load_torch_checkpoint(output_dir / "plp_terminal_zero_v3.pt")
    concat_payload = load_torch_checkpoint(output_dir / "alps_plp_concat_v1.pt")
    residual_payload = load_torch_checkpoint(output_dir / "alps_plp_residual_v2.pt")
    plp_metadata = {**plp_payload["metadata"], "device": "cpu"}
    concat_metadata = {**concat_payload["metadata"], "device": "cpu"}
    residual_metadata = {**residual_payload["metadata"], "device": "cpu"}
    plp_head = build_progressive_head(
        int(plp_metadata["input_dim"]),
        hidden_dim=int(plp_metadata["hidden_dim"]),
        num_bins=int(plp_metadata["num_bins"]),
        value_range=tuple(float(value) for value in plp_metadata["target_range"]),
        terminal_zero=bool(plp_metadata["terminal_zero_bin"]),
        dropout=0.0,
    )
    concat_head = build_progressive_head(
        int(concat_metadata["input_dim"]),
        hidden_dim=int(concat_metadata["hidden_dim"]),
        num_bins=int(concat_metadata["num_bins"]),
        value_range=tuple(float(value) for value in concat_metadata["target_range"]),
        terminal_zero=bool(concat_metadata["terminal_zero_bin"]),
        dropout=0.0,
    )
    residual_head = build_residual_head(
        int(residual_metadata["input_dim"]),
        hidden_dim=int(residual_metadata["hidden_dim"]),
        dropout=0.0,
    )
    plp_head.load_state_dict(plp_payload["state_dict"])
    concat_head.load_state_dict(concat_payload["state_dict"])
    residual_head.load_state_dict(residual_payload["state_dict"])
    return FittedHybridVersions(
        alps_prior=alps,
        plp_head=plp_head,
        plp_metadata=plp_metadata,
        concat_head=concat_head,
        concat_metadata=concat_metadata,
        concat_scaler=SummaryScaler.from_dict(concat_payload["concat_scaler"]),
        residual_head=residual_head,
        residual_metadata=residual_metadata,
        residual_scaler=ControlScaler.from_dict(residual_payload["residual_scaler"]),
        reports={},
    )
