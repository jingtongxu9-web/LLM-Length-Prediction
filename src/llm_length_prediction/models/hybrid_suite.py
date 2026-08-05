"""Frozen eight-method comparison suite used by Hybrid v3."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.checkpoints import atomic_torch_save, load_torch_checkpoint
from llm_length_prediction.models.dynamic import (
    StandardizedMLPRemainingLength,
    fit_plp_mlp,
)
from llm_length_prediction.models.hybrid import (
    HybridSample,
    SummaryScaler,
    WeightedLogRidge,
    alps_prior_summaries,
    build_progressive_head,
    fit_alps_prior,
    fit_log1p_ridge,
    fit_progressive_head,
    fit_summary_scaler,
    hybrid_feature_matrix,
    predict_progressive_head,
)

METHOD_IDS = (
    "step_only_ridge",
    "alps_countdown",
    "dynamic_ridge",
    "dynamic_signal_mlp_v1",
    "plp_v2_frozen",
    "plp_small_terminal_v3",
    "alps_dynamic_ridge",
    "alps_plp_hybrid_v3",
)


def dynamic_matrix(samples: Sequence[HybridSample]) -> np.ndarray:
    return np.asarray([sample.dynamic_features for sample in samples], dtype=np.float64)


def plp_matrix(samples: Sequence[HybridSample]) -> np.ndarray:
    return np.stack([sample.plp_features for sample in samples]).astype(np.float32)


def targets(samples: Sequence[HybridSample]) -> np.ndarray:
    return np.asarray([sample.remaining_tokens for sample in samples], dtype=np.float64)


def weights(samples: Sequence[HybridSample]) -> np.ndarray:
    return np.asarray([sample.sequence_weight for sample in samples], dtype=np.float64)


@dataclass
class FittedSuite:
    alps_prior: WeightedLogRidge
    step_ridge: WeightedLogRidge
    dynamic_ridge: WeightedLogRidge
    dynamic_mlp: Any
    plp_v2_head: Any
    plp_v2_metadata: dict[str, Any]
    plp_small_head: Any
    plp_small_metadata: dict[str, Any]
    alps_dynamic_ridge: WeightedLogRidge
    hybrid_head: Any
    hybrid_metadata: dict[str, Any]
    prior_summary_scaler: SummaryScaler
    reports: dict[str, Any]


def fit_suite(
    samples: Sequence[HybridSample],
    cross_fitted_prior: np.ndarray,
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
    device: str,
) -> FittedSuite:
    """Fit every frozen comparator with exactly the same training rollouts."""

    if not samples or cross_fitted_prior.shape != (len(samples), 5):
        raise ValueError("suite training inputs are empty or misaligned")
    y = targets(samples)
    sample_weights = weights(samples)
    dynamic = dynamic_matrix(samples)
    alpha = float(config["stacking"]["alps_ridge_alpha"])
    alps_prior = fit_alps_prior(samples, alpha=alpha)
    step_ridge = fit_log1p_ridge(
        dynamic[:, :1], y, sample_weights, alpha=alpha, target_name="log1p_remaining_tokens"
    )
    dynamic_ridge = fit_log1p_ridge(
        dynamic, y, sample_weights, alpha=alpha, target_name="log1p_remaining_tokens"
    )
    nonterminal = [index for index, sample in enumerate(samples) if sample.remaining_tokens > 0]
    dynamic_mlp, dynamic_report = fit_plp_mlp(
        dynamic[nonterminal],
        y[nonterminal].astype(int),
        sample_weights[nonterminal],
        hidden_sizes=(128, 64),
        dropout=0.1,
        epochs=50,
        batch_size=512,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=int(config["training"]["seed"]),
        device=device,
    )
    head = config["progressive_head"]
    training = config["training"]
    common = {
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
    plp = plp_matrix(samples)
    plp_v2_head, plp_v2_metadata = fit_progressive_head(
        samples,
        plp,
        hidden_dim=plp.shape[1] // 2,
        terminal_zero=False,
        weighted_range=False,
        **common,
    )
    small_dim = int(protocol["methods"]["plp_small_terminal_v3"]["hidden_dim"])
    plp_small_head, plp_small_metadata = fit_progressive_head(
        samples,
        plp,
        hidden_dim=small_dim,
        terminal_zero=True,
        weighted_range=True,
        **common,
    )
    summary_scaler = fit_summary_scaler(cross_fitted_prior, sample_weights)
    scaled_prior = summary_scaler.transform(cross_fitted_prior)
    alps_dynamic_ridge = fit_log1p_ridge(
        np.concatenate((dynamic, scaled_prior), axis=1),
        y,
        sample_weights,
        alpha=alpha,
        target_name="log1p_remaining_tokens",
    )
    hybrid_features = hybrid_feature_matrix(samples, cross_fitted_prior, scaler=summary_scaler)
    hybrid_dim = int(protocol["methods"]["alps_plp_hybrid_v3"]["hidden_dim"])
    hybrid_head, hybrid_metadata = fit_progressive_head(
        samples,
        hybrid_features,
        hidden_dim=hybrid_dim,
        terminal_zero=True,
        weighted_range=True,
        **common,
    )
    return FittedSuite(
        alps_prior=alps_prior,
        step_ridge=step_ridge,
        dynamic_ridge=dynamic_ridge,
        dynamic_mlp=dynamic_mlp,
        plp_v2_head=plp_v2_head,
        plp_v2_metadata=plp_v2_metadata,
        plp_small_head=plp_small_head,
        plp_small_metadata=plp_small_metadata,
        alps_dynamic_ridge=alps_dynamic_ridge,
        hybrid_head=hybrid_head,
        hybrid_metadata=hybrid_metadata,
        prior_summary_scaler=summary_scaler,
        reports={"dynamic_signal_mlp_v1": dynamic_report},
    )


def predict_suite(
    fitted: FittedSuite,
    samples: Sequence[HybridSample],
    *,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dynamic = dynamic_matrix(samples)
    prior = alps_prior_summaries(fitted.alps_prior, samples)
    steps = dynamic[:, 0]
    plp = plp_matrix(samples)
    scaled_prior = fitted.prior_summary_scaler.transform(prior)
    hybrid = hybrid_feature_matrix(samples, prior, scaler=fitted.prior_summary_scaler)
    v2_device = str(fitted.plp_v2_metadata["device"])
    small_device = str(fitted.plp_small_metadata["device"])
    hybrid_device = str(fitted.hybrid_metadata["device"])
    predictions = {
        "step_only_ridge": fitted.step_ridge.predict_mean(dynamic[:, :1]),
        "alps_countdown": np.maximum(prior[:, 2] - steps, 0.0),
        "dynamic_ridge": fitted.dynamic_ridge.predict_mean(dynamic),
        "dynamic_signal_mlp_v1": fitted.dynamic_mlp.predict_remaining_many(dynamic),
        "plp_v2_frozen": predict_progressive_head(
            fitted.plp_v2_head, plp, batch_size=batch_size, device=v2_device
        ),
        "plp_small_terminal_v3": predict_progressive_head(
            fitted.plp_small_head, plp, batch_size=batch_size, device=small_device
        ),
        "alps_dynamic_ridge": fitted.alps_dynamic_ridge.predict_mean(
            np.concatenate((dynamic, scaled_prior), axis=1)
        ),
        "alps_plp_hybrid_v3": predict_progressive_head(
            fitted.hybrid_head, hybrid, batch_size=batch_size, device=hybrid_device
        ),
    }
    if tuple(predictions) != METHOD_IDS:
        raise RuntimeError("method suite does not match the frozen protocol")
    return predictions


def save_suite(fitted: FittedSuite, output_dir: Path) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_models = {
        "alps_prior.json": fitted.alps_prior.to_dict(),
        "step_only_ridge.json": fitted.step_ridge.to_dict(),
        "dynamic_ridge.json": fitted.dynamic_ridge.to_dict(),
        "dynamic_signal_mlp_v1.json": fitted.dynamic_mlp.to_dict(),
        "alps_dynamic_ridge.json": fitted.alps_dynamic_ridge.to_dict(),
    }
    for name, payload in json_models.items():
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    checkpoints = {
        "plp_v2_frozen.pt": (fitted.plp_v2_head, fitted.plp_v2_metadata, None),
        "plp_small_terminal_v3.pt": (
            fitted.plp_small_head,
            fitted.plp_small_metadata,
            None,
        ),
        "alps_plp_hybrid_v3.pt": (
            fitted.hybrid_head,
            fitted.hybrid_metadata,
            fitted.prior_summary_scaler.to_dict(),
        ),
    }
    for name, (model, metadata, scaler) in checkpoints.items():
        atomic_torch_save(
            {
                "schema_version": 1,
                "state_dict": model.state_dict(),
                "metadata": metadata,
                "prior_summary_scaler": scaler,
            },
            output_dir / name,
        )
    return {
        "step_only_ridge": ["step_only_ridge.json"],
        "alps_countdown": ["alps_prior.json"],
        "dynamic_ridge": ["dynamic_ridge.json"],
        "dynamic_signal_mlp_v1": ["dynamic_signal_mlp_v1.json"],
        "plp_v2_frozen": ["plp_v2_frozen.pt"],
        "plp_small_terminal_v3": ["plp_small_terminal_v3.pt"],
        "alps_dynamic_ridge": ["alps_prior.json", "alps_dynamic_ridge.json"],
        "alps_plp_hybrid_v3": ["alps_prior.json", "alps_plp_hybrid_v3.pt"],
    }


def _load_head(path: Path) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    payload = load_torch_checkpoint(path)
    metadata = payload["metadata"]
    model = build_progressive_head(
        int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        num_bins=int(metadata["num_bins"]),
        value_range=tuple(float(value) for value in metadata["target_range"]),
        terminal_zero=bool(metadata["terminal_zero_bin"]),
        dropout=0.0,
    )
    model.load_state_dict(payload["state_dict"])
    model.to("cpu")
    metadata = {**metadata, "device": "cpu"}
    return model, metadata, payload.get("prior_summary_scaler")


def load_suite(output_dir: Path) -> FittedSuite:
    def read(name: str) -> dict[str, Any]:
        return json.loads((output_dir / name).read_text(encoding="utf-8"))

    v2_head, v2_metadata, _ = _load_head(output_dir / "plp_v2_frozen.pt")
    small_head, small_metadata, _ = _load_head(output_dir / "plp_small_terminal_v3.pt")
    hybrid_head, hybrid_metadata, scaler_payload = _load_head(output_dir / "alps_plp_hybrid_v3.pt")
    if scaler_payload is None:
        raise ValueError("Hybrid checkpoint is missing its prior summary scaler")
    return FittedSuite(
        alps_prior=WeightedLogRidge.from_dict(read("alps_prior.json")),
        step_ridge=WeightedLogRidge.from_dict(read("step_only_ridge.json")),
        dynamic_ridge=WeightedLogRidge.from_dict(read("dynamic_ridge.json")),
        dynamic_mlp=StandardizedMLPRemainingLength.from_dict(read("dynamic_signal_mlp_v1.json")),
        plp_v2_head=v2_head,
        plp_v2_metadata=v2_metadata,
        plp_small_head=small_head,
        plp_small_metadata=small_metadata,
        alps_dynamic_ridge=WeightedLogRidge.from_dict(read("alps_dynamic_ridge.json")),
        hybrid_head=hybrid_head,
        hybrid_metadata=hybrid_metadata,
        prior_summary_scaler=SummaryScaler.from_dict(scaler_payload),
        reports={},
    )
