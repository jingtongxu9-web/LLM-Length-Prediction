"""Standalone frozen PLP v2 control and terminal-zero PLP v3 candidate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.checkpoints import atomic_torch_save, load_torch_checkpoint
from llm_length_prediction.models.hybrid import (
    HybridSample,
    build_progressive_head,
    fit_progressive_head,
    predict_progressive_head,
)

PLP_V3_METHOD_IDS = ("plp_v2_frozen", "plp_terminal_zero_v3")


def plp_feature_matrix(samples: Sequence[HybridSample]) -> np.ndarray:
    if not samples:
        raise ValueError("PLP training requires at least one sample")
    return np.stack([sample.plp_features for sample in samples]).astype(np.float32)


def method_settings(
    protocol: dict[str, Any], method_id: str
) -> tuple[int, bool, bool]:
    if method_id not in PLP_V3_METHOD_IDS:
        raise ValueError(f"unsupported PLP-only v3 method: {method_id}")
    method = protocol["methods"][method_id]
    hidden_dim = int(method["hidden_dim"])
    terminal_zero = method["terminal_zero_bin"]
    if hidden_dim <= 0 or not isinstance(terminal_zero, bool):
        raise ValueError(f"invalid frozen settings for {method_id}")
    weighting = str(method["target_range_weighting"])
    scope = "positive_targets" if terminal_zero else "all_targets"
    allowed = {f"unweighted_{scope}", f"rollout_balanced_{scope}"}
    if weighting not in allowed:
        raise ValueError(f"{method_id} has inconsistent target-range weighting")
    return hidden_dim, terminal_zero, weighting.startswith("rollout_balanced_")


@dataclass
class FittedPLPV3:
    heads: dict[str, Any]
    metadata: dict[str, dict[str, Any]]


def fit_plp_v3(
    samples: Sequence[HybridSample],
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
    device: str,
) -> FittedPLPV3:
    features = plp_feature_matrix(samples)
    head_config = config["progressive_head"]
    training = config["training"]
    common = {
        "num_bins": int(head_config["num_bins"]),
        "percentiles": tuple(float(x) for x in head_config["target_range_percentiles"]),
        "lambda_ce": float(head_config["lambda_ce"]),
        "dropout": float(head_config["dropout"]),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "seed": int(training["seed"]),
        "device": device,
    }
    heads: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for method_id in PLP_V3_METHOD_IDS:
        hidden_dim, terminal_zero, weighted_range = method_settings(protocol, method_id)
        head, report = fit_progressive_head(
            samples,
            features,
            hidden_dim=hidden_dim,
            terminal_zero=terminal_zero,
            weighted_range=weighted_range,
            **common,
        )
        heads[method_id] = head
        metadata[method_id] = {**report, "method_id": method_id}
    return FittedPLPV3(heads=heads, metadata=metadata)


def predict_plp_v3(
    fitted: FittedPLPV3,
    samples: Sequence[HybridSample],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    features = plp_feature_matrix(samples)
    return {
        method_id: predict_progressive_head(
            fitted.heads[method_id],
            features,
            batch_size=batch_size,
            device=str(fitted.metadata[method_id]["device"]),
        )
        for method_id in PLP_V3_METHOD_IDS
    }


def save_plp_v3(fitted: FittedPLPV3, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for method_id in PLP_V3_METHOD_IDS:
        name = f"{method_id}.pt"
        atomic_torch_save(
            {
                "schema_version": 1,
                "state_dict": fitted.heads[method_id].state_dict(),
                "metadata": fitted.metadata[method_id],
            },
            output_dir / name,
        )
        files[method_id] = name
    return files


def _load_head(path: Path) -> tuple[Any, dict[str, Any]]:
    payload = load_torch_checkpoint(path)
    metadata = payload["metadata"]
    head = build_progressive_head(
        int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        num_bins=int(metadata["num_bins"]),
        value_range=tuple(float(x) for x in metadata["target_range"]),
        terminal_zero=bool(metadata["terminal_zero_bin"]),
        dropout=0.0,
    )
    head.load_state_dict(payload["state_dict"])
    head.to("cpu")
    return head, {**metadata, "device": "cpu"}


def load_plp_v3(model_dir: Path) -> FittedPLPV3:
    heads: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for method_id in PLP_V3_METHOD_IDS:
        heads[method_id], metadata[method_id] = _load_head(model_dir / f"{method_id}.pt")
    return FittedPLPV3(heads=heads, metadata=metadata)
