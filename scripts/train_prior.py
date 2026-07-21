"""Fit the frozen train-only ALPS log1p Ridge shifted-log-normal prior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.data.io import read_trace_jsonl
from llm_length_prediction.evaluation.metrics import mae, rmse
from llm_length_prediction.models.prior import fit_log1p_ridge_prior

DEFAULT_EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")


def _load_training_rows(
    trace_root: Path, *, layer: int, frozen_revision: str
) -> tuple[list[dict[str, Any]], list[list[float]], list[int]]:
    rows: list[dict[str, Any]] = []
    features: list[list[float]] = []
    lengths: list[int] = []
    paths = sorted((trace_root / "train").glob("*/seed_*.jsonl"))
    if not paths:
        raise ValueError(f"no training traces found under {trace_root / 'train'}")
    for path in paths:
        trace = read_trace_jsonl(path)[0]
        if trace.metadata.get("split") != "train":
            raise ValueError(f"non-training record found in training directory: {path}")
        if trace.model_revision != frozen_revision or trace.tokenizer_revision != frozen_revision:
            raise ValueError(f"trace does not use frozen model/tokenizer revision: {path}")
        try:
            hidden_state = trace.prefill_hidden_states[layer]
        except KeyError as error:
            raise ValueError(f"trace is missing frozen layer {layer}: {path}") from error
        rows.append(
            {
                "prompt_id": trace.prompt_id,
                "seed": trace.seed,
                "actual_output_tokens": trace.output_tokens,
                "trace_path": str(path),
                "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        features.append(hidden_state)
        lengths.append(trace.output_tokens)
    return rows, features, lengths


def _dataset_digest(rows: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        f"{row['prompt_id']}\t{row['seed']}\t{row['trace_sha256']}" for row in rows
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _metrics(
    actual: list[int], predicted: list[float], mus: list[float], variance: float
) -> dict[str, float]:
    target = np.log1p(np.asarray(actual, dtype=np.float64))
    mu = np.asarray(mus, dtype=np.float64)
    denominator = float(np.square(target - target.mean()).sum())
    r_squared = (
        0.0
        if denominator == 0.0
        else 1.0 - float(np.square(target - mu).sum()) / denominator
    )
    safe_variance = max(variance, 1e-12)
    nll = float(
        np.mean(
            0.5 * math.log(2.0 * math.pi * safe_variance)
            + np.square(target - mu) / (2.0 * safe_variance)
            + target
        )
    )
    lower = np.expm1(mu - 1.959963984540054 * math.sqrt(safe_variance))
    upper = np.expm1(mu + 1.959963984540054 * math.sqrt(safe_variance))
    coverage = float(np.mean((np.asarray(actual) >= lower) & (np.asarray(actual) <= upper)))
    return {
        "count": float(len(actual)),
        "mae_tokens": mae(actual, predicted),
        "rmse_tokens": rmse(actual, predicted),
        "r_squared_log1p": r_squared,
        "negative_log_likelihood": nll,
        "interval_95_coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    trace_root = args.trace_root or Path(experiment["outputs"]["trace_root"])
    output_dir = args.output_dir or Path(experiment["outputs"]["run_root"]) / "stage1"
    layer = int(experiment["model"]["feature_layer"])
    revision = experiment["model"]["revision"]
    rows, features, actual = _load_training_rows(
        trace_root, layer=layer, frozen_revision=revision
    )
    prior = fit_log1p_ridge_prior(features, actual, alpha=args.alpha)
    mus = [prior.predict_mu(feature) for feature in features]
    predicted = [prior.predict_mean_length(feature) for feature in features]

    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = prior.to_dict()
    model_payload.update(
        {
            "experiment_id": experiment["experiment_id"],
            "feature_layer": layer,
            "model_revision": revision,
            "tokenizer_revision": experiment["model"]["tokenizer_revision"],
            "fit_split": "train",
            "training_dataset_sha256": _dataset_digest(rows),
            "training_count": len(rows),
        }
    )
    (output_dir / "prior.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = _metrics(actual, predicted, mus, prior.residual_variance)
    metrics["residual_variance_mle"] = prior.residual_variance
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "seed",
                "actual_output_tokens",
                "predicted_log1p_mu",
                "predicted_mean_output_tokens",
                "trace_path",
                "trace_sha256",
            ],
        )
        writer.writeheader()
        for row, mu, prediction in zip(rows, mus, predicted, strict=True):
            writer.writerow(
                {**row, "predicted_log1p_mu": mu, "predicted_mean_output_tokens": prediction}
            )
    print(f"fitted {len(rows)} train traces; outputs: {output_dir}")


if __name__ == "__main__":
    main()
