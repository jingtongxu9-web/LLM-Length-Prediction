"""Fit the frozen train-only ALPS log1p Ridge shifted-log-normal prior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from llm_length_prediction.data.io import read_trace_jsonl
from llm_length_prediction.evaluation.metrics import log1p_prior_metrics
from llm_length_prediction.experiment import (
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
    trace_path,
    validate_frozen_trace,
)
from llm_length_prediction.models.prior import fit_log1p_ridge_prior

DEFAULT_EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")


def _load_training_rows(
    trace_root: Path,
    *,
    layer: int,
    experiment: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[float]], list[int]]:
    rows: list[dict[str, Any]] = []
    features: list[list[float]] = []
    lengths: list[int] = []
    jobs = list(rollout_jobs(records, split="train"))
    missing = [
        trace_path(trace_root, record, seed)
        for record, seed in jobs
        if not trace_path(trace_root, record, seed).is_file()
    ]
    if missing:
        raise ValueError(
            f"training collection is incomplete: missing {len(missing)} of {len(jobs)} traces; "
            f"first missing path: {missing[0]}"
        )
    for record, seed in jobs:
        path = trace_path(trace_root, record, seed)
        trace = read_trace_jsonl(path)[0]
        validate_frozen_trace(trace, record=record, seed=seed, experiment=experiment)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    experiment = load_experiment(args.experiment)
    records = load_frozen_prompts(experiment)
    trace_root = args.trace_root or Path(experiment["outputs"]["trace_root"])
    output_dir = args.output_dir or Path(experiment["outputs"]["run_root"]) / "stage1"
    layer = int(experiment["model"]["feature_layer"])
    rows, features, actual = _load_training_rows(
        trace_root, layer=layer, experiment=experiment, records=records
    )
    ridge = experiment["ridge"]
    if ridge.get("standardize") is not True:
        raise ValueError("ALPS v1 requires train-only feature standardization")
    prior = fit_log1p_ridge_prior(features, actual, alpha=float(ridge["alpha"]))
    mus = [prior.predict_mu(feature) for feature in features]
    predicted = [prior.predict_mean_length(feature) for feature in features]

    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = prior.to_dict()
    model_payload.update(
        {
            "experiment_id": experiment["experiment_id"],
            "feature_layer": layer,
            "model_revision": experiment["model"]["revision"],
            "tokenizer_revision": experiment["model"]["tokenizer_revision"],
            "fit_split": "train",
            "training_dataset_sha256": _dataset_digest(rows),
            "training_count": len(rows),
        }
    )
    (output_dir / "prior.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = log1p_prior_metrics(actual, predicted, mus, prior.residual_variance)
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
