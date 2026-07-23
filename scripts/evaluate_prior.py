"""Evaluate a frozen ALPS prior; final-test access requires explicit confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_length_prediction.data.io import read_trace_jsonl
from llm_length_prediction.evaluation.metrics import log1p_prior_metrics
from llm_length_prediction.experiment import (
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
    trace_path,
    validate_frozen_trace,
)
from llm_length_prediction.models.prior import StandardizedRidgeLogNormalPrior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/alps_v1_manifest.json"),
    )
    parser.add_argument(
        "--prior", type=Path, default=Path("artifacts/runs/alps_v1/stage1/prior.json")
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit("refusing final-test evaluation without --confirm-final-test")

    experiment = load_experiment(args.experiment)
    records = load_frozen_prompts(experiment)
    prior_payload = json.loads(args.prior.read_text(encoding="utf-8"))
    if prior_payload.get("fit_split") != "train":
        raise SystemExit("prior was not fitted exclusively on the training split")
    expected_prior_values = {
        "experiment_id": experiment["experiment_id"],
        "model_revision": experiment["model"]["revision"],
        "tokenizer_revision": experiment["model"]["tokenizer_revision"],
        "feature_layer": experiment["model"]["feature_layer"],
        "ridge_alpha": experiment["ridge"]["alpha"],
    }
    for name, expected in expected_prior_values.items():
        if prior_payload.get(name) != expected:
            raise SystemExit(
                f"prior {name}={prior_payload.get(name)!r}, expected frozen value {expected!r}"
            )
    prior = StandardizedRidgeLogNormalPrior.from_dict(prior_payload)
    layer = int(prior_payload["feature_layer"])
    trace_root = Path(experiment["outputs"]["trace_root"])
    jobs = list(rollout_jobs(records, split=args.split))
    missing = [
        trace_path(trace_root, record, seed)
        for record, seed in jobs
        if not trace_path(trace_root, record, seed).is_file()
    ]
    if missing:
        raise SystemExit(
            f"{args.split} collection is incomplete: missing {len(missing)} of {len(jobs)} traces; "
            f"first missing path: {missing[0]}"
        )

    rows = []
    actual = []
    predicted = []
    mus = []
    for record, seed in jobs:
        path = trace_path(trace_root, record, seed)
        trace = read_trace_jsonl(path)[0]
        validate_frozen_trace(trace, record=record, seed=seed, experiment=experiment)
        mu = prior.predict_mu(trace.prefill_hidden_states[layer])
        prediction = prior.predict_mean_length(trace.prefill_hidden_states[layer])
        actual.append(trace.output_tokens)
        predicted.append(prediction)
        mus.append(mu)
        rows.append(
            {
                "prompt_id": trace.prompt_id,
                "seed": trace.seed,
                "actual_output_tokens": trace.output_tokens,
                "predicted_log1p_mu": mu,
                "predicted_mean_output_tokens": prediction,
            }
        )
    output = args.output or args.prior.parent / f"{args.split}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split": args.split,
                **log1p_prior_metrics(actual, predicted, mus, prior.residual_variance),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path = output.with_suffix(".csv")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"evaluated {len(rows)} {args.split} traces: {output}")


if __name__ == "__main__":
    main()
