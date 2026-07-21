"""Evaluate a frozen ALPS prior; final-test access requires explicit confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_length_prediction.data.io import read_trace_jsonl
from llm_length_prediction.evaluation.metrics import mae, rmse
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

    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    prior_payload = json.loads(args.prior.read_text(encoding="utf-8"))
    if prior_payload.get("fit_split") != "train":
        raise SystemExit("prior was not fitted exclusively on the training split")
    prior = StandardizedRidgeLogNormalPrior.from_dict(prior_payload)
    layer = int(prior_payload["feature_layer"])
    trace_root = Path(experiment["outputs"]["trace_root"])
    paths = sorted((trace_root / args.split).glob("*/seed_*.jsonl"))
    if not paths:
        raise SystemExit(f"no {args.split} traces found")

    rows = []
    actual = []
    predicted = []
    for path in paths:
        trace = read_trace_jsonl(path)[0]
        if trace.metadata.get("split") != args.split:
            raise SystemExit(f"split metadata mismatch: {path}")
        prediction = prior.predict_mean_length(trace.prefill_hidden_states[layer])
        actual.append(trace.output_tokens)
        predicted.append(prediction)
        rows.append(
            {
                "prompt_id": trace.prompt_id,
                "seed": trace.seed,
                "actual_output_tokens": trace.output_tokens,
                "predicted_mean_output_tokens": prediction,
            }
        )
    output = args.output or args.prior.parent / f"{args.split}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split": args.split,
                "count": len(rows),
                "mae_tokens": mae(actual, predicted),
                "rmse_tokens": rmse(actual, predicted),
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
