"""Analyze frozen ALPS predictions by length, task, seed, and matched prompt family."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from llm_length_prediction.evaluation.breakdown import (
    build_group_breakdowns,
    build_matched_length_analysis,
    build_prompt_mean_point_analysis,
    build_seed_stability,
    flatten_group_breakdowns,
    flatten_matched_length_analysis,
    flatten_prompt_mean_point_analysis,
    render_markdown_report,
)
from llm_length_prediction.experiment import (
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
)

DEFAULT_EXPERIMENT = Path("configs/experiments/alps_v1_manifest.json")
DEFAULT_PRIOR = Path("artifacts/runs/alps_v1/stage1/prior.json")


def _load_prediction_rows(
    predictions_path: Path,
    *,
    split: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not predictions_path.is_file():
        raise SystemExit(
            f"missing {predictions_path}; run scripts/evaluate_prior.py --split {split} first"
        )

    records_by_prompt = {
        record["prompt_id"]: record for record in records if record["split"] == split
    }
    expected = {(record["prompt_id"], seed) for record, seed in rollout_jobs(records, split=split)}
    rows = []
    seen: set[tuple[str, int]] = set()
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            prompt_id = raw_row["prompt_id"]
            seed = int(raw_row["seed"])
            key = (prompt_id, seed)
            if key in seen:
                raise SystemExit(f"duplicate prediction row: prompt_id={prompt_id}, seed={seed}")
            if key not in expected:
                raise SystemExit(
                    f"unexpected {split} prediction row: prompt_id={prompt_id}, seed={seed}"
                )
            record = records_by_prompt[prompt_id]
            intended_range = record["intended_output_tokens"]
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_family_id": record["prompt_family_id"],
                    "task_type": record["task_type"],
                    "intended_length": record["intended_length"],
                    "intended_output_min": int(intended_range["min"]),
                    "intended_output_max": int(intended_range["max"]),
                    "seed": seed,
                    "actual_output_tokens": int(raw_row["actual_output_tokens"]),
                    "predicted_log1p_mu": float(raw_row["predicted_log1p_mu"]),
                    "predicted_mean_output_tokens": float(raw_row["predicted_mean_output_tokens"]),
                }
            )
            seen.add(key)

    missing = expected - seen
    if missing:
        prompt_id, seed = sorted(missing)[0]
        raise SystemExit(
            f"incomplete {split} predictions: missing {len(missing)} of {len(expected)} rows; "
            f"first missing prompt_id={prompt_id}, seed={seed}"
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty report: {path}")
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _analyze_split(
    *,
    split: str,
    records: list[dict[str, Any]],
    residual_variance: float,
    evaluation_dir: Path,
    output_dir: Path,
    experiment_id: str,
) -> tuple[Path, Path, Path, Path, Path]:
    predictions_path = evaluation_dir / f"{split}_evaluation.csv"
    rows = _load_prediction_rows(predictions_path, split=split, records=records)
    breakdowns = build_group_breakdowns(rows, residual_variance)
    matched = build_matched_length_analysis(rows)
    report = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "split": split,
        "source_predictions": str(predictions_path),
        "notes": {
            "intended_length": "Frozen prompt condition, not an observed output-length bin.",
            "family_mean_level": "Averages the three seeds before matched length contrasts.",
            "r_squared_subgroups": (
                "Can be unstable for narrow groups; interpret with count and absolute errors."
            ),
        },
        **breakdowns,
        "prompt_mean_point_analysis": build_prompt_mean_point_analysis(rows),
        "seed_stability": build_seed_stability(rows),
        "matched_length_analysis": matched,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{split}_breakdown.json"
    csv_path = output_dir / f"{split}_breakdown.csv"
    contrasts_path = output_dir / f"{split}_length_contrasts.csv"
    prompt_mean_path = output_dir / f"{split}_prompt_mean_breakdown.csv"
    markdown_path = output_dir / f"{split}_breakdown.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, flatten_group_breakdowns(breakdowns))
    _write_csv(contrasts_path, flatten_matched_length_analysis(matched))
    _write_csv(
        prompt_mean_path,
        flatten_prompt_mean_point_analysis(report["prompt_mean_point_analysis"]),
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, csv_path, contrasts_path, prompt_mean_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=["train"],
        help="Analyze one or both existing evaluation CSV files.",
    )
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    if "test" in args.splits and not args.confirm_final_test:
        raise SystemExit("refusing final-test analysis without --confirm-final-test")

    experiment = load_experiment(args.experiment)
    records = load_frozen_prompts(experiment)
    prior_payload = json.loads(args.prior.read_text(encoding="utf-8"))
    if prior_payload.get("fit_split") != "train":
        raise SystemExit("prior was not fitted exclusively on the training split")
    residual_variance = float(prior_payload["residual_variance"])
    evaluation_dir = args.evaluation_dir or args.prior.parent
    output_dir = args.output_dir or args.prior.parent

    for split in args.splits:
        outputs = _analyze_split(
            split=split,
            records=records,
            residual_variance=residual_variance,
            evaluation_dir=evaluation_dir,
            output_dir=output_dir,
            experiment_id=experiment["experiment_id"],
        )
        print(f"analyzed {split} predictions: " + ", ".join(str(output) for output in outputs))


if __name__ == "__main__":
    main()
