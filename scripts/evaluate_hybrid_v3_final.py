"""Evaluate all frozen methods exactly once on the unopened-family Test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_length_prediction.evaluation.hybrid import (
    absolute_step_breakdown,
    family_bootstrap_interval,
    family_macro_metrics,
    family_metric_rows,
    paired_family_mae_difference,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.hybrid_experiment import (
    enforce_censoring_policy,
    hybrid_samples,
    load_complete_hybrid_split,
    load_hybrid_config,
    load_hybrid_experiment,
    partition_censored,
    validate_hybrid_config,
)
from llm_length_prediction.models.hybrid_suite import METHOD_IDS, load_suite, predict_suite

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_hybrid_v3_protocol.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    config = load_hybrid_config(args.config)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    run_root = args.run_root or Path(config["outputs"]["run_root"])
    marker_path = run_root / "final_test" / "OPENED.json"
    registry_path = run_root / "models" / "model_registry.json"
    if not marker_path.is_file() or not registry_path.is_file():
        raise SystemExit("final Test gate/model registry is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if marker.get("test_opened") is not True:
        raise SystemExit("final Test gate is not open")
    if marker.get("model_registry_sha256") != file_sha256(registry_path):
        raise SystemExit("model registry changed after final Test opened")
    for method in METHOD_IDS:
        for name, expected in registry["methods"][method]["sha256"].items():
            if file_sha256(registry_path.parent / name) != expected:
                raise SystemExit(f"frozen model changed after gate: {name}")
    loaded = load_complete_hybrid_split(
        config, experiment, records, split="test", trace_root=args.trace_root
    )
    effective, censored = partition_censored(loaded)
    censoring = enforce_censoring_policy(
        loaded_count=len(loaded),
        censored_count=censored,
        warning_rate=float(config["censoring"]["warning_rate"]),
        abort_rate=float(config["censoring"]["abort_rate"]),
    )
    samples = hybrid_samples(effective)
    fitted = load_suite(registry_path.parent)
    predictions = predict_suite(
        fitted,
        samples,
        device="cpu",
        batch_size=int(config["training"]["batch_size"]),
    )
    bootstrap = protocol["evaluation"]["bootstrap"]
    familywise = protocol["evaluation"]["primary_familywise_rule"]
    methods = {}
    for index, method in enumerate(METHOD_IDS):
        family_rows = family_metric_rows(samples, predictions[method])
        family_mae = {
            str(row["prompt_family_id"]): float(row["sequence_balanced_mae_tokens"])
            for row in family_rows
        }
        methods[method] = {
            "metrics": family_macro_metrics(samples, predictions[method]),
            "family_macro_mae_ci_95": family_bootstrap_interval(
                family_mae,
                replicates=int(bootstrap["replicates"]),
                confidence=float(bootstrap["confidence_level"]),
                seed=int(bootstrap["seed"]) + index,
            ),
            "absolute_step_breakdown": absolute_step_breakdown(
                samples,
                predictions[method],
                boundaries=protocol["evaluation"]["absolute_step_bins"],
            ),
        }
    differences = {}
    for index, comparator in enumerate(METHOD_IDS[:-1]):
        differences[comparator] = {
            "ci_95": paired_family_mae_difference(
                samples,
                predictions["alps_plp_hybrid_v3"],
                predictions[comparator],
                replicates=int(bootstrap["replicates"]),
                confidence=float(bootstrap["confidence_level"]),
                seed=int(bootstrap["seed"]) + 100 + index,
            ),
            "primary_familywise_ci": paired_family_mae_difference(
                samples,
                predictions["alps_plp_hybrid_v3"],
                predictions[comparator],
                replicates=int(bootstrap["replicates"]),
                confidence=float(familywise["per_comparison_confidence_level"]),
                seed=int(bootstrap["seed"]) + 200 + index,
            ),
        }
    passed = all(
        comparison["primary_familywise_ci"]["upper"] < 0 for comparison in differences.values()
    )
    rows = []
    for index, sample in enumerate(samples):
        rows.append(
            {
                "prompt_id": sample.prompt_id,
                "prompt_family_id": sample.prompt_family_id,
                "task": sample.task,
                "intended_length": sample.intended_length,
                "seed": sample.seed,
                "step": sample.step,
                "output_tokens": sample.output_tokens,
                "remaining_tokens": sample.remaining_tokens,
                **{method: float(predictions[method][index]) for method in METHOD_IDS},
            }
        )
    output = run_root / protocol["outputs"]["final_test"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "split": "final_test",
        "test_opened": True,
        "test_access": marker,
        "censoring": censoring,
        "methods": methods,
        "hybrid_paired_differences": differences,
        "primary_claim_result": {
            "passed": passed,
            "rule": familywise["claim_passes_when"],
            "interpretation": (
                "prediction superiority supported on the frozen holdout"
                if passed
                else "the preregistered prediction superiority claim is not supported"
            ),
            "serving_superiority_requires_separate_benchmark": True,
        },
    }
    (output / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"final Test evaluated once; primary claim passed={passed}; output={output}")


if __name__ == "__main__":
    main()
