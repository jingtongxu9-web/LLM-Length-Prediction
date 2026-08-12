"""Generate frozen Stage-6 uncertainty, convergence, long-tail, and serving reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage6_analysis import (
    convergence_metrics,
    load_baseline_rows,
    load_posterior_rows,
    load_stage6_sources,
    long_tail_metrics,
    replay_selected_uncertainty_cone,
    runtime_metrics,
    serving_replay,
    strict_json_write,
    uncertainty_cone_curve_rows,
    uncertainty_curve_rows,
    uncertainty_findings,
    write_csv,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage6_analysis_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--stage5-root", type=Path, required=True)
    parser.add_argument("--verify-stage5-files", action="store_true")
    parser.add_argument("--verify-trace-hashes", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    sources = load_stage6_sources(
        args.config,
        stage4_root=args.stage4_root,
        stage5_root=args.stage5_root,
        verify_stage5_files=args.verify_stage5_files,
    )
    config = sources.config
    output = args.output_dir or Path(config["outputs"]["run_root"])
    output.mkdir(parents=True, exist_ok=True)
    posterior_methods = {
        method: load_posterior_rows(sources, method)
        for method in config["analysis"]["uncertainty_methods"]
    }
    selected_method = config["stage5"]["selected_method"]
    selected_rows = posterior_methods[selected_method]
    baseline_rows = load_baseline_rows(sources)
    curves = uncertainty_curve_rows(posterior_methods, config)
    write_csv(output / config["outputs"]["uncertainty_curves"], curves)
    cone, replay = replay_selected_uncertainty_cone(
        sources,
        selected_rows,
        verify_trace_hashes=args.verify_trace_hashes,
    )
    cone_curves = uncertainty_cone_curve_rows(cone, config)
    write_csv(output / config["outputs"]["uncertainty_cone"], cone_curves)
    convergence = {
        "definition": config["analysis"]["convergence"],
        "overall": convergence_metrics(
            selected_rows,
            threshold=float(config["analysis"]["convergence"]["threshold"]),
            group=lambda row: "all",
        )["all"],
        "by_temperature": convergence_metrics(
            selected_rows,
            threshold=float(config["analysis"]["convergence"]["threshold"]),
            group=lambda row: f"{float(row['temperature']):.1f}",
        ),
        "by_task": convergence_metrics(
            selected_rows,
            threshold=float(config["analysis"]["convergence"]["threshold"]),
            group=lambda row: str(row["task"]),
        ),
        "by_intended_length": convergence_metrics(
            selected_rows,
            threshold=float(config["analysis"]["convergence"]["threshold"]),
            group=lambda row: str(row["intended_length"]),
        ),
    }
    long_tail = long_tail_metrics(posterior_methods, baseline_rows, config)
    runtime = runtime_metrics(selected_rows, sources)
    serving = serving_replay(
        sources=sources,
        posterior_methods=posterior_methods,
        baseline_rows=baseline_rows,
        cone_rows=cone,
    )
    strict_json_write(output / config["outputs"]["serving_report"], serving)
    findings = uncertainty_findings(curves, config)
    selected_tail = long_tail["methods"][selected_method][
        "primary_temperature_empirical_top_decile_early"
    ]
    alps_tail = long_tail["methods"]["alps_countdown"][
        "primary_temperature_empirical_top_decile_early"
    ]
    findings["stable_5pct_convergence"] = {
        "success_rate": convergence["overall"]["success_rate"],
        "median_progress_on_success": convergence["overall"][
            "stable_progress_quantiles_on_success"
        ]["q50"],
        "predeclared_success_threshold": None,
        "interpretation": "diagnostic observation; no post-hoc pass threshold",
    }
    findings["long_tail_early_underestimation"] = {
        "empirical_top_decile_threshold_tokens": long_tail[
            "empirical_top_decile_threshold_tokens_from_primary_temperature"
        ],
        "selected_method_positive_underestimation_tokens": selected_tail[
            "sequence_balanced_positive_underestimation_tokens"
        ],
        "alps_positive_underestimation_tokens": alps_tail[
            "sequence_balanced_positive_underestimation_tokens"
        ],
        "selected_method_minus_alps_tokens": (
            selected_tail["sequence_balanced_positive_underestimation_tokens"]
            - alps_tail["sequence_balanced_positive_underestimation_tokens"]
        ),
        "selected_method_improves_over_alps": (
            selected_tail["sequence_balanced_positive_underestimation_tokens"]
            < alps_tail["sequence_balanced_positive_underestimation_tokens"]
        ),
    }
    mean_serving = serving["metrics"]["bayesian_entropy_scalar_v1_mean"]
    q975_serving = serving["metrics"]["bayesian_entropy_scalar_v1_q975"]
    findings["serving_tradeoff"] = {
        "posterior_mean_underallocation_rate": mean_serving["underallocation_rate"],
        "posterior_q975_underallocation_rate": q975_serving["underallocation_rate"],
        "posterior_mean_kv_overreservation_rate": mean_serving[
            "kv_overreservation_rate"
        ],
        "posterior_q975_kv_overreservation_rate": q975_serving[
            "kv_overreservation_rate"
        ],
        "serving_superiority_claimed": False,
        "scope": serving["scope"],
    }
    report = {
        "schema_version": 1,
        "stage6_id": config["stage6_id"],
        "status": "pass",
        "claim_scope": config["claim_scope"],
        "config_sha256": file_sha256(args.config),
        "dataset_digest": config["stage5"]["dataset_digest"],
        "source_validation": {
            "stage4_trace_count": len(sources.stage4_rows),
            "stage5_fold_count": len(sources.stage5_report["fold_reports"]),
            "stage5_sequence_count": len({
                (row["prompt_id"], float(row["temperature"]), int(row["seed"]))
                for row in selected_rows
            }),
            "stage5_observation_count": len(selected_rows),
            "selected_method": selected_method,
            "stage5_files_verified": args.verify_stage5_files,
            "stage5_manifest_file_count": config.get("_runtime_validation", {}).get(
                "stage5_manifest_file_count"
            ),
            "checkpoint_replay": replay,
        },
        "uncertainty": {
            "curve_rows": len(curves),
            "cone_curve_rows": len(cone_curves),
            "curve_artifact": config["outputs"]["uncertainty_curves"],
            "cone_artifact": config["outputs"]["uncertainty_cone"],
        },
        "convergence": convergence,
        "long_tail_underestimation": long_tail,
        "runtime": runtime,
        "serving_replay": serving,
        "scientific_findings": findings,
        "model_refit_performed": False,
        "robustness_refit_performed": False,
        "final_holdout_accessed": False,
    }
    strict_json_write(output / config["outputs"]["report"], report)
    summary = {
        "schema_version": 1,
        "stage6_id": config["stage6_id"],
        "status": "pass",
        "selected_method": selected_method,
        "uncertainty_findings": findings,
        "convergence": convergence,
        "long_tail_underestimation": long_tail,
        "runtime": runtime,
        "serving_replay_metrics": serving["metrics"],
        "model_refit_performed": False,
        "final_holdout_accessed": False,
    }
    strict_json_write(output / config["outputs"]["summary"], summary)
    print(
        f"Bayesian Stage-6 complete; status=pass selected={selected_method}; "
        f"report={output / config['outputs']['report']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
