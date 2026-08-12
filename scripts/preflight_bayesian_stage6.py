"""Validate frozen Stage-4/5 evidence before Stage-6 diagnostics."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage6_analysis import load_stage6_sources

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage6_analysis_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--stage5-root", type=Path, required=True)
    parser.add_argument("--verify-stage5-files", action="store_true")
    args = parser.parse_args()
    sources = load_stage6_sources(
        args.config,
        stage4_root=args.stage4_root,
        stage5_root=args.stage5_root,
        verify_stage5_files=args.verify_stage5_files,
    )
    output = Path(sources.config["outputs"]["run_root"]) / sources.config["outputs"][
        "preflight"
    ]
    report = {
        "stage6_id": sources.config["stage6_id"],
        "status": "pass",
        "ready": True,
        "config_sha256": file_sha256(args.config),
        "stage4_root": str(sources.stage4_root),
        "stage5_root": str(sources.stage5_root),
        "stage4_trace_count": len(sources.stage4_rows),
        "stage5_fold_count": len(sources.stage5_report["fold_reports"]),
        "stage5_sequence_count": sources.stage5_report["methods"][
            sources.config["stage5"]["selected_method"]
        ]["all_temperatures"]["sequence_count"],
        "stage5_observation_count": sources.stage5_report["methods"][
            sources.config["stage5"]["selected_method"]
        ]["all_temperatures"]["observation_count"],
        "selected_method": sources.selection["selected_method"],
        "stage5_files_verified": args.verify_stage5_files,
        "stage5_manifest_file_count": sources.config.get("_runtime_validation", {}).get(
            "stage5_manifest_file_count"
        ),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "model_refit_performed": False,
        "robustness_refit_performed": False,
        "final_holdout_accessed": False,
        "warnings": [],
        "failures": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print(f"\nBayesian Stage-6 preflight: {output}")


if __name__ == "__main__":
    main()
