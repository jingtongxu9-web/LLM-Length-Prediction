"""Validate frozen Stage-4/5 evidence before Stage-7 OOF error feedback."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage7_error_feedback import load_stage7_sources

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage7_error_feedback_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--stage5-root", type=Path, required=True)
    parser.add_argument("--verify-stage5-files", action="store_true")
    args = parser.parse_args()
    sources = load_stage7_sources(
        args.config,
        stage4_root=args.stage4_root,
        stage5_root=args.stage5_root,
        verify_stage5_files=args.verify_stage5_files,
    )
    config = sources.config
    output = Path(config["outputs"]["run_root"]) / config["outputs"]["preflight"]
    report = {
        "stage7_id": config["stage7_id"],
        "status": "pass",
        "ready": True,
        "config_sha256": file_sha256(args.config),
        "stage4_root": str(sources.stage6.stage4_root),
        "stage5_root": str(sources.stage6.stage5_root),
        "selected_method": sources.stage6.selection["selected_method"],
        "sequence_count": config["stage5"]["required_sequence_count"],
        "observation_count": config["stage5"]["required_observation_count"],
        "stage5_files_verified": args.verify_stage5_files,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "model_refit_performed": False,
        "method_reselection_performed": False,
        "final_holdout_accessed": False,
        "warnings": ["open_ended_prompt and hallucination require manual semantic review"],
        "failures": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print(f"\nBayesian Stage-7 preflight: {output}")


if __name__ == "__main__":
    main()
