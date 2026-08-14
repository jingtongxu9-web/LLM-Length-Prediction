"""Validate Stage-8A final-model fitting sources without opening a holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_freeze import (
    load_stage8_catalog,
    load_stage8a_config,
    validate_training_environment,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--verify-trace-hashes", action="store_true")
    parser.add_argument("--verify-training-environment", action="store_true")
    args = parser.parse_args()
    config = load_stage8a_config(args.config)
    catalog = load_stage8_catalog(
        config,
        dataset_root=args.dataset_root,
        verify_trace_hashes=args.verify_trace_hashes,
    )
    primary = [
        row
        for row in catalog.references
        if row.temperature == config["final_training"]["temperature"]
    ]
    report = {
        "stage8a_id": config["stage8a_id"],
        "status": "pass",
        "ready_for_final_model_training": True,
        "config_sha256": file_sha256(args.config),
        "dataset_digest": catalog.dataset_digest,
        "training_trace_count": len(primary),
        "training_family_count": len({row.prompt_family_id for row in primary}),
        "training_temperature": config["final_training"]["temperature"],
        "trace_hashes_verified": args.verify_trace_hashes,
        "training_environment_verified": args.verify_training_environment,
        "training_environment": (
            validate_training_environment(config) if args.verify_training_environment else None
        ),
        "final_holdout_status": config["final_holdout_plan"]["status"],
        "final_holdout_opened": False,
        "final_holdout_accessed": False,
        "warnings": [],
        "failures": [],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
