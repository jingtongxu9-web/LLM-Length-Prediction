"""Verify every frozen final model and registry without opening the final holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_freeze import (
    MODEL_FILES,
    load_final_models,
    load_stage8a_config,
    validate_checkpoint_registry,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    config = load_stage8a_config(args.config)
    config["_config_sha256"] = file_sha256(args.config)
    root = args.model_root or Path(config["outputs"]["model_root"])
    registry = validate_checkpoint_registry(config, output_dir=root)
    models = load_final_models(config, output_dir=root, device="cpu")
    report = {
        "stage8a_id": config["stage8a_id"],
        "status": "pass",
        "config_sha256": config["_config_sha256"],
        "checkpoint_registry_sha256": file_sha256(root / config["outputs"]["checkpoint_registry"]),
        "primary_method": registry["primary_method"],
        "model_file_count": len(MODEL_FILES),
        "all_models_loaded_on_cpu": True,
        "loaded_bayesian_methods": sorted(models.bayesian_scorers),
        "final_holdout_opened": False,
        "final_holdout_accessed": False,
        "warnings": [],
        "failures": [],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
