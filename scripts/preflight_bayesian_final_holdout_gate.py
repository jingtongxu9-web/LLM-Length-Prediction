"""Fail closed until the Stage-8B lock pins every final model and benchmark input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_freeze import (
    final_holdout_gate_report,
    load_final_models,
    load_stage8a_config,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--verify-model-loading", action="store_true")
    args = parser.parse_args()
    report = final_holdout_gate_report(args.config, model_root=args.model_root)
    if args.verify_model_loading:
        config = load_stage8a_config(args.config)
        config["_config_sha256"] = file_sha256(args.config)
        load_final_models(
            config,
            output_dir=args.model_root or Path(config["outputs"]["model_root"]),
            device="cpu",
        )
        report["all_final_models_loaded_on_cpu"] = True
    else:
        report["all_final_models_loaded_on_cpu"] = False
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit("final holdout gate is blocked; do not author or collect holdout")


if __name__ == "__main__":
    main()
