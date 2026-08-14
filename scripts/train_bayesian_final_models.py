"""Fit frozen final comparison models on all opened Train families."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_freeze import (
    fit_final_models,
    load_stage8_catalog,
    load_stage8a_config,
    validate_checkpoint_registry,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-trace-hashes", action="store_true")
    args = parser.parse_args()
    config = load_stage8a_config(args.config)
    config["_config_sha256"] = file_sha256(args.config)
    catalog = load_stage8_catalog(
        config,
        dataset_root=args.dataset_root,
        verify_trace_hashes=args.verify_trace_hashes,
    )
    output = args.output_dir or Path(config["outputs"]["model_root"])
    registry_path = output / config["outputs"]["checkpoint_registry"]
    if registry_path.is_file():
        validate_checkpoint_registry(config, output_dir=output)
        print(f"Stage-8A final models already passed: {registry_path}")
        return
    existing = [path for path in output.glob("*") if path.is_file()]
    if existing:
        raise SystemExit(
            "refusing to overwrite an incomplete final-model directory: "
            + ", ".join(path.name for path in existing)
        )
    registry = fit_final_models(
        config,
        catalog,
        output_dir=output,
        device=args.device,
    )
    print(
        f"Stage-8A final models complete; primary={registry['primary_method']}; "
        f"registry={registry_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
