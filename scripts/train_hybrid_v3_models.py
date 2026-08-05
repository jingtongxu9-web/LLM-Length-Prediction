"""Freeze all eight final Hybrid v3 models after grouped OOF is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_length_prediction.evaluation.hybrid import task_stratified_family_folds
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.hybrid_experiment import (
    enforce_censoring_policy,
    hybrid_dataset_digest,
    hybrid_samples,
    load_complete_hybrid_split,
    load_hybrid_config,
    load_hybrid_experiment,
    partition_censored,
    validate_hybrid_config,
)
from llm_length_prediction.models.hybrid import cross_fitted_prior_summaries
from llm_length_prediction.models.hybrid_suite import METHOD_IDS, fit_suite, save_suite

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_hybrid_v3_protocol.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_hybrid_config(args.config)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    run_root = args.run_root or Path(config["outputs"]["run_root"])
    oof_path = run_root / protocol["outputs"]["oof_report"]
    if not oof_path.is_file():
        raise SystemExit(f"run grouped OOF first; missing {oof_path}")
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    if tuple(oof.get("methods", {})) != METHOD_IDS or oof.get("test_opened") is not False:
        raise SystemExit("OOF report is incomplete or Test was already opened")
    if oof.get("config_sha256") != file_sha256(args.config):
        raise SystemExit("config changed after OOF")
    if oof.get("protocol_sha256") != file_sha256(args.protocol):
        raise SystemExit("protocol changed after OOF")
    loaded = load_complete_hybrid_split(
        config, experiment, records, split="train", trace_root=args.trace_root
    )
    effective, censored = partition_censored(loaded)
    censoring = enforce_censoring_policy(
        loaded_count=len(loaded),
        censored_count=censored,
        warning_rate=float(config["censoring"]["warning_rate"]),
        abort_rate=float(config["censoring"]["abort_rate"]),
    )
    digest = hybrid_dataset_digest(effective)
    if digest != oof.get("training_dataset_digest"):
        raise SystemExit("Train traces changed after OOF")
    samples = hybrid_samples(effective)
    policy = protocol["data_policy"]
    crossfit = task_stratified_family_folds(
        samples,
        folds=int(policy["train_oof_folds"]),
        seed=int(policy["fold_seed"]) + 10_000,
    )
    summaries = cross_fitted_prior_summaries(samples, crossfit)
    fitted = fit_suite(
        samples,
        summaries,
        config=config,
        protocol=protocol,
        device=args.device,
    )
    model_dir = run_root / protocol["outputs"]["models"]
    method_files = save_suite(fitted, model_dir)
    unique_files = sorted({name for names in method_files.values() for name in names})
    artifact_hashes = {name: file_sha256(model_dir / name) for name in unique_files}
    registry = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "test_opened": False,
        "config_sha256": file_sha256(args.config),
        "protocol_sha256": file_sha256(args.protocol),
        "oof_report_sha256": file_sha256(oof_path),
        "training_dataset_digest": digest,
        "censoring": censoring,
        "methods": {
            method: {
                "files": method_files[method],
                "sha256": {name: artifact_hashes[name] for name in method_files[method]},
            }
            for method in METHOD_IDS
        },
        "training_reports": {
            **fitted.reports,
            "plp_v2_frozen": fitted.plp_v2_metadata,
            "plp_small_terminal_v3": fitted.plp_small_metadata,
            "alps_plp_hybrid_v3": fitted.hybrid_metadata,
        },
    }
    (model_dir / "model_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"froze {len(METHOD_IDS)} methods in {model_dir}")


if __name__ == "__main__":
    main()
