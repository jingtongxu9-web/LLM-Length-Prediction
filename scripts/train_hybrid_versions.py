"""Fit deployable Hybrid v1/v2 models on all design-Train families after OOF."""

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
from llm_length_prediction.models.hybrid_versions import (
    METHOD_IDS,
    fit_hybrid_versions,
    save_hybrid_versions,
)

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_hybrid_versions.json")


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
    if tuple(protocol.get("methods", {})) != METHOD_IDS:
        raise SystemExit("Hybrid v1/v2 protocol does not match the implementation")
    run_root = args.run_root or Path(protocol["outputs"]["run_root"])
    oof_path = run_root / protocol["outputs"]["oof_report"]
    if not oof_path.is_file():
        raise SystemExit(f"run evaluate_hybrid_versions_oof.py first; missing {oof_path}")
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    if tuple(oof.get("methods", {})) != METHOD_IDS or oof.get("test_opened") is not False:
        raise SystemExit("Hybrid v1/v2 OOF report is incomplete or invalid")
    if oof.get("config_sha256") != file_sha256(args.config):
        raise SystemExit("method config changed after OOF")
    if oof.get("protocol_sha256") != file_sha256(args.protocol):
        raise SystemExit("Hybrid v1/v2 protocol changed after OOF")

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
    crossfit_folds = task_stratified_family_folds(
        samples,
        folds=int(policy["train_oof_folds"]),
        seed=int(policy["fold_seed"]) + 10_000,
    )
    prior = cross_fitted_prior_summaries(samples, crossfit_folds)
    fitted = fit_hybrid_versions(
        samples, prior, config=config, protocol=protocol, device=args.device
    )
    model_dir = run_root / protocol["outputs"]["models"]
    files = save_hybrid_versions(fitted, model_dir)
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
                "files": files[method],
                "sha256": {
                    name: file_sha256(model_dir / name) for name in files[method]
                },
            }
            for method in METHOD_IDS
        },
        "training_reports": fitted.reports,
        "holdout_status": "no_new_hybrid_holdout_has_been_opened",
    }
    (model_dir / "model_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"fitted and froze {len(METHOD_IDS)} Hybrid v1/v2 models in {model_dir}")


if __name__ == "__main__":
    main()
