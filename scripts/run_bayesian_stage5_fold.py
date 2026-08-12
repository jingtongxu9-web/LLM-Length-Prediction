"""Run one resumable family-grouped Stage-5 OOF fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_length_prediction.bayesian_stage5 import load_stage5_catalog, validate_stage5_grid
from llm_length_prediction.stage5_oof import run_stage5_fold

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage5_oof_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-trace-hashes", action="store_true")
    parser.add_argument("--skip-discriminative-baselines", action="store_true")
    args = parser.parse_args()
    catalog = load_stage5_catalog(
        args.config,
        dataset_root=args.dataset_root,
        verify_trace_hashes=args.verify_trace_hashes,
    )
    validate_stage5_grid(catalog)
    output = args.output_dir or (
        Path(catalog.config["outputs"]["run_root"])
        / catalog.config["outputs"]["folds"]
        / f"fold_{args.fold}"
    )
    marker = output / "fold_report.json"
    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "pass"
            and existing.get("dataset_digest") == catalog.dataset_digest
            and existing.get("fold") == args.fold
        ):
            print(f"Stage-5 fold {args.fold} already passed: {marker}")
            return
        raise SystemExit(f"refusing to overwrite an incompatible fold report: {marker}")
    print(
        f"starting Stage-5 fold={args.fold}; dataset={catalog.dataset_digest}; "
        f"output={output}",
        flush=True,
    )
    report = run_stage5_fold(
        catalog,
        fold=args.fold,
        output_dir=output,
        device=args.device,
        skip_discriminative_baselines=args.skip_discriminative_baselines,
    )
    print(
        f"completed Stage-5 fold={args.fold}; "
        f"train={report['training_trace_count']} validation={report['validation_trace_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
