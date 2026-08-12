"""Validate the frozen Stage-4 archive before any Stage-5 model fitting."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from llm_length_prediction.bayesian_stage5 import (
    load_stage5_catalog,
    validate_stage5_grid,
)
from llm_length_prediction.experiment import file_sha256

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage5_oof_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--verify-trace-hashes",
        action="store_true",
        help="Rehash all 1,620 NPZ files; slower but appropriate for first local use.",
    )
    args = parser.parse_args()
    catalog = load_stage5_catalog(
        args.config,
        dataset_root=args.dataset_root,
        verify_trace_hashes=args.verify_trace_hashes,
    )
    grid = validate_stage5_grid(catalog)
    run_root = Path(catalog.config["outputs"]["run_root"])
    output = run_root / catalog.config["outputs"]["preflight"]
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    versions = {}
    for package in ("numpy", "torch", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"required Stage-5 package is missing: {package}")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        cuda_available = False
        gpu_name = None
    requested_device = "auto"
    warnings = (
        []
        if cuda_available
        else ["CUDA is unavailable; preflight passes, but full five-fold training will be slow"]
    )
    report = {
        **grid,
        "status": "pass",
        "ready": True,
        "config_sha256": file_sha256(args.config),
        "dataset_root": str(catalog.dataset_root),
        "trace_hashes_verified": args.verify_trace_hashes,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "package_versions": versions,
        "requested_device": requested_device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "folds": catalog.family_folds,
        "warnings": warnings,
        "failures": failures,
    }
    report["status"] = "failed" if failures else "pass"
    report["ready"] = not failures
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print(f"\nBayesian Stage-5 OOF preflight: {output}")
    if failures:
        raise SystemExit("Bayesian Stage-5 OOF preflight failed")


if __name__ == "__main__":
    main()
