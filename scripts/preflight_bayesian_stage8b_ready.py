"""Verify the merged Stage-8B lock and all transitive evidence without collecting."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_freeze import (
    final_holdout_gate_report,
    load_final_models,
    load_stage8a_config,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--verify-model-loading", action="store_true")
    args = parser.parse_args()
    config = load_stage8a_config(args.config)
    config["_config_sha256"] = file_sha256(args.config)
    model_root = args.model_root or Path(config["outputs"]["model_root"])
    report = final_holdout_gate_report(args.config, model_root=model_root)
    failures = list(report["failures"])
    lock_path = Path(config["holdout_gate"]["benchmark_lock"])
    lock = _read_json(lock_path) if lock_path.is_file() else {}
    review = lock.get("semantic_overlap_review", {})
    review_path = Path(str(review.get("path", "")))
    if (
        review.get("status") != "pass"
        or not review_path.is_file()
        or review.get("sha256") != file_sha256(review_path)
    ):
        failures.append("Stage-8B semantic-overlap evidence changed")
    own_pin = lock.get("stage8b_gate_preflight", {})
    own_path = Path(str(own_pin.get("path", "")))
    if (
        own_path != Path(__file__).resolve().relative_to(Path.cwd().resolve())
        or not own_path.is_file()
        or own_pin.get("sha256") != file_sha256(own_path)
    ):
        failures.append("Stage-8B ready preflight changed")
    try:
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        remote_main = subprocess.run(
            ("git", "rev-parse", "origin/main"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        failures.append(f"cannot verify merged clean Git state: {error}")
    else:
        if status or head != remote_main:
            failures.append("Stage-8B requires a clean checkout of current origin/main")
    all_models_loaded = False
    if args.verify_model_loading and not failures:
        load_final_models(config, output_dir=model_root, device="cpu")
        all_models_loaded = True
    result = {
        "stage8a_id": config["stage8a_id"],
        "status": "ready" if not failures else "blocked",
        "ready": not failures,
        "failures": failures,
        "semantic_overlap_evidence_verified": not any(
            "semantic-overlap" in failure for failure in failures
        ),
        "stage8b_preflight_hash_verified": not any(
            "ready preflight" in failure for failure in failures
        ),
        "clean_current_origin_main": not any(
            "clean checkout" in failure or "Git state" in failure for failure in failures
        ),
        "all_final_models_loaded_on_cpu": all_models_loaded,
        "final_holdout_opened": False,
        "final_holdout_accessed": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Stage-8B ready preflight is blocked; do not collect final holdout")


if __name__ == "__main__":
    main()
