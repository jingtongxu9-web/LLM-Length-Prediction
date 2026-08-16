"""Create the Stage-8B ready lock from frozen models and a passing audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from llm_length_prediction.stage8_lock import build_ready_lock

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")
DEFAULT_MODEL_ROOT = Path("artifacts/runs/bayesian_sequential_v1/final_models")
DEFAULT_AUDIT = Path(
    "docs/results/bayesian_sequential/stage8b_final_holdout_overlap_review.json"
)
DEFAULT_OUTPUT = Path("configs/experiments/bayesian_sequential_stage8b_lock_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--git-commit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    git_commit = args.git_commit or subprocess.run(
        ("git", "rev-parse", "origin/main"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = build_ready_lock(
        config_path=args.config,
        model_root=args.model_root,
        audit_report_path=args.audit_report,
        git_commit=git_commit,
    )
    content = json.dumps(lock, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output.is_file() and args.output.read_text(encoding="utf-8") != content:
        raise SystemExit(f"refusing to overwrite a different Stage-8B lock: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(content, end="")


if __name__ == "__main__":
    main()
