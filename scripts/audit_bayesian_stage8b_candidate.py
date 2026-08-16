"""Audit the blind final-holdout candidate before creating the ready lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_length_prediction.stage8_lock import audit_final_holdout_candidate

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")
DEFAULT_CANDIDATE = Path("data/prompts/bayesian_sequential_v1_final_holdout.jsonl")
DEFAULT_MANUAL_REVIEW = Path(
    "configs/reviews/bayesian_sequential_stage8b_semantic_review_v1.json"
)
DEFAULT_OUTPUT = Path(
    "docs/results/bayesian_sequential/stage8b_final_holdout_overlap_review.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_MANUAL_REVIEW)
    parser.add_argument("--prompt-root", type=Path, default=Path("data/prompts"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_final_holdout_candidate(
        config_path=args.config,
        candidate_path=args.candidate,
        manual_review_path=args.manual_review,
        prompt_root=args.prompt_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if report["status"] != "pass":
        raise SystemExit("Stage-8B candidate overlap audit failed")


if __name__ == "__main__":
    main()
