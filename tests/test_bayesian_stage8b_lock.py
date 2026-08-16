from __future__ import annotations

import json
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_freeze import LOCK_ID, MODEL_FILES
from llm_length_prediction.stage8_lock import audit_final_holdout_candidate

CONFIG = Path("configs/experiments/bayesian_sequential_stage8a_freeze_v1.json")
CANDIDATE = Path("data/prompts/bayesian_sequential_v1_final_holdout.jsonl")
MANUAL_REVIEW = Path("configs/reviews/bayesian_sequential_stage8b_semantic_review_v1.json")
AUDIT_REPORT = Path(
    "docs/results/bayesian_sequential/stage8b_final_holdout_overlap_review.json"
)
LOCK = Path("configs/experiments/bayesian_sequential_stage8b_lock_v1.json")


def test_stage8b_candidate_audit_is_deterministic_and_passing() -> None:
    report = audit_final_holdout_candidate(
        config_path=CONFIG,
        candidate_path=CANDIDATE,
        manual_review_path=MANUAL_REVIEW,
        prompt_root=Path("data/prompts"),
    )
    frozen = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    assert report == frozen
    assert report["status"] == "pass"
    assert report["candidate_grid"]["prompt_count"] == 36
    assert report["candidate_grid"]["family_count"] == 12
    assert report["opened_unique_family_count"] == 72
    assert report["exact_overlap"] == {
        "prompt_id_overlap": [],
        "prompt_family_id_overlap": [],
        "normalized_prompt_overlap_count": 0,
    }
    assert report["failed_families"] == []
    assert report["prompt_semantic_overlap_review_complete"] is True
    assert report["final_holdout_opened"] is False
    assert report["final_holdout_accessed"] is False


def test_stage8b_ready_lock_pins_every_blind_input() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["lock_id"] == LOCK_ID
    assert lock["status"] == "ready_for_one_time_final_holdout"
    assert lock["stage8a_config_sha256"] == file_sha256(CONFIG)
    assert lock["final_holdout_manifest"] == str(CANDIDATE)
    assert lock["final_holdout_manifest_sha256"] == file_sha256(CANDIDATE)
    assert lock["semantic_overlap_review"]["sha256"] == file_sha256(AUDIT_REPORT)
    preflight = lock["stage8b_gate_preflight"]
    assert preflight["sha256"] == file_sha256(preflight["path"])
    assert set(lock["checkpoint_sha256"]) == set(MODEL_FILES)
    assert all(len(digest) == 64 for digest in lock["checkpoint_sha256"].values())
    assert len(lock["git_commit"]) == 40
    assert lock["prompt_semantic_overlap_review_complete"] is True
    assert lock["final_holdout_opened"] is False
    assert lock["final_holdout_selects_nothing"] is True
