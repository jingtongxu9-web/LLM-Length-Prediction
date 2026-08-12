from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm_length_prediction.bayesian_full_train import bayesian_full_train_jobs
from llm_length_prediction.bayesian_stage5 import (
    Stage5TraceRef,
    load_stage5_config,
    task_stratified_family_folds,
)

CONFIG = Path("configs/experiments/bayesian_sequential_stage5_oof_v1.json")


def test_stage5_config_freezes_primary_fit_and_robustness_evaluation() -> None:
    config = load_stage5_config(CONFIG)
    policy = config["data_policy"]
    assert policy["training_temperature"] == 0.7
    assert policy["evaluation_temperatures"] == [0.3, 0.7, 1.0]
    assert policy["robustness_temperatures_are_evaluation_only"] is True
    assert policy["robustness_refit_forbidden"] is True
    assert "forbidden" in policy["new_final_holdout_access"]
    assert config["stage4_collection"]["required_trace_count"] == 1620


def test_stage5_config_rejects_robustness_refitting(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["data_policy"]["robustness_refit_forbidden"] = False
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot participate"):
        load_stage5_config(path)


def test_stage5_family_folds_keep_all_temperature_seed_and_length_rows_together() -> None:
    config = load_stage5_config(CONFIG)
    from llm_length_prediction.bayesian_full_train import load_bayesian_full_train

    collection, _, records = load_bayesian_full_train(
        config["stage4_collection"]["config"]
    )
    references = []
    for job in bayesian_full_train_jobs(collection, records):
        references.append(
            Stage5TraceRef(
                job=job,
                path=Path(f"trace-{job.rank}.npz"),
                trace_sha256=hashlib.sha256(str(job.rank).encode()).hexdigest(),
                observed_tokens=10,
                stop_reason="eos",
            )
        )
    folds = task_stratified_family_folds(references, folds=5, seed=20260810)
    assert len(folds) == 60
    assert {value for value in folds.values()} == {0, 1, 2, 3, 4}
    assert all(
        len({folds[row.prompt_family_id] for row in references if row.prompt_id == prompt_id})
        == 1
        for prompt_id in {row.prompt_id for row in references}
    )
    for task in ("qa", "summarization", "code"):
        assert {folds[row.prompt_family_id] for row in references if row.task == task} == {
            0,
            1,
            2,
            3,
            4,
        }
