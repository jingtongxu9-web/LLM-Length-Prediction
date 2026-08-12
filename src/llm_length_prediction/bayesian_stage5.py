"""Frozen Stage-5 catalog, fold, and trace-loading contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_length_prediction.bayesian_full_train import (
    BayesianFullTrainJob,
    bayesian_full_train_jobs,
    load_bayesian_full_train,
    validate_bayesian_full_train_trace,
)
from llm_length_prediction.data.bayesian_trace import BayesianTraceV1, read_bayesian_trace
from llm_length_prediction.experiment import file_sha256

STAGE5_ID = "bayesian-sequential-v1-stage5-family-oof"


@dataclass(frozen=True)
class Stage5TraceRef:
    job: BayesianFullTrainJob
    path: Path
    trace_sha256: str
    observed_tokens: int
    stop_reason: str

    @property
    def prompt_id(self) -> str:
        return str(self.job.record["prompt_id"])

    @property
    def prompt_family_id(self) -> str:
        return str(self.job.record["prompt_family_id"])

    @property
    def task(self) -> str:
        return str(self.job.record["task_type"])

    @property
    def intended_length(self) -> str:
        return str(self.job.record["intended_length"])

    @property
    def temperature(self) -> float:
        return float(self.job.temperature)

    @property
    def seed(self) -> int:
        return int(self.job.seed)

    @property
    def identity(self) -> tuple[str, float, int]:
        return self.prompt_id, self.temperature, self.seed


@dataclass(frozen=True)
class Stage5Catalog:
    config: dict[str, Any]
    collection: dict[str, Any]
    collection_config_sha256: str
    dataset_root: Path
    references: tuple[Stage5TraceRef, ...]
    family_folds: dict[str, int]
    dataset_digest: str

    def load_trace(self, reference: Stage5TraceRef) -> BayesianTraceV1:
        trace = read_bayesian_trace(reference.path)
        validate_bayesian_full_train_trace(
            trace,
            job=reference.job,
            collection=self.collection,
            collection_config_sha256=self.collection_config_sha256,
        )
        return trace


def load_stage5_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("stage5_id") != STAGE5_ID:
        raise ValueError("unsupported Stage-5 OOF configuration")
    policy = payload["data_policy"]
    if policy.get("group_unit") != "prompt_family_id":
        raise ValueError("Stage-5 grouping must use prompt_family_id")
    if policy.get("training_temperature") != 0.7:
        raise ValueError("Stage-5 model fitting must use temperature 0.7")
    if policy.get("evaluation_temperatures") != [0.3, 0.7, 1.0]:
        raise ValueError("Stage-5 evaluation temperatures changed")
    if (
        policy.get("robustness_temperatures_are_evaluation_only") is not True
        or policy.get("robustness_refit_forbidden") is not True
    ):
        raise ValueError("robustness temperatures cannot participate in fitting")
    if "forbidden" not in policy.get("new_final_holdout_access", ""):
        raise ValueError("Stage-5 must not access a final holdout")
    contract = payload["scientific_contract"]
    if file_sha256(contract["path"]) != contract["sha256"]:
        raise ValueError("scientific contract changed after Stage-5 freeze")
    collection = payload["stage4_collection"]
    if file_sha256(collection["config"]) != collection["config_sha256"]:
        raise ValueError("Stage-4 collection config changed")
    return payload


def task_stratified_family_folds(
    references: list[Stage5TraceRef] | tuple[Stage5TraceRef, ...],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    family_tasks: dict[str, str] = {}
    for reference in references:
        previous = family_tasks.setdefault(reference.prompt_family_id, reference.task)
        if previous != reference.task:
            raise ValueError("one family cannot span task strata")
    by_task: dict[str, list[str]] = defaultdict(list)
    for family, task in family_tasks.items():
        by_task[task].append(family)
    import numpy as np

    generator = np.random.default_rng(seed)
    assignments: dict[str, int] = {}
    for task in sorted(by_task):
        families = sorted(by_task[task])
        generator.shuffle(families)
        for index, family in enumerate(families):
            assignments[family] = index % folds
    if set(assignments.values()) != set(range(folds)):
        raise ValueError("every outer fold must contain at least one family")
    return assignments


def _canonical_dataset_digest(rows: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        f"{int(row['job_rank'])}\t{row['prompt_id']}\t"
        f"{float(row['temperature']):.3f}\t{int(row['seed'])}\t{row['trace_sha256']}"
        for row in sorted(rows, key=lambda item: int(item["job_rank"]))
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_stage5_catalog(
    config_path: str | Path,
    *,
    dataset_root: str | Path,
    verify_trace_hashes: bool = False,
) -> Stage5Catalog:
    config = load_stage5_config(config_path)
    root = Path(dataset_root).expanduser().resolve()
    stage4 = config["stage4_collection"]
    report_path = root / "artifacts/runs/bayesian_sequential_v1/full_train/collection_report.json"
    index_path = root / "artifacts/runs/bayesian_sequential_v1/full_train/collection_index.jsonl"
    if not report_path.is_file() or not index_path.is_file():
        raise ValueError("dataset_root does not contain the frozen Stage-4 report and index")
    if file_sha256(report_path) != stage4["collection_report_sha256"]:
        raise ValueError("Stage-4 collection report digest differs from the frozen data")
    if file_sha256(index_path) != stage4["collection_index_sha256"]:
        raise ValueError("Stage-4 collection index digest differs from the frozen data")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "status": stage4["required_status"],
        "valid_trace_count": stage4["required_trace_count"],
        "missing_trace_count": 0,
        "valid_prompt_family_count": stage4["required_family_count"],
        "valid_task_length_cell_count": stage4["required_task_length_cells"],
        "full_train_collection_complete": True,
        "final_holdout_accessed": False,
    }
    changed = [name for name, value in required.items() if report.get(name) != value]
    if changed or report.get("warnings") or report.get("failures"):
        raise ValueError(f"Stage-4 acceptance report is not clean: {changed}")
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    digest = _canonical_dataset_digest(rows)
    if digest != stage4["dataset_digest"]:
        raise ValueError("Stage-4 canonical dataset digest differs from the Stage-5 freeze")
    collection, _, records = load_bayesian_full_train(stage4["config"])
    collection_sha256 = file_sha256(stage4["config"])
    jobs = bayesian_full_train_jobs(collection, records)
    rows_by_rank = {int(row["job_rank"]): row for row in rows}
    if len(rows_by_rank) != len(rows) or len(rows) != len(jobs):
        raise ValueError("Stage-4 index does not contain exactly one row per frozen job")
    references = []
    for job in jobs:
        row = rows_by_rank.get(job.rank)
        if row is None:
            raise ValueError(f"Stage-4 index is missing job rank {job.rank}")
        expected_identity = (job.record["prompt_id"], job.temperature, job.seed)
        actual_identity = (row["prompt_id"], float(row["temperature"]), int(row["seed"]))
        if actual_identity != expected_identity:
            raise ValueError(f"Stage-4 index identity mismatch at rank {job.rank}")
        relative = Path(row["trace_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Stage-4 trace paths must be safe relative paths")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Stage-4 trace is missing: {relative}")
        if verify_trace_hashes and file_sha256(path) != row["trace_sha256"]:
            raise ValueError(f"Stage-4 trace hash mismatch: {relative}")
        references.append(
            Stage5TraceRef(
                job=job,
                path=path,
                trace_sha256=str(row["trace_sha256"]),
                observed_tokens=int(row["observed_tokens"]),
                stop_reason=str(row["stop_reason"]),
            )
        )
    folds = task_stratified_family_folds(
        references,
        folds=int(config["data_policy"]["outer_oof_folds"]),
        seed=int(config["data_policy"]["fold_seed"]),
    )
    return Stage5Catalog(
        config=config,
        collection=collection,
        collection_config_sha256=collection_sha256,
        dataset_root=root,
        references=tuple(references),
        family_folds=folds,
        dataset_digest=digest,
    )


def validate_stage5_grid(catalog: Stage5Catalog) -> dict[str, Any]:
    references = catalog.references
    temperatures = catalog.config["data_policy"]["evaluation_temperatures"]
    seeds = catalog.collection["generation"]["seeds"]
    families = {reference.prompt_family_id for reference in references}
    cells = {f"{reference.task}:{reference.intended_length}" for reference in references}
    if len(references) != 1620 or len(families) != 60 or len(cells) != 9:
        raise ValueError("Stage-5 catalog lost frozen grid coverage")
    for family in families:
        family_rows = [
            reference for reference in references if reference.prompt_family_id == family
        ]
        if len(family_rows) != 27:
            raise ValueError(f"family {family} does not contain 27 frozen rollouts")
        if {reference.temperature for reference in family_rows} != set(temperatures):
            raise ValueError(f"family {family} lacks a frozen temperature")
        if {reference.seed for reference in family_rows} != set(seeds):
            raise ValueError(f"family {family} lacks a frozen seed")
    fold_family_counts = {
        str(fold): sum(value == fold for value in catalog.family_folds.values())
        for fold in range(int(catalog.config["data_policy"]["outer_oof_folds"]))
    }
    if any(count == 0 for count in fold_family_counts.values()):
        raise ValueError("a Stage-5 outer fold contains no families")
    if any(not math.isfinite(float(value)) for value in fold_family_counts.values()):
        raise ValueError("invalid fold counts")
    return {
        "stage5_id": STAGE5_ID,
        "dataset_digest": catalog.dataset_digest,
        "trace_count": len(references),
        "family_count": len(families),
        "task_length_cell_count": len(cells),
        "training_trace_count": sum(reference.temperature == 0.7 for reference in references),
        "evaluation_trace_count": len(references),
        "fold_family_counts": fold_family_counts,
        "final_holdout_accessed": False,
    }
