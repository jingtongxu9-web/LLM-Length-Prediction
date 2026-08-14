"""Fail-closed contracts for the one-time Stage-8 final holdout."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_length_prediction.data.bayesian_trace import BayesianTraceV1
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_freeze import LOCK_ID, load_stage8a_config


@dataclass(frozen=True)
class FinalHoldoutJob:
    rank: int
    record: dict[str, Any]
    temperature: float
    seed: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL on line {line_number}: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"manifest row {line_number} must be an object")
        records.append(record)
    return records


def _normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def _existing_opened_records(manifest_path: Path) -> list[dict[str, Any]]:
    records = []
    prompt_root = manifest_path.parent
    if not prompt_root.is_dir():
        return records
    for path in sorted(prompt_root.glob("*.jsonl")):
        if path.resolve() == manifest_path.resolve():
            continue
        records.extend(_read_jsonl(path))
    return records


def load_final_holdout_contract(
    config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Load the frozen manifest only after a ready Stage-8B lock exists."""

    config_path = Path(config_path)
    config = load_stage8a_config(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    lock_path = Path(config["holdout_gate"]["benchmark_lock"])
    if not lock_path.is_file():
        raise ValueError("Stage-8B lock does not exist; final holdout must remain unopened")
    lock = _read_json(lock_path)
    if (
        lock.get("schema_version") != 1
        or lock.get("lock_id") != LOCK_ID
        or lock.get("status") != config["holdout_gate"]["required_lock_status"]
        or lock.get("stage8a_config_sha256") != config["_config_sha256"]
        or lock.get("prompt_semantic_overlap_review_complete") is not True
        or lock.get("final_holdout_opened") is not False
        or lock.get("final_holdout_selects_nothing") is not True
    ):
        raise ValueError("Stage-8B lock is not ready; final holdout must remain unopened")
    try:
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
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
        remote_main = subprocess.run(
            ("git", "rev-parse", "origin/main"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot verify the frozen Git state") from error
    if status or head != remote_main:
        raise ValueError("final holdout requires a clean checkout of current origin/main")
    frozen_commit = str(lock.get("git_commit", ""))
    if (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", frozen_commit, head), check=False
        ).returncode
        != 0
    ):
        raise ValueError("frozen Stage-8 implementation commit is not in current main")
    manifest_path = Path(str(lock.get("final_holdout_manifest", "")))
    expected_manifest_digest = lock.get("final_holdout_manifest_sha256")
    if (
        not manifest_path.is_file()
        or not isinstance(expected_manifest_digest, str)
        or file_sha256(manifest_path) != expected_manifest_digest
    ):
        raise ValueError("frozen final holdout manifest is missing or changed")
    records = _read_jsonl(manifest_path)
    plan = config["final_holdout_plan"]
    required = {
        "prompt_id",
        "prompt_family_id",
        "task_type",
        "intended_length",
        "prompt",
        "split",
        "provenance",
        "generation_seeds",
    }
    for index, record in enumerate(records, 1):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"final holdout row {index} is missing: {sorted(missing)}")
        if (
            record["split"] != "final_holdout"
            or record["provenance"] != "new_blind_final_holdout_v1"
            or record["generation_seeds"] != plan["seeds"]
            or not str(record["prompt"]).strip()
        ):
            raise ValueError(f"final holdout row {index} violates the frozen data role")
    if len(records) != plan["prompt_count"]:
        raise ValueError("final holdout prompt count changed")
    prompt_ids = [str(record["prompt_id"]) for record in records]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("final holdout prompt IDs must be unique")
    normalized_prompts = [_normalized_prompt(str(record["prompt"])) for record in records]
    if len(set(normalized_prompts)) != len(normalized_prompts):
        raise ValueError("final holdout prompt texts must be unique")
    families = Counter(str(record["prompt_family_id"]) for record in records)
    if len(families) != plan["new_family_count"] or set(families.values()) != {
        len(plan["intended_lengths"])
    }:
        raise ValueError("final holdout family grid is unbalanced")
    tasks = Counter(str(record["task_type"]) for record in records)
    if set(tasks) != set(plan["tasks"]) or set(tasks.values()) != {
        plan["families_per_task"] * len(plan["intended_lengths"])
    }:
        raise ValueError("final holdout task grid changed")
    cells = Counter(
        (str(record["task_type"]), str(record["intended_length"])) for record in records
    )
    expected_cells = {
        (task, length): plan["families_per_task"]
        for task in plan["tasks"]
        for length in plan["intended_lengths"]
    }
    if cells != expected_cells:
        raise ValueError("final holdout task-by-length cells are unbalanced")
    family_roles: dict[str, tuple[str, set[str]]] = {}
    for record in records:
        family = str(record["prompt_family_id"])
        task = str(record["task_type"])
        length = str(record["intended_length"])
        if family not in family_roles:
            family_roles[family] = (task, set())
        if family_roles[family][0] != task:
            raise ValueError("one final family cannot cross task types")
        family_roles[family][1].add(length)
    if any(lengths != set(plan["intended_lengths"]) for _, lengths in family_roles.values()):
        raise ValueError("each final family must contain short, medium, and long prompts")
    opened = _existing_opened_records(manifest_path)
    opened_ids = {str(record.get("prompt_id")) for record in opened}
    opened_families = {str(record.get("prompt_family_id")) for record in opened}
    opened_prompts = {_normalized_prompt(str(record.get("prompt", ""))) for record in opened}
    if set(prompt_ids).intersection(opened_ids):
        raise ValueError("final holdout prompt IDs overlap an existing manifest")
    if set(families).intersection(opened_families):
        raise ValueError("final holdout families overlap an existing manifest")
    if {_normalized_prompt(str(record["prompt"])) for record in records}.intersection(
        opened_prompts
    ):
        raise ValueError("final holdout contains an exact normalized opened prompt")
    return config, lock, records


def final_holdout_jobs(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> list[FinalHoldoutJob]:
    plan = config["final_holdout_plan"]
    rows = [
        (record, float(temperature), int(seed))
        for record in records
        for temperature in plan["temperatures"]
        for seed in plan["seeds"]
    ]
    rows.sort(
        key=lambda item: hashlib.sha256(
            (f"{LOCK_ID}|{item[0]['prompt_id']}|{item[1]:.3f}|{item[2]}").encode()
        ).hexdigest()
    )
    jobs = [
        FinalHoldoutJob(rank=index, record=record, temperature=temperature, seed=seed)
        for index, (record, temperature, seed) in enumerate(rows)
    ]
    if len(jobs) != plan["expected_rollout_count"]:
        raise ValueError("final holdout rollout count changed")
    return jobs


def validate_final_holdout_trace(
    trace: BayesianTraceV1,
    *,
    job: FinalHoldoutJob,
    config: dict[str, Any],
    lock_sha256: str,
    manifest_sha256: str,
) -> None:
    trace.validate(stride=5)
    record = job.record
    plan = config["final_holdout_plan"]
    expected = {
        "prompt_id": (trace.prompt_id, record["prompt_id"]),
        "prompt_family_id": (trace.prompt_family_id, record["prompt_family_id"]),
        "task": (trace.task, record["task_type"]),
        "intended_length": (trace.intended_length, record["intended_length"]),
        "split": (trace.split, "final_holdout"),
        "temperature": (trace.temperature, job.temperature),
        "top_p": (trace.top_p, plan["top_p"]),
        "seed": (trace.seed, job.seed),
        "max_new_tokens": (trace.max_new_tokens, plan["max_new_tokens"]),
        "model_revision": (
            trace.model_revision,
            "a09a35458c702b33eeacc393d103063234e8bc28",
        ),
        "tokenizer_revision": (trace.tokenizer_revision, trace.model_revision),
    }
    mismatches = [
        name for name, (actual, expected_value) in expected.items() if actual != expected_value
    ]
    if mismatches:
        raise ValueError(f"final holdout trace identity changed: {mismatches}")
    metadata = trace.metadata
    metadata_expected = {
        "stage8a_config_sha256": config["_config_sha256"],
        "stage8b_lock_sha256": lock_sha256,
        "final_holdout_manifest_sha256": manifest_sha256,
        "collection_job_rank": job.rank,
        "final_holdout_accessed": True,
    }
    changed = [
        name
        for name, expected_value in metadata_expected.items()
        if metadata.get(name) != expected_value
    ]
    if changed:
        raise ValueError(f"final holdout trace provenance changed: {changed}")


def build_final_holdout_summary(
    config: dict[str, Any],
    *,
    lock_sha256: str,
    manifest_sha256: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = int(config["final_holdout_plan"]["expected_rollout_count"])
    stops = Counter(str(row["stop_reason"]) for row in rows)
    censoring = stops.get("max_new_tokens", 0) / expected if expected else 0.0
    failures = []
    warnings = []
    if len(rows) != expected:
        failures.append("final holdout collection is incomplete")
    if any(row.get("split") != "final_holdout" for row in rows):
        failures.append("non-holdout trace entered final collection")
    if len(rows) >= int(config["final_holdout_plan"]["censoring_minimum_trace_count"]):
        if censoring > float(config["final_holdout_plan"]["censoring_abort_rate"]):
            failures.append("final holdout censoring exceeds the frozen abort rate")
        elif censoring > float(config["final_holdout_plan"]["censoring_warning_rate"]):
            warnings.append("final holdout censoring exceeds the frozen warning rate")
    return {
        "schema_version": 1,
        "collection_id": "bayesian-sequential-v1-final-holdout",
        "status": "pass" if not failures else "incomplete_or_failed",
        "stage8a_config_sha256": config["_config_sha256"],
        "stage8b_lock_sha256": lock_sha256,
        "final_holdout_manifest_sha256": manifest_sha256,
        "expected_trace_count": expected,
        "valid_trace_count": len(rows),
        "missing_trace_count": expected - len(rows),
        "by_stop_reason": dict(sorted(stops.items())),
        "censoring_rate": censoring,
        "total_observed_tokens": sum(int(row["observed_tokens"]) for row in rows),
        "total_duration_ms": sum(float(row["duration_ms"]) for row in rows),
        "total_trace_bytes": sum(int(row["trace_bytes"]) for row in rows),
        "warnings": warnings,
        "failures": failures,
        "final_holdout_accessed": bool(rows),
        "final_holdout_collection_complete": not failures,
    }
