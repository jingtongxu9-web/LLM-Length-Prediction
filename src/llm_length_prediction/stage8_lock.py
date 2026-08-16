"""Auditable Stage-8B candidate review and ready-lock construction."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage8_freeze import (
    LOCK_ID,
    MODEL_FILES,
    load_stage8a_config,
    validate_checkpoint_registry,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"manifest row {line_number} is not an object: {path}")
        records.append(payload)
    return records


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _semantic_core(record: dict[str, Any]) -> str:
    prompt = str(record["prompt"])
    if str(record["task_type"]) == "summarization" and "材料标题：" in prompt:
        prompt = "材料标题：" + prompt.split("材料标题：", 1)[1]
    elif "\n" in prompt:
        prompt = prompt.rsplit("\n", 1)[1]
    return "".join(character for character in prompt.casefold() if character.isalnum())


def _trigrams(value: str) -> set[str]:
    if len(value) < 3:
        return {value} if value else set()
    return {value[index : index + 3] for index in range(len(value) - 2)}


def _trigram_jaccard(left: str, right: str) -> float:
    left_grams = _trigrams(left)
    right_grams = _trigrams(right)
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 1.0


def _validate_candidate_grid(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    plan = config["final_holdout_plan"]
    required = {
        "dataset_version",
        "prompt_family_id",
        "prompt_id",
        "task_type",
        "intended_length",
        "intended_output_tokens",
        "language",
        "split",
        "generation_seeds",
        "prompt",
        "provenance",
    }
    for line_number, record in enumerate(records, 1):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"candidate row {line_number} is missing {sorted(missing)}")
        if (
            record["split"] != "final_holdout"
            or record["provenance"] != "new_blind_final_holdout_v1"
            or record["generation_seeds"] != plan["seeds"]
            or record["language"] != "zh-CN"
            or not str(record["prompt"]).strip()
        ):
            raise ValueError(f"candidate row {line_number} violates its frozen data role")
    if len(records) != int(plan["prompt_count"]):
        raise ValueError("candidate prompt count changed")
    prompt_ids = [str(record["prompt_id"]) for record in records]
    prompts = [_normalized(str(record["prompt"])) for record in records]
    if len(set(prompt_ids)) != len(prompt_ids) or len(set(prompts)) != len(prompts):
        raise ValueError("candidate IDs and normalized prompt texts must be unique")
    families = Counter(str(record["prompt_family_id"]) for record in records)
    expected_per_family = len(plan["intended_lengths"])
    if len(families) != int(plan["new_family_count"]) or set(families.values()) != {
        expected_per_family
    }:
        raise ValueError("candidate family grid is unbalanced")
    tasks = Counter(str(record["task_type"]) for record in records)
    expected_per_task = int(plan["families_per_task"]) * expected_per_family
    if set(tasks) != set(plan["tasks"]) or set(tasks.values()) != {expected_per_task}:
        raise ValueError("candidate task grid is unbalanced")
    cells = Counter(
        (str(record["task_type"]), str(record["intended_length"]))
        for record in records
    )
    expected_cells = {
        (task, length): int(plan["families_per_task"])
        for task in plan["tasks"]
        for length in plan["intended_lengths"]
    }
    if cells != expected_cells:
        raise ValueError("candidate task-by-length grid is unbalanced")
    family_roles: dict[str, tuple[str, set[str]]] = {}
    for record in records:
        family = str(record["prompt_family_id"])
        task = str(record["task_type"])
        length = str(record["intended_length"])
        if family not in family_roles:
            family_roles[family] = (task, set())
        if family_roles[family][0] != task:
            raise ValueError("a candidate family crosses task types")
        family_roles[family][1].add(length)
    if any(
        lengths != set(plan["intended_lengths"])
        for _, lengths in family_roles.values()
    ):
        raise ValueError("each candidate family must contain every intended length")
    return {
        "prompt_count": len(records),
        "family_count": len(families),
        "task_counts": dict(sorted(tasks.items())),
        "task_length_cells": {
            f"{task}:{length}": count
            for (task, length), count in sorted(cells.items())
        },
    }


def audit_final_holdout_candidate(
    *,
    config_path: str | Path,
    candidate_path: str | Path,
    manual_review_path: str | Path,
    prompt_root: str | Path,
) -> dict[str, Any]:
    """Review a candidate manifest without loading it through the holdout runtime."""

    config_path = Path(config_path)
    candidate_path = Path(candidate_path)
    manual_review_path = Path(manual_review_path)
    prompt_root = Path(prompt_root)
    config = load_stage8a_config(config_path)
    candidate = _read_jsonl(candidate_path)
    grid = _validate_candidate_grid(config, candidate)
    opened_paths = sorted(
        path
        for path in prompt_root.glob("*.jsonl")
        if path.resolve() != candidate_path.resolve()
    )
    if not opened_paths:
        raise ValueError("no opened manifests were found for overlap review")
    opened = [(path, record) for path in opened_paths for record in _read_jsonl(path)]
    candidate_ids = {str(record["prompt_id"]) for record in candidate}
    candidate_families = {str(record["prompt_family_id"]) for record in candidate}
    candidate_prompts = {_normalized(str(record["prompt"])) for record in candidate}
    opened_ids = {str(record.get("prompt_id")) for _, record in opened}
    opened_families = {str(record.get("prompt_family_id")) for _, record in opened}
    opened_prompts = {_normalized(str(record.get("prompt", ""))) for _, record in opened}
    exact = {
        "prompt_id_overlap": sorted(candidate_ids & opened_ids),
        "prompt_family_id_overlap": sorted(candidate_families & opened_families),
        "normalized_prompt_overlap_count": len(candidate_prompts & opened_prompts),
    }
    if any(exact.values()):
        raise ValueError(f"candidate has an exact opened-manifest overlap: {exact}")

    manual = _read_json(manual_review_path)
    reviews = manual.get("family_reviews")
    if (
        manual.get("status") != "manual_topic_review_complete"
        or manual.get("candidate_manifest") != str(candidate_path)
        or manual.get("prompt_semantic_overlap_review_complete") is not True
        or manual.get("final_holdout_opened") is not False
        or manual.get("final_holdout_accessed") is not False
        or not isinstance(reviews, list)
    ):
        raise ValueError("manual semantic review is incomplete")
    reviewed_families = {str(row.get("prompt_family_id")) for row in reviews}
    if reviewed_families != candidate_families or any(
        row.get("decision") != "pass"
        or str(row.get("nearest_opened_family")) not in opened_families
        or not str(row.get("rationale", "")).strip()
        for row in reviews
    ):
        raise ValueError("manual semantic family decisions are incomplete")
    thresholds = manual["thresholds"]
    sequence_limit = float(thresholds["maximum_sequence_ratio"])
    trigram_limit = float(thresholds["maximum_trigram_jaccard"])
    family_results = []
    for family in sorted(candidate_families):
        family_records = [
            record for record in candidate if str(record["prompt_family_id"]) == family
        ]
        best_sequence = (-1.0, "", "")
        best_trigram = (-1.0, "", "")
        for candidate_record in family_records:
            candidate_core = _semantic_core(candidate_record)
            for opened_path, opened_record in opened:
                opened_core = _semantic_core(opened_record)
                sequence = SequenceMatcher(None, candidate_core, opened_core).ratio()
                trigram = _trigram_jaccard(candidate_core, opened_core)
                identity = f"{opened_path.name}:{opened_record.get('prompt_id')}"
                if sequence > best_sequence[0]:
                    best_sequence = (sequence, identity, str(opened_record.get("prompt_family_id")))
                if trigram > best_trigram[0]:
                    best_trigram = (trigram, identity, str(opened_record.get("prompt_family_id")))
        passed = best_sequence[0] <= sequence_limit and best_trigram[0] <= trigram_limit
        family_results.append(
            {
                "prompt_family_id": family,
                "status": "pass" if passed else "fail",
                "maximum_sequence_ratio": best_sequence[0],
                "nearest_sequence_record": best_sequence[1],
                "nearest_sequence_family": best_sequence[2],
                "maximum_trigram_jaccard": best_trigram[0],
                "nearest_trigram_record": best_trigram[1],
                "nearest_trigram_family": best_trigram[2],
            }
        )
    failures = [row["prompt_family_id"] for row in family_results if row["status"] != "pass"]
    return {
        "schema_version": 1,
        "review_id": manual["review_id"],
        "status": "pass" if not failures else "fail",
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "stage8a_config_sha256": file_sha256(config_path),
        "manual_review": str(manual_review_path),
        "manual_review_sha256": file_sha256(manual_review_path),
        "opened_manifests": [
            {"path": str(path), "sha256": file_sha256(path)} for path in opened_paths
        ],
        "opened_record_count": len(opened),
        "opened_unique_family_count": len(opened_families),
        "candidate_grid": grid,
        "exact_overlap": exact,
        "similarity_thresholds": {
            "maximum_sequence_ratio": sequence_limit,
            "maximum_trigram_jaccard": trigram_limit,
        },
        "family_results": family_results,
        "failed_families": failures,
        "manual_topic_review_complete": True,
        "prompt_semantic_overlap_review_complete": not failures,
        "final_holdout_opened": False,
        "final_holdout_accessed": False,
    }


def build_ready_lock(
    *,
    config_path: str | Path,
    model_root: str | Path,
    audit_report_path: str | Path,
    git_commit: str,
) -> dict[str, Any]:
    """Build a ready lock only from a passing audit and valid final registry."""

    config_path = Path(config_path)
    model_root = Path(model_root)
    audit_report_path = Path(audit_report_path)
    config = load_stage8a_config(config_path)
    config_sha256 = file_sha256(config_path)
    config["_config_sha256"] = config_sha256
    registry = validate_checkpoint_registry(config, output_dir=model_root)
    registry_path = model_root / config["outputs"]["checkpoint_registry"]
    audit = _read_json(audit_report_path)
    manifest_path = Path(str(audit.get("candidate_manifest", "")))
    if (
        audit.get("status") != "pass"
        or audit.get("prompt_semantic_overlap_review_complete") is not True
        or audit.get("final_holdout_opened") is not False
        or audit.get("final_holdout_accessed") is not False
        or not manifest_path.is_file()
        or audit.get("candidate_manifest_sha256") != file_sha256(manifest_path)
    ):
        raise ValueError("Stage-8B semantic overlap audit is not frozen and passing")
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise ValueError("Stage-8B lock requires a full lowercase Git commit")
    remote_main = subprocess.run(
        ("git", "rev-parse", "origin/main"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", git_commit, remote_main),
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("the frozen implementation commit is not on origin/main")
    stage8b_preflight = Path("scripts/preflight_bayesian_stage8b_ready.py")
    if not stage8b_preflight.is_file():
        raise ValueError("Stage-8B ready preflight is missing")
    return {
        "schema_version": 1,
        "lock_id": LOCK_ID,
        "status": config["holdout_gate"]["required_lock_status"],
        "stage8a_config_sha256": config_sha256,
        "git_commit": git_commit,
        "checkpoint_registry_sha256": file_sha256(registry_path),
        "checkpoint_sha256": {
            name: registry["files"][name]["sha256"] for name in MODEL_FILES
        },
        "final_holdout_manifest": str(manifest_path),
        "final_holdout_manifest_sha256": file_sha256(manifest_path),
        "semantic_overlap_review": {
            "path": str(audit_report_path),
            "sha256": file_sha256(audit_report_path),
            "status": audit["status"],
            "opened_manifest_count": len(audit["opened_manifests"]),
            "opened_unique_family_count": audit["opened_unique_family_count"],
            "candidate_family_count": audit["candidate_grid"]["family_count"],
        },
        "stage8b_gate_preflight": {
            "path": str(stage8b_preflight),
            "sha256": file_sha256(stage8b_preflight),
        },
        "prompt_semantic_overlap_review_complete": True,
        "final_holdout_opened": False,
        "final_holdout_selects_nothing": True,
    }
