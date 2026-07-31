from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm_length_prediction.data.io import read_trace_jsonl
from llm_length_prediction.data.schema import GenerationTrace
from llm_length_prediction.experiment import (
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
    trace_path,
    validate_frozen_trace,
)
from llm_length_prediction.models.dynamic import PLP_FEATURE_NAMES


def load_method_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported method config schema_version")
    return config


def load_base_experiment_for_method(
    method_config: dict[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    experiment_path = Path(method_config["base_experiment"])
    experiment = load_experiment(experiment_path)
    if experiment["experiment_id"] != method_config["base_experiment_id"]:
        raise ValueError(
            "method config base_experiment_id does not match the frozen ALPS experiment"
        )
    return experiment_path, experiment, load_frozen_prompts(experiment)


def load_complete_split_traces(
    experiment: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    split: str,
    trace_root: Path | None = None,
) -> list[tuple[dict[str, Any], int, Path, GenerationTrace]]:
    root = trace_root or Path(experiment["outputs"]["trace_root"])
    jobs = list(rollout_jobs(records, split=split))
    missing = [
        trace_path(root, record, seed)
        for record, seed in jobs
        if not trace_path(root, record, seed).is_file()
    ]
    if missing:
        raise ValueError(
            f"{split} collection is incomplete: missing {len(missing)} of {len(jobs)} traces; "
            f"first missing path: {missing[0]}"
        )

    loaded = []
    for record, seed in jobs:
        path = trace_path(root, record, seed)
        trace = read_trace_jsonl(path)[0]
        validate_frozen_trace(trace, record=record, seed=seed, experiment=experiment)
        loaded.append((record, seed, path, trace))
    return loaded


def trace_dataset_digest(
    loaded: list[tuple[dict[str, Any], int, Path, GenerationTrace]],
) -> str:
    canonical = "\n".join(
        f"{trace.prompt_id}\t{seed}\t{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for _, seed, path, trace in loaded
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_project_plp_contract(
    method_config: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    expected_scope = {
        "uses_existing_rollouts": True,
        "uses_alps_prior": False,
        "uses_prefill_hidden_state": False,
        "uses_decode_hidden_state": False,
        "paper_exact_plp": False,
    }
    mismatches = []
    for name, expected in expected_scope.items():
        actual = method_config.get("scope", {}).get(name)
        if actual != expected:
            mismatches.append(f"scope.{name}: expected {expected!r}, got {actual!r}")
    if tuple(method_config.get("features", ())) != PLP_FEATURE_NAMES:
        mismatches.append(
            f"features must be exactly {list(PLP_FEATURE_NAMES)!r}"
        )
    trace_config = method_config.get("trace", {})
    frozen_trace = {
        "stride": int(experiment["generation"]["trace_stride"]),
        "entropy_window": int(experiment["generation"]["entropy_window"]),
        "exclude_terminal_point": True,
        "sequence_balanced_loss": True,
    }
    for name, expected in frozen_trace.items():
        actual = trace_config.get(name)
        if actual != expected:
            mismatches.append(f"trace.{name}: expected {expected!r}, got {actual!r}")
    if method_config.get("target", {}).get("name") != "log1p_remaining_tokens":
        mismatches.append("target.name must be log1p_remaining_tokens")
    if method_config.get("training", {}).get("hyperparameter_selection") != "none":
        mismatches.append("Dynamic-Signal MLP v1 must not select hyperparameters from Test")
    if mismatches:
        raise ValueError("Dynamic-Signal MLP v1 contract mismatch: " + "; ".join(mismatches))
