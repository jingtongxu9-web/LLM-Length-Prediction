"""One-way access gate for the PLP terminal-zero v3 final holdout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_length_prediction.data.hybrid import hybrid_trace_path
from llm_length_prediction.experiment import file_sha256, rollout_jobs
from llm_length_prediction.hybrid_experiment import load_hybrid_config, load_hybrid_experiment
from llm_length_prediction.hybrid_gate import _atomic_json, shared_holdout_owner_path
from llm_length_prediction.models.plp_v3 import PLP_V3_METHOD_IDS

PLP_V3_HOLDOUT_OWNER = "plp-terminal-zero-v3-confirmatory-2026"


def validate_plp_v3_test_gate(
    *, protocol_path: Path, config_path: Path, trace_root: Path | None = None
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PLP_V3_HOLDOUT_OWNER:
        raise ValueError("unexpected PLP-only v3 protocol")
    config = load_hybrid_config(config_path)
    _, records = load_hybrid_experiment(config)
    run_root = Path(protocol["outputs"]["run_root"])
    shared_run_root = Path(config["outputs"]["run_root"])
    root = trace_root or Path(config["outputs"]["trace_root"])
    if (shared_run_root / "final_test" / "OPENED.json").is_file():
        raise ValueError("the shared holdout was already opened by the Hybrid protocol")
    owner_path = shared_holdout_owner_path(config)
    if owner_path.is_file():
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        if owner.get("protocol_id") != PLP_V3_HOLDOUT_OWNER:
            raise ValueError(
                "the shared holdout belongs to " f"{owner.get('protocol_id')}"
            )
    selection_path = run_root / protocol["outputs"]["selection_report"]
    registry_path = run_root / protocol["outputs"]["models"] / "model_registry.json"
    checks_path = run_root / "validation" / "pre_open_checks.json"
    for path in (selection_path, registry_path, checks_path):
        if not path.is_file():
            raise ValueError(f"PLP final-Test gate is missing {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    config_hash = file_sha256(config_path)
    protocol_hash = file_sha256(protocol_path)
    if registry.get("config_sha256") != config_hash:
        raise ValueError("shared config changed after PLP model freeze")
    if registry.get("protocol_sha256") != protocol_hash:
        raise ValueError("PLP protocol changed after model freeze")
    if registry.get("selection_report_sha256") != file_sha256(selection_path):
        raise ValueError("PLP selection report changed after model freeze")
    if checks.get("config_sha256") != config_hash or checks.get("protocol_sha256") != protocol_hash:
        raise ValueError("pre-open checks do not match the frozen PLP protocol")
    if checks.get("pytest") != "passed" or checks.get("ruff") != "passed":
        raise ValueError("pytest and ruff must pass immediately before opening Test")
    if tuple(registry.get("methods", {})) != PLP_V3_METHOD_IDS:
        raise ValueError("PLP model registry must contain exactly two methods")
    if registry.get("test_opened") is not False or selection.get("test_opened") is not False:
        raise ValueError("pre-Test artifacts must declare test_opened=false")
    for method_id in PLP_V3_METHOD_IDS:
        item = registry["methods"][method_id]
        path = registry_path.parent / item["file"]
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError(f"frozen PLP model changed or is missing: {path}")
    existing_test = [
        str(path)
        for record, seed in rollout_jobs(records, split="test")
        if (path := hybrid_trace_path(root, record, seed)).exists()
    ]
    if existing_test:
        raise ValueError(
            "Test traces exist before the PLP one-way gate; first: " + existing_test[0]
        )
    return {
        "protocol_id": protocol["protocol_id"],
        "config_sha256": config_hash,
        "protocol_sha256": protocol_hash,
        "selection_report_sha256": file_sha256(selection_path),
        "model_registry_sha256": file_sha256(registry_path),
        "training_dataset_digest": registry["training_dataset_digest"],
        "repo_commit": checks.get("repo_commit"),
    }


def begin_plp_v3_test_access(
    *, protocol_path: Path, config_path: Path, trace_root: Path | None = None
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = load_hybrid_config(config_path)
    run_root = Path(protocol["outputs"]["run_root"])
    marker_path = run_root / protocol["outputs"]["final_test"] / "OPENED.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = {
            "config_sha256": file_sha256(config_path),
            "protocol_sha256": file_sha256(protocol_path),
            "selection_report_sha256": file_sha256(
                run_root / protocol["outputs"]["selection_report"]
            ),
            "model_registry_sha256": file_sha256(
                run_root / protocol["outputs"]["models"] / "model_registry.json"
            ),
        }
        for name, value in expected.items():
            if marker.get(name) != value:
                raise ValueError(f"frozen PLP artifact changed after Test opened: {name}")
        owner_path = shared_holdout_owner_path(config)
        if not owner_path.is_file():
            raise ValueError("shared holdout ownership marker is missing")
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        if owner.get("protocol_id") != PLP_V3_HOLDOUT_OWNER:
            raise ValueError("shared holdout ownership changed after PLP Test opened")
        return marker
    evidence = validate_plp_v3_test_gate(
        protocol_path=protocol_path, config_path=config_path, trace_root=trace_root
    )
    opened_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        shared_holdout_owner_path(config),
        {
            "schema_version": 1,
            "protocol_id": PLP_V3_HOLDOUT_OWNER,
            "assigned_at_utc": opened_at,
            "consequence": "future Hybrid confirmatory testing requires newly authored families",
        },
    )
    marker = {
        "schema_version": 1,
        "test_opened": True,
        "opened_at_utc": opened_at,
        **evidence,
    }
    _atomic_json(marker_path, marker)
    return marker
