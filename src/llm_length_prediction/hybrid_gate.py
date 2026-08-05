"""One-way final-Test access gate for the frozen Hybrid v3 protocol."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_length_prediction.data.hybrid import hybrid_trace_path
from llm_length_prediction.experiment import file_sha256, rollout_jobs
from llm_length_prediction.hybrid_experiment import (
    load_hybrid_config,
    load_hybrid_experiment,
)
from llm_length_prediction.models.hybrid_suite import METHOD_IDS


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_hybrid_test_gate(
    *, protocol_path: Path, config_path: Path, trace_root: Path | None = None
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = load_hybrid_config(config_path)
    _, records = load_hybrid_experiment(config)
    run_root = Path(config["outputs"]["run_root"])
    root = trace_root or Path(config["outputs"]["trace_root"])
    oof_path = run_root / protocol["outputs"]["oof_report"]
    registry_path = run_root / protocol["outputs"]["models"] / "model_registry.json"
    checks_path = run_root / "validation" / "pre_open_checks.json"
    for path in (oof_path, registry_path, checks_path):
        if not path.is_file():
            raise ValueError(f"final-Test gate is missing {path}")
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    config_hash = file_sha256(config_path)
    protocol_hash = file_sha256(protocol_path)
    if oof.get("config_sha256") != config_hash or registry.get("config_sha256") != config_hash:
        raise ValueError("config changed after OOF/model freeze")
    if (
        oof.get("protocol_sha256") != protocol_hash
        or registry.get("protocol_sha256") != protocol_hash
    ):
        raise ValueError("protocol changed after OOF/model freeze")
    if checks.get("config_sha256") != config_hash or checks.get("protocol_sha256") != protocol_hash:
        raise ValueError("lint/test record does not match the frozen protocol")
    if checks.get("pytest") != "passed" or checks.get("ruff") != "passed":
        raise ValueError("pytest and ruff must pass immediately before opening Test")
    if (
        tuple(oof.get("methods", {})) != METHOD_IDS
        or tuple(registry.get("methods", {})) != METHOD_IDS
    ):
        raise ValueError("OOF/model registry does not contain exactly eight methods")
    if oof.get("test_opened") is not False or registry.get("test_opened") is not False:
        raise ValueError("pre-Test artifacts must declare test_opened=false")
    if oof.get("training_dataset_digest") != registry.get("training_dataset_digest"):
        raise ValueError("OOF and final models were not trained on identical Train traces")
    model_root = registry_path.parent
    for method in METHOD_IDS:
        for name, expected in registry["methods"][method]["sha256"].items():
            path = model_root / name
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"frozen model artifact changed or is missing: {path}")
    existing_test = [
        str(path)
        for record, seed in rollout_jobs(records, split="test")
        if (path := hybrid_trace_path(root, record, seed)).exists()
    ]
    if existing_test:
        raise ValueError(
            "Test traces exist before the one-way gate was opened; first: " + existing_test[0]
        )
    return {
        "protocol_id": protocol["protocol_id"],
        "config_sha256": config_hash,
        "protocol_sha256": protocol_hash,
        "oof_report_sha256": file_sha256(oof_path),
        "model_registry_sha256": file_sha256(registry_path),
        "training_dataset_digest": oof["training_dataset_digest"],
        "repo_commit": checks.get("repo_commit"),
    }


def begin_hybrid_test_access(
    *, protocol_path: Path, config_path: Path, trace_root: Path | None = None
) -> dict[str, Any]:
    config = load_hybrid_config(config_path)
    run_root = Path(config["outputs"]["run_root"])
    marker_path = run_root / "final_test" / "OPENED.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        for name in (
            "config_sha256",
            "protocol_sha256",
            "oof_report_sha256",
            "model_registry_sha256",
        ):
            expected = file_sha256(
                {
                    "config_sha256": config_path,
                    "protocol_sha256": protocol_path,
                    "oof_report_sha256": run_root / "oof" / "oof_report.json",
                    "model_registry_sha256": run_root / "models" / "model_registry.json",
                }[name]
            )
            if marker.get(name) != expected:
                raise ValueError(f"frozen artifact changed after Test opened: {name}")
        return marker
    evidence = validate_hybrid_test_gate(
        protocol_path=protocol_path, config_path=config_path, trace_root=trace_root
    )
    marker = {
        "schema_version": 1,
        "test_opened": True,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        **evidence,
    }
    _atomic_json(marker_path, marker)
    return marker
