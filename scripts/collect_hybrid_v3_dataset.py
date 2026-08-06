"""Collect shared v3 traces for Hybrid development or the selected PLP-only final Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from llm_length_prediction.data.hybrid import (
    hybrid_trace_path,
    read_hybrid_trace,
    write_hybrid_trace,
)
from llm_length_prediction.experiment import file_sha256, rollout_jobs
from llm_length_prediction.hybrid_experiment import (
    load_hybrid_config,
    load_hybrid_experiment,
    validate_hybrid_config,
    validate_hybrid_trace,
)
from llm_length_prediction.instrumentation.hybrid import HuggingFaceHybridV3Collector
from llm_length_prediction.runtime.model_paths import resolve_model_source

DEFAULT_CONFIG = Path("configs/experiments/alps_plp_hybrid_v3.json")
DEFAULT_PROTOCOL = Path("configs/experiments/alps_plp_hybrid_v3_protocol.json")
DEFAULT_PLP_PROTOCOL = Path("configs/experiments/plp_terminal_v3_protocol.json")


def _validate_local_revision(model_source: str, expected: str) -> None:
    path = Path(model_source)
    if not path.is_dir():
        return
    marker = path / ".frozen_revision"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected:
        raise ValueError("local model is missing the matching .frozen_revision marker")


def _atomic_trace(path: Path, trace: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        write_hybrid_trace(temporary, trace)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _valid_existing(
    path: Path,
    record: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        validate_hybrid_trace(
            read_hybrid_trace(path),
            record=record,
            seed=seed,
            config=config,
            experiment=experiment,
        )
    except (OSError, TypeError, ValueError):
        return False
    return True


def _write_index(
    root: Path,
    run_root: Path,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    counts = {"train": 0, "test": 0}
    reasons: Counter[str] = Counter()
    total_points = 0
    total_bytes = 0
    for record, seed in rollout_jobs(records):
        path = hybrid_trace_path(root, record, seed)
        if not path.is_file():
            continue
        trace = read_hybrid_trace(path)
        validate_hybrid_trace(
            trace,
            record=record,
            seed=seed,
            config=config,
            experiment=experiment,
        )
        counts[record["split"]] += 1
        reasons[trace.stop_reason] += 1
        total_points += len(trace.steps)
        total_bytes += path.stat().st_size
        rows.append(
            {
                "split": record["split"],
                "prompt_id": trace.prompt_id,
                "prompt_family_id": record["prompt_family_id"],
                "seed": seed,
                "output_tokens": trace.output_tokens,
                "point_count": len(trace.steps),
                "stop_reason": trace.stop_reason,
                "generated_token_ids_sha256": hashlib.sha256(
                    trace.generated_token_ids.tobytes()
                ).hexdigest(),
                "trace_sha256": file_sha256(path),
                "trace_path": str(path),
            }
        )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "collection_index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "completed": len(rows),
        "by_split": counts,
        "by_stop_reason": dict(sorted(reasons.items())),
        "censored_max_new_tokens": reasons["max_new_tokens"],
        "total_points": total_points,
        "total_trace_bytes": total_bytes,
        "total_trace_gib": total_bytes / 1024**3,
    }
    (run_root / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train"])
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument(
        "--test-owner",
        choices=("hybrid", "plp-terminal-v3"),
        default="hybrid",
        help="one protocol that irreversibly owns the shared 12-family holdout",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if "test" in args.splits and not args.confirm_final_test:
        raise SystemExit("refusing Hybrid v3 Test collection without --confirm-final-test")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config = load_hybrid_config(args.config)
    experiment, records = load_hybrid_experiment(config)
    validate_hybrid_config(config, experiment)
    root = Path(config["outputs"]["trace_root"])
    run_root = Path(config["outputs"]["run_root"])
    if "test" in args.splits:
        if args.test_owner == "plp-terminal-v3":
            from llm_length_prediction.plp_v3_gate import begin_plp_v3_test_access

            marker = begin_plp_v3_test_access(
                protocol_path=args.protocol or DEFAULT_PLP_PROTOCOL,
                config_path=args.config,
                trace_root=root,
            )
            print(f"PLP terminal-zero v3 Test marker: {marker['opened_at_utc']}")
        else:
            from llm_length_prediction.hybrid_gate import begin_hybrid_test_access

            marker = begin_hybrid_test_access(
                protocol_path=args.protocol or DEFAULT_PROTOCOL,
                config_path=args.config,
                trace_root=root,
            )
            print(f"Hybrid v3 Test marker: {marker['opened_at_utc']}")
    selected = set(args.splits)
    jobs = [job for job in rollout_jobs(records) if job[0]["split"] in selected]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    pending = []
    resumed = 0
    for record, seed in jobs:
        path = hybrid_trace_path(root, record, seed)
        if _valid_existing(path, record, seed, config, experiment):
            resumed += 1
        else:
            pending.append((record, seed))
    collector = None
    if pending:
        model_source = resolve_model_source(args.model)
        _validate_local_revision(model_source, experiment["model"]["revision"])
        collector = HuggingFaceHybridV3Collector(
            model_source,
            revision=experiment["model"]["revision"],
            dtype=experiment["model"]["dtype"],
            max_new_tokens=experiment["generation"]["max_new_tokens"],
            temperature=experiment["generation"]["temperature"],
            top_p=experiment["generation"]["top_p"],
            trace_stride=config["trace"]["stride"],
            entropy_window=experiment["generation"]["entropy_window"],
            pooling_temperature=config["representation"]["prompt_pooling_temperature"],
            prior_layer=config["representation"]["prior_layer"],
        )
    completed = 0
    for record, seed in pending:
        assert collector is not None
        collector.seed = seed
        trace = collector.collect_trace(
            record["prompt"], prompt_id=record["prompt_id"], task=record["task_type"]
        )
        trace.metadata.update(
            {
                "experiment_id": experiment["experiment_id"],
                "prompt_family_id": record["prompt_family_id"],
                "intended_length": record["intended_length"],
                "split": record["split"],
                "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
            }
        )
        validate_hybrid_trace(
            trace,
            record=record,
            seed=seed,
            config=config,
            experiment=experiment,
        )
        path = hybrid_trace_path(root, record, seed)
        _atomic_trace(path, trace)
        completed += 1
        print(f"completed Hybrid v3 {record['prompt_id']} seed={seed}: {path}")
    summary = _write_index(root, run_root, records, config, experiment)
    print(
        f"new={completed} resumed={resumed} indexed={summary['completed']} "
        f"by_split={summary['by_split']} total_gib={summary['total_trace_gib']:.3f}"
    )


if __name__ == "__main__":
    main()
