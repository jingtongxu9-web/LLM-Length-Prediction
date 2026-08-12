"""Run frozen Train-family OOF Stage-7 error feedback without refitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.stage7_error_feedback import (
    audit_sequences,
    build_review_queue,
    load_stage7_sources,
    summarize_audit,
    write_jsonl,
    write_report,
)

DEFAULT_CONFIG = Path("configs/experiments/bayesian_sequential_stage7_error_feedback_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--stage5-root", type=Path, required=True)
    parser.add_argument("--verify-stage5-files", action="store_true")
    parser.add_argument("--verify-trace-hashes", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    sources = load_stage7_sources(
        args.config,
        stage4_root=args.stage4_root,
        stage5_root=args.stage5_root,
        verify_stage5_files=args.verify_stage5_files,
    )
    config = sources.config
    output = args.output_dir or Path(config["outputs"]["run_root"])
    output.mkdir(parents=True, exist_ok=True)
    rows, cohort_report = audit_sequences(sources, verify_trace_hashes=args.verify_trace_hashes)
    review_queue = build_review_queue(rows)
    summary = summarize_audit(rows)
    write_jsonl(output / config["outputs"]["sequence_audit"], rows)
    write_jsonl(output / config["outputs"]["review_queue"], review_queue)
    report = {
        "schema_version": 1,
        "stage7_id": config["stage7_id"],
        "status": "pass",
        "claim_scope": config["claim_scope"],
        "config_sha256": file_sha256(args.config),
        "dataset_digest": config["stage5"]["dataset_digest"],
        "selected_method": config["stage5"]["selected_method"],
        "cohort_definition": config["cohorts"],
        "cohort_results": cohort_report,
        "failure_analysis": summary,
        "sequence_audit_artifact": config["outputs"]["sequence_audit"],
        "manual_review_queue_artifact": config["outputs"]["review_queue"],
        "semantic_labels_are_not_automatically_resolved": True,
        "model_refit_performed": False,
        "method_reselection_performed": False,
        "threshold_tuning_performed": False,
        "changes_require_new_method_id": True,
        "final_holdout_accessed": False,
    }
    write_report(output / config["outputs"]["report"], report)
    compact = {
        "schema_version": 1,
        "stage7_id": config["stage7_id"],
        "status": "pass",
        "selected_method": config["stage5"]["selected_method"],
        "cohort_results": cohort_report,
        "failure_analysis": summary,
        "model_refit_performed": False,
        "changes_require_new_method_id": True,
        "final_holdout_accessed": False,
    }
    write_report(output / config["outputs"]["summary"], compact)
    print(
        f"Bayesian Stage-7 complete; status=pass review={len(review_queue)}; "
        f"report={output / config['outputs']['report']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
