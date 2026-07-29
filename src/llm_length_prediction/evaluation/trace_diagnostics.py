from __future__ import annotations

from pathlib import Path

from llm_length_prediction.data.io import read_trace_jsonl
from llm_length_prediction.evaluation.grouped_cv import DiagnosticRow
from llm_length_prediction.experiment import (
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
    trace_path,
    validate_frozen_trace,
)


def load_train_diagnostic_rows(experiment_path: Path) -> list[DiagnosticRow]:
    """Load and contract-check ALPS training traces without touching final test."""

    experiment = load_experiment(experiment_path)
    records = load_frozen_prompts(experiment)
    trace_root = Path(experiment["outputs"]["trace_root"])
    layer = int(experiment["model"]["feature_layer"])
    rows = []
    for record, seed in rollout_jobs(records, split="train"):
        path = trace_path(trace_root, record, seed)
        if not path.is_file():
            raise ValueError(f"training trace is missing: {path}")
        trace = read_trace_jsonl(path)[0]
        validate_frozen_trace(trace, record=record, seed=seed, experiment=experiment)
        rows.append(
            DiagnosticRow(
                prompt_id=record["prompt_id"],
                prompt_family_id=record["prompt_family_id"],
                task_type=record["task_type"],
                intended_length=record["intended_length"],
                prompt_tokens=trace.prompt_tokens,
                output_tokens=trace.output_tokens,
                hidden_state=tuple(trace.prefill_hidden_states[layer]),
            )
        )
    return rows
