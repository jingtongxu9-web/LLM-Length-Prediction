"""Frozen contract helpers for the stage-three Bayesian unified-trace pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_length_prediction.bayesian_contract import load_bayesian_contract
from llm_length_prediction.data.bayesian_trace import (
    BAYESIAN_TRACE_SCHEMA_NAME,
    BAYESIAN_TRACE_SCHEMA_VERSION,
    BayesianTraceV1,
)
from llm_length_prediction.experiment import file_sha256


@dataclass(frozen=True)
class BayesianPilotJob:
    record: dict[str, Any]
    temperature: float
    seed: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL on line {line_number}: {path}") from error
    return records


def load_bayesian_pilot(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Load and validate the pilot, scientific contract, and selected design prompts."""

    pilot_path = Path(path)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("schema_version") != 1:
        raise ValueError("unsupported Bayesian pilot schema_version")
    if pilot.get("method_id") != "bayesian-sequential-v1":
        raise ValueError("Bayesian pilot method_id changed")
    if pilot.get("stage") != "stage3_unified_collector_pilot":
        raise ValueError("Bayesian pilot stage changed")
    if pilot.get("status") not in {"ready_for_server_execution", "server_pilot_complete"}:
        raise ValueError("unsupported Bayesian pilot status")

    reference = pilot["scientific_contract"]
    contract_path = Path(reference["path"])
    if file_sha256(contract_path) != reference["sha256"]:
        raise ValueError("Bayesian scientific contract digest changed after pilot freeze")
    contract = load_bayesian_contract(contract_path)
    generation = pilot["generation"]
    contract_generation = contract["generation"]
    expected_generation = {
        "model": (pilot["model"]["id"], contract_generation["model"]),
        "revision": (pilot["model"]["revision"], contract_generation["revision"]),
        "max_new_tokens": (
            generation["max_new_tokens"],
            contract_generation["max_new_tokens"],
        ),
        "top_p": (generation["top_p"], contract_generation["top_p"]),
        "primary_temperature": (
            generation["temperatures"],
            [contract_generation["primary_temperature"]],
        ),
    }
    mismatches = [
        name
        for name, (actual, expected) in expected_generation.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(f"pilot differs from scientific contract: {mismatches}")
    if pilot["model"]["prior_layer_zero_based"] != contract["prior"][
        "feature_layer_zero_based"
    ]:
        raise ValueError("pilot prior layer differs from the scientific contract")
    trace = pilot["trace"]
    if trace["schema_name"] != BAYESIAN_TRACE_SCHEMA_NAME:
        raise ValueError("pilot trace schema_name changed")
    if trace["schema_version"] != BAYESIAN_TRACE_SCHEMA_VERSION:
        raise ValueError("pilot trace schema_version changed")
    if trace["stride"] != contract["updates"]["nominal_stride"]:
        raise ValueError("pilot stride differs from the scientific contract")
    if trace["probability_source"] != contract["evidence"]["probability_source"]:
        raise ValueError("pilot probability source differs from the scientific contract")
    if trace["evidence_unit"] != contract["evidence"]["unit"]:
        raise ValueError("pilot evidence unit differs from the scientific contract")

    source = pilot["source_prompts"]
    if source["new_final_holdout_access"] != "forbidden":
        raise ValueError("pilot cannot access the future final holdout")
    prompt_path = Path(source["path"])
    if file_sha256(prompt_path) != source["sha256"]:
        raise ValueError("source prompt manifest digest changed")
    all_records = _read_jsonl(prompt_path)
    families = set(source["selected_families"])
    records = [
        record
        for record in all_records
        if record.get("prompt_family_id") in families
        and record.get("split") == source["allowed_split"]
    ]
    if len(records) != source["expected_prompt_count"]:
        raise ValueError("pilot prompt count does not match selected families")
    if {record["prompt_family_id"] for record in records} != families:
        raise ValueError("one or more selected pilot families are missing")
    tasks = {record["task_type"] for record in records}
    lengths = {record["intended_length"] for record in records}
    if tasks != set(source["required_tasks"]):
        raise ValueError("pilot does not cover every required task")
    if lengths != set(source["required_intended_lengths"]):
        raise ValueError("pilot does not cover every intended length")
    cells = {(record["task_type"], record["intended_length"]) for record in records}
    if len(cells) != pilot["acceptance"]["required_task_length_cells"]:
        raise ValueError("pilot does not cover the complete task-by-length grid")
    if any(record.get("provenance") != "opened_v1_design_data" for record in records):
        raise ValueError("pilot may use only previously opened design families")
    expected_rollouts = len(records) * len(generation["temperatures"]) * len(
        generation["seeds"]
    )
    if expected_rollouts != generation["expected_rollout_count"]:
        raise ValueError("pilot expected_rollout_count is inconsistent")
    report_schema = pilot["acceptance_report_schema"]
    report_schema_path = Path(report_schema["path"])
    if file_sha256(report_schema_path) != report_schema["sha256"]:
        raise ValueError("pilot acceptance report schema digest changed")
    schema = json.loads(report_schema_path.read_text(encoding="utf-8"))
    if schema.get("$id") != "bayesian-sequential-pilot-report-v1":
        raise ValueError("unsupported pilot acceptance report schema")
    return pilot, contract, records


def bayesian_pilot_jobs(
    pilot: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[BayesianPilotJob]:
    return [
        BayesianPilotJob(record=record, temperature=float(temperature), seed=int(seed))
        for record in records
        for temperature in pilot["generation"]["temperatures"]
        for seed in pilot["generation"]["seeds"]
    ]


def validate_bayesian_pilot_trace(
    trace: BayesianTraceV1,
    *,
    job: BayesianPilotJob,
    pilot: dict[str, Any],
    require_cuda: bool = True,
) -> None:
    trace.validate(stride=int(pilot["trace"]["stride"]))
    record = job.record
    model = pilot["model"]
    generation = pilot["generation"]
    expected = {
        "prompt_id": (trace.prompt_id, record["prompt_id"]),
        "prompt_family_id": (trace.prompt_family_id, record["prompt_family_id"]),
        "task": (trace.task, record["task_type"]),
        "intended_length": (trace.intended_length, record["intended_length"]),
        "split": (trace.split, record["split"]),
        "temperature": (trace.temperature, job.temperature),
        "top_p": (trace.top_p, generation["top_p"]),
        "seed": (trace.seed, job.seed),
        "max_new_tokens": (trace.max_new_tokens, generation["max_new_tokens"]),
        "model_name": (trace.model_name, model["id"]),
        "model_revision": (trace.model_revision, model["revision"]),
        "tokenizer_revision": (trace.tokenizer_revision, model["tokenizer_revision"]),
    }
    errors = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    metadata_expected = {
        "pilot_id": pilot["pilot_id"],
        "scientific_contract_sha256": pilot["scientific_contract"]["sha256"],
        "source_prompt_manifest_sha256": pilot["source_prompts"]["sha256"],
        "trace_schema": pilot["trace"]["schema_name"],
        "trace_schema_version": pilot["trace"]["schema_version"],
        "trace_stride": pilot["trace"]["stride"],
        "prior_feature_layer": model["prior_layer_zero_based"],
        "prior_layer_indexing": "zero_based_transformer_block",
        "decode_hidden_layer": model["decode_layer"],
        "hidden_size": model["hidden_size"],
        "prompt_pooling": pilot["prompt_representation"]["pooling"],
        "prompt_pooling_temperature": pilot["prompt_representation"][
            "pooling_temperature"
        ],
        "probability_source": pilot["trace"]["probability_source"],
        "evidence_unit": pilot["trace"]["evidence_unit"],
        "storage_dtype": pilot["trace"]["storage_dtype"],
        "chat_template": generation["chat_template"],
        "output_length_includes_eos": True,
        "prompt_sha256": hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest(),
    }
    errors.extend(
        f"metadata.{name}"
        for name, wanted in metadata_expected.items()
        if trace.metadata.get(name) != wanted
    )
    if trace.prior_feature.shape != (model["hidden_size"],):
        errors.append("prior_feature.shape")
    if trace.prompt_feature.shape != (model["hidden_size"],):
        errors.append("prompt_feature.shape")
    if trace.initial_decode_hidden_state.shape != (model["hidden_size"],):
        errors.append("initial_decode_hidden_state.shape")
    if trace.decode_hidden_states.shape[1:] != (model["hidden_size"],):
        errors.append("decode_hidden_states.shape")
    if model["dtype"] not in str(trace.metadata.get("dtype")):
        errors.append("metadata.dtype")
    if require_cuda and pilot["acceptance"]["require_cuda_peak_memory_metadata"]:
        if not str(trace.metadata.get("device", "")).startswith("cuda"):
            errors.append("metadata.device")
        for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            if not isinstance(trace.metadata.get(name), int):
                errors.append(f"metadata.{name}")
    if errors:
        raise ValueError("Bayesian pilot trace mismatch: " + "; ".join(errors))
