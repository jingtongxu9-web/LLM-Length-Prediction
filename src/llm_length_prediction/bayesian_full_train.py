"""Frozen contracts and reporting for Bayesian stage-four full-Train collection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
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
class BayesianFullTrainJob:
    """One frozen prompt/temperature/seed Qwen rollout."""

    rank: int
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


def _job_digest(
    collection_id: str,
    record: dict[str, Any],
    temperature: float,
    seed: int,
) -> str:
    identity = (
        f"{collection_id}|{record['prompt_id']}|{temperature:.3f}|{seed}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_bayesian_full_train(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Load and strictly validate the stage-four collection contract."""

    config_path = Path(path)
    collection = json.loads(config_path.read_text(encoding="utf-8"))
    if collection.get("schema_version") != 1:
        raise ValueError("unsupported full-Train schema_version")
    if collection.get("collection_id") != "bayesian-sequential-v1-full-train":
        raise ValueError("full-Train collection_id changed")
    if collection.get("method_id") != "bayesian-sequential-v1":
        raise ValueError("full-Train method_id changed")
    if collection.get("stage") != "stage4_full_train_collection":
        raise ValueError("full-Train stage changed")
    if collection.get("status") != "ready_for_server_execution":
        raise ValueError("full-Train collection is not ready for server execution")
    source_pins = collection["implementation"]["required_source_sha256"]
    if not source_pins:
        raise ValueError("full-Train implementation source pins are missing")
    changed_sources = [
        source_path
        for source_path, expected_sha256 in source_pins.items()
        if file_sha256(source_path) != expected_sha256
    ]
    if changed_sources:
        raise ValueError(f"full-Train implementation source changed: {changed_sources}")

    reference = collection["scientific_contract"]
    contract_path = Path(reference["path"])
    if file_sha256(contract_path) != reference["sha256"]:
        raise ValueError("Bayesian scientific contract digest changed")
    contract = load_bayesian_contract(contract_path)

    gate = collection["stage3_pilot_gate"]
    gate_path = Path(gate["path"])
    if file_sha256(gate_path) != gate["sha256"]:
        raise ValueError("stage-three pilot summary digest changed")
    pilot_summary = json.loads(gate_path.read_text(encoding="utf-8"))
    if pilot_summary.get("status") != gate["required_status"]:
        raise ValueError("stage-three pilot did not pass")
    pilot_acceptance = pilot_summary["acceptance"]
    if pilot_acceptance.get("real_qwen_pilot_complete") is not gate[
        "required_real_qwen_pilot_complete"
    ]:
        raise ValueError("stage-three real Qwen pilot is incomplete")
    if pilot_acceptance.get("local_archive_and_trace_revalidation") != gate[
        "required_local_revalidation"
    ]:
        raise ValueError("stage-three local revalidation did not pass")

    model = collection["model"]
    generation = collection["generation"]
    contract_generation = contract["generation"]
    expected_pairs = {
        "model": (model["id"], contract_generation["model"]),
        "revision": (model["revision"], contract_generation["revision"]),
        "tokenizer_revision": (
            model["tokenizer_revision"],
            contract_generation["revision"],
        ),
        "dtype": (model["dtype"], "bfloat16"),
        "require_local_frozen_snapshot": (
            model["require_local_frozen_snapshot"],
            True,
        ),
        "prior_layer": (
            model["prior_layer_zero_based"],
            contract["prior"]["feature_layer_zero_based"],
        ),
        "max_new_tokens": (
            generation["max_new_tokens"],
            contract_generation["max_new_tokens"],
        ),
        "top_p": (generation["top_p"], contract_generation["top_p"]),
        "primary_temperature": (
            generation["primary_temperature"],
            contract_generation["primary_temperature"],
        ),
        "robustness_temperatures": (
            generation["robustness_temperatures"],
            contract_generation["robustness_temperatures"],
        ),
        "seeds": (generation["seeds"], contract_generation["seeds"]),
    }
    mismatches = [
        name for name, (actual, expected) in expected_pairs.items() if actual != expected
    ]
    if mismatches:
        raise ValueError(f"full-Train differs from scientific contract: {mismatches}")
    expected_temperatures = {
        generation["primary_temperature"],
        *generation["robustness_temperatures"],
    }
    if set(generation["temperatures"]) != expected_temperatures:
        raise ValueError("full-Train temperatures are incomplete")
    if len(generation["temperatures"]) != len(set(generation["temperatures"])):
        raise ValueError("full-Train temperatures must be unique")
    if len(generation["seeds"]) != len(set(generation["seeds"])):
        raise ValueError("full-Train seeds must be unique")
    if generation["temperature_is_model_input"]:
        raise ValueError("temperature must not become a model input")
    if not generation["robustness_refit_forbidden"]:
        raise ValueError("robustness temperatures cannot refit the method")

    trace = collection["trace"]
    trace_expected = {
        "schema_name": (trace["schema_name"], BAYESIAN_TRACE_SCHEMA_NAME),
        "schema_version": (trace["schema_version"], BAYESIAN_TRACE_SCHEMA_VERSION),
        "stride": (trace["stride"], contract["updates"]["nominal_stride"]),
        "probability_source": (
            trace["probability_source"],
            contract["evidence"]["probability_source"],
        ),
        "evidence_unit": (trace["evidence_unit"], contract["evidence"]["unit"]),
    }
    trace_mismatches = [
        name for name, (actual, expected) in trace_expected.items() if actual != expected
    ]
    if trace_mismatches:
        raise ValueError(f"full-Train trace contract changed: {trace_mismatches}")

    source = collection["source_prompts"]
    if source["allowed_split"] != "train":
        raise ValueError("full-Train may use only the Train split")
    if source["new_final_holdout_access"] != "forbidden":
        raise ValueError("full-Train cannot access the future final holdout")
    prompt_path = Path(source["path"])
    if file_sha256(prompt_path) != source["sha256"]:
        raise ValueError("source prompt manifest digest changed")
    records = [
        record
        for record in _read_jsonl(prompt_path)
        if record.get("split") == source["allowed_split"]
    ]
    if len(records) != source["expected_prompt_count"]:
        raise ValueError("full-Train prompt count changed")
    if len({record["prompt_id"] for record in records}) != len(records):
        raise ValueError("full-Train prompt IDs must be unique")
    if any(not record.get("prompt") for record in records):
        raise ValueError("full-Train prompts must not be empty")
    families = Counter(record["prompt_family_id"] for record in records)
    if len(families) != source["expected_prompt_family_count"]:
        raise ValueError("full-Train prompt-family count changed")
    expected_per_family = len(records) // source["expected_prompt_family_count"]
    if set(families.values()) != {expected_per_family}:
        raise ValueError("full-Train prompt families are unbalanced")
    if {record["task_type"] for record in records} != set(source["required_tasks"]):
        raise ValueError("full-Train task coverage changed")
    if {record["intended_length"] for record in records} != set(
        source["required_intended_lengths"]
    ):
        raise ValueError("full-Train length coverage changed")
    cells = Counter(
        (record["task_type"], record["intended_length"]) for record in records
    )
    if len(cells) != source["required_task_length_cells"]:
        raise ValueError("full-Train task-by-length grid changed")
    expected_per_cell = len(records) // source["required_task_length_cells"]
    if set(cells.values()) != {expected_per_cell}:
        raise ValueError("full-Train task-by-length cells are unbalanced")
    if any(
        record.get("provenance") != source["required_provenance"]
        for record in records
    ):
        raise ValueError("full-Train prompt provenance changed")
    if any(record.get("generation_seeds") != generation["seeds"] for record in records):
        raise ValueError("full-Train prompt seeds differ from the collection seeds")

    expected_rollouts = (
        len(records)
        * len(generation["temperatures"])
        * len(generation["seeds"])
    )
    if expected_rollouts != generation["expected_rollout_count"]:
        raise ValueError("full-Train expected_rollout_count is inconsistent")
    if expected_rollouts != collection["acceptance"]["expected_trace_count"]:
        raise ValueError("full-Train acceptance count is inconsistent")

    resumability = collection["resumability"]
    if resumability["invalid_existing_trace_policy"] != "fail_without_overwrite":
        raise ValueError("invalid full-Train traces must never be overwritten silently")
    if resumability["maximum_new_jobs_default"] <= 0:
        raise ValueError("maximum_new_jobs_default must be positive")

    budget = collection["budget"]
    if budget["basis_trace_count"] != pilot_acceptance["valid_trace_count"]:
        raise ValueError("budget pilot trace count changed")
    budget_evidence = {
        "basis_total_duration_ms": pilot_acceptance["total_duration_ms"],
        "basis_total_trace_bytes": pilot_acceptance["total_trace_bytes"],
        "basis_peak_cuda_reserved_bytes": pilot_acceptance[
            "peak_cuda_reserved_bytes"
        ],
    }
    for name, expected in budget_evidence.items():
        if budget[name] != expected:
            raise ValueError(f"budget evidence changed: {name}")
    projected_duration = (
        budget["basis_total_duration_ms"]
        / budget["basis_trace_count"]
        * expected_rollouts
    )
    if abs(projected_duration - budget["projected_duration_ms"]) > 1e-6:
        raise ValueError("projected full-Train duration is inconsistent")
    projected_bytes = round(
        budget["basis_total_trace_bytes"]
        / budget["basis_trace_count"]
        * expected_rollouts
    )
    if projected_bytes != budget["projected_compressed_trace_bytes"]:
        raise ValueError("projected full-Train trace bytes are inconsistent")
    minimum_free_gib = (
        budget["maximum_uncompressed_trace_gib"] * budget["disk_safety_multiplier"]
    )
    if budget["required_free_disk_gib_at_empty_start"] < minimum_free_gib:
        raise ValueError("full-Train free-disk budget lacks the frozen safety multiplier")

    report_schema = collection["acceptance_report_schema"]
    report_schema_path = Path(report_schema["path"])
    if file_sha256(report_schema_path) != report_schema["sha256"]:
        raise ValueError("full-Train report schema digest changed")
    schema = json.loads(report_schema_path.read_text(encoding="utf-8"))
    if schema.get("$id") != "bayesian-sequential-full-train-report-v1":
        raise ValueError("unsupported full-Train report schema")
    if schema["properties"]["expected_trace_count"].get("const") != expected_rollouts:
        raise ValueError("full-Train report schema count changed")
    if collection["outputs"]["trace_root"] != contract["outputs"]["trace_root"]:
        raise ValueError("full-Train trace root differs from the scientific contract")
    return collection, contract, records


def bayesian_full_train_jobs(
    collection: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[BayesianFullTrainJob]:
    """Enumerate all jobs in a deterministic, content-derived order."""

    raw_jobs = [
        (record, float(temperature), int(seed))
        for record in records
        for temperature in collection["generation"]["temperatures"]
        for seed in collection["generation"]["seeds"]
    ]
    raw_jobs.sort(
        key=lambda item: _job_digest(collection["collection_id"], *item)
    )
    return [
        BayesianFullTrainJob(
            rank=rank,
            record=record,
            temperature=temperature,
            seed=seed,
        )
        for rank, (record, temperature, seed) in enumerate(raw_jobs)
    ]


def validate_bayesian_full_train_trace(
    trace: BayesianTraceV1,
    *,
    job: BayesianFullTrainJob,
    collection: dict[str, Any],
    collection_config_sha256: str,
    require_cuda: bool = True,
) -> None:
    """Reject any trace that differs from its frozen job or collection contract."""

    trace.validate(stride=int(collection["trace"]["stride"]))
    record = job.record
    model = collection["model"]
    generation = collection["generation"]
    expected = {
        "prompt_id": (trace.prompt_id, record["prompt_id"]),
        "prompt_family_id": (trace.prompt_family_id, record["prompt_family_id"]),
        "task": (trace.task, record["task_type"]),
        "intended_length": (trace.intended_length, record["intended_length"]),
        "split": (trace.split, "train"),
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
        "collection_job_rank": job.rank,
        "collection_id": collection["collection_id"],
        "collection_stage": collection["stage"],
        "collection_config_sha256": collection_config_sha256,
        "scientific_contract_sha256": collection["scientific_contract"]["sha256"],
        "source_prompt_manifest_sha256": collection["source_prompts"]["sha256"],
        "stage3_pilot_summary_sha256": collection["stage3_pilot_gate"]["sha256"],
        "trace_schema": collection["trace"]["schema_name"],
        "trace_schema_version": collection["trace"]["schema_version"],
        "trace_stride": collection["trace"]["stride"],
        "prior_feature_layer": model["prior_layer_zero_based"],
        "prior_layer_indexing": "zero_based_transformer_block",
        "decode_hidden_layer": model["decode_layer"],
        "hidden_size": model["hidden_size"],
        "prompt_pooling": collection["prompt_representation"]["pooling"],
        "prompt_pooling_temperature": collection["prompt_representation"][
            "pooling_temperature"
        ],
        "probability_source": collection["trace"]["probability_source"],
        "evidence_unit": collection["trace"]["evidence_unit"],
        "storage_dtype": collection["trace"]["storage_dtype"],
        "chat_template": generation["chat_template"],
        "output_length_includes_eos": True,
        "final_holdout_accessed": False,
        "prompt_sha256": hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest(),
    }
    errors.extend(
        f"metadata.{name}"
        for name, wanted in metadata_expected.items()
        if trace.metadata.get(name) != wanted
    )
    hidden_size = int(model["hidden_size"])
    if trace.prior_feature.shape != (hidden_size,):
        errors.append("prior_feature.shape")
    if trace.prompt_feature.shape != (hidden_size,):
        errors.append("prompt_feature.shape")
    if trace.initial_decode_hidden_state.shape != (hidden_size,):
        errors.append("initial_decode_hidden_state.shape")
    if trace.decode_hidden_states.shape[1:] != (hidden_size,):
        errors.append("decode_hidden_states.shape")
    if str(trace.metadata.get("dtype")) != "torch.bfloat16":
        errors.append("metadata.dtype")
    if require_cuda and collection["acceptance"]["require_cuda_peak_memory_metadata"]:
        device = str(trace.metadata.get("device", ""))
        if device != "cuda" and not (
            device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
        ):
            errors.append("metadata.device")
        memory_names = (
            "gpu_memory_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        )
        for name in memory_names:
            value = trace.metadata.get(name)
            if type(value) is not int or value <= 0:
                errors.append(f"metadata.{name}")
        if not any(f"metadata.{name}" in errors for name in memory_names):
            allocated = trace.metadata["cuda_peak_allocated_bytes"]
            reserved = trace.metadata["cuda_peak_reserved_bytes"]
            total = trace.metadata["gpu_memory_bytes"]
            if not allocated <= reserved <= total:
                errors.append("metadata.cuda_memory_order")
    if trace.stop_reason == "max_new_tokens" and trace.eos_token_ids:
        if int(trace.generated_token_ids[-1]) in trace.eos_token_ids:
            errors.append("censored_final_token_is_eos")
    if errors:
        raise ValueError("Bayesian full-Train trace mismatch: " + "; ".join(errors))


def build_bayesian_full_train_summary(
    collection: dict[str, Any],
    *,
    collection_config_sha256: str,
    rows: list[dict[str, Any]],
    new_trace_count: int,
    resumed_trace_count: int,
) -> dict[str, Any]:
    """Build the machine-readable progress/final report from validated trace rows."""

    acceptance = collection["acceptance"]
    expected = int(acceptance["expected_trace_count"])
    valid = len(rows)
    missing = expected - valid
    if missing < 0:
        raise ValueError("validated trace rows exceed the frozen expected count")
    ranks = [int(row["job_rank"]) for row in rows]
    identities = [
        (
            str(row["prompt_id"]),
            float(row["temperature"]),
            int(row["seed"]),
        )
        for row in rows
    ]
    if any(rank < 0 or rank >= expected for rank in ranks):
        raise ValueError("validated trace row has an out-of-range job rank")
    if len(set(ranks)) != valid or len(set(identities)) != valid:
        raise ValueError("validated trace rows contain duplicate jobs")
    if new_trace_count < 0 or resumed_trace_count < 0:
        raise ValueError("new and resumed trace counts cannot be negative")
    if new_trace_count + resumed_trace_count != valid:
        raise ValueError("new and resumed trace counts must partition validated rows")
    stop_reasons = Counter(str(row["stop_reason"]) for row in rows)
    task_lengths = Counter(
        f"{row['task']}:{row['intended_length']}" for row in rows
    )
    temperatures = Counter(f"{float(row['temperature']):.3f}" for row in rows)
    seeds = Counter(str(int(row["seed"])) for row in rows)
    families = Counter(str(row["prompt_family_id"]) for row in rows)
    total_duration_ms = sum(float(row["duration_ms"]) for row in rows)
    average_duration_ms = (
        total_duration_ms / valid
        if valid
        else collection["budget"]["basis_total_duration_ms"]
        / collection["budget"]["basis_trace_count"]
    )
    censoring_rate = stop_reasons["max_new_tokens"] / valid if valid else 0.0
    warnings = []
    failures = []
    if (
        valid >= acceptance["minimum_valid_traces_for_censoring_warning"]
        and censoring_rate >= acceptance["warning_censoring_rate"]
    ):
        warnings.append("full-Train censoring rate reaches the frozen warning threshold")
    if (
        valid >= acceptance["minimum_valid_traces_for_censoring_abort"]
        and censoring_rate >= acceptance["abort_censoring_rate"]
    ):
        failures.append("full-Train censoring rate reaches the frozen abort threshold")
    if missing == 0:
        if len(families) != acceptance["expected_prompt_family_count"]:
            failures.append("complete full-Train collection has the wrong family count")
        expected_per_family = expected // acceptance["expected_prompt_family_count"]
        if set(families.values()) != {expected_per_family}:
            failures.append("complete full-Train collection has unbalanced families")
        expected_per_task_length = expected // acceptance["required_task_length_cells"]
        if len(task_lengths) != acceptance["required_task_length_cells"] or set(
            task_lengths.values()
        ) != {expected_per_task_length}:
            failures.append("complete full-Train collection lacks task-length cells")
        expected_per_temperature = expected // acceptance["required_temperature_count"]
        if len(temperatures) != acceptance["required_temperature_count"] or set(
            temperatures.values()
        ) != {expected_per_temperature}:
            failures.append("complete full-Train collection lacks temperatures")
        expected_per_seed = expected // acceptance["required_seed_count"]
        if len(seeds) != acceptance["required_seed_count"] or set(
            seeds.values()
        ) != {expected_per_seed}:
            failures.append("complete full-Train collection lacks seeds")
    status = "failed" if failures else ("pass" if missing == 0 else "incomplete")
    total_trace_bytes = sum(int(row["trace_bytes"]) for row in rows)
    summary = {
        "collection_id": collection["collection_id"],
        "collection_config_sha256": collection_config_sha256,
        "scientific_contract_sha256": collection["scientific_contract"]["sha256"],
        "source_prompt_manifest_sha256": collection["source_prompts"]["sha256"],
        "status": status,
        "expected_trace_count": expected,
        "valid_trace_count": valid,
        "missing_trace_count": missing,
        "completion_fraction": valid / expected,
        "valid_prompt_family_count": len(families),
        "valid_task_length_cell_count": len(task_lengths),
        "by_stop_reason": dict(sorted(stop_reasons.items())),
        "by_task_length": dict(sorted(task_lengths.items())),
        "by_temperature": dict(sorted(temperatures.items())),
        "by_seed": dict(sorted(seeds.items())),
        "censoring_rate": censoring_rate,
        "total_observed_tokens": sum(int(row["observed_tokens"]) for row in rows),
        "total_duration_ms": total_duration_ms,
        "total_trace_bytes": total_trace_bytes,
        "total_trace_gib": total_trace_bytes / 1024**3,
        "peak_cuda_allocated_bytes": max(
            (int(row["cuda_peak_allocated_bytes"]) for row in rows),
            default=0,
        ),
        "peak_cuda_reserved_bytes": max(
            (int(row["cuda_peak_reserved_bytes"]) for row in rows),
            default=0,
        ),
        "projected_remaining_duration_ms": average_duration_ms * missing,
        "new_trace_count_this_run": new_trace_count,
        "resumed_trace_count_this_run": resumed_trace_count,
        "warnings": warnings,
        "failures": failures,
        "full_train_collection_complete": status == "pass",
        "final_holdout_accessed": False,
    }
    return summary
