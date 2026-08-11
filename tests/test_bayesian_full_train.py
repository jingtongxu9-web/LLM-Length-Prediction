from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from llm_length_prediction.bayesian_full_train import (
    BayesianFullTrainJob,
    bayesian_full_train_jobs,
    build_bayesian_full_train_summary,
    load_bayesian_full_train,
    validate_bayesian_full_train_trace,
)
from llm_length_prediction.data.bayesian_trace import (
    BayesianTraceV1,
    bayesian_trace_path,
    write_bayesian_trace,
)
from llm_length_prediction.experiment import file_sha256
from scripts.collect_bayesian_full_train import _scan_existing

CONFIG_PATH = Path("configs/experiments/bayesian_sequential_full_train_v1.json")


def _loaded() -> tuple[dict[str, object], list[dict[str, object]]]:
    collection, _, records = load_bayesian_full_train(CONFIG_PATH)
    return collection, records


def test_full_train_contract_expands_to_balanced_deterministic_jobs() -> None:
    collection, records = _loaded()
    jobs = bayesian_full_train_jobs(collection, records)
    repeated = bayesian_full_train_jobs(collection, records)

    assert len(records) == 180
    assert len({record["prompt_family_id"] for record in records}) == 60
    assert {record["split"] for record in records} == {"train"}
    assert len(jobs) == 1620
    assert [job.rank for job in jobs] == list(range(1620))
    identities = [
        (job.record["prompt_id"], job.temperature, job.seed) for job in jobs
    ]
    assert len(set(identities)) == 1620
    assert identities == [
        (job.record["prompt_id"], job.temperature, job.seed) for job in repeated
    ]

    assert Counter(job.temperature for job in jobs) == {0.3: 540, 0.7: 540, 1.0: 540}
    assert Counter(job.seed for job in jobs) == {42: 540, 43: 540, 44: 540}
    assert set(
        Counter(
            (job.record["task_type"], job.record["intended_length"])
            for job in jobs
        ).values()
    ) == {180}
    assert set(Counter(job.record["prompt_family_id"] for job in jobs).values()) == {
        27
    }


def test_full_train_contract_forbids_final_holdout_access(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["source_prompts"]["allowed_split"] = "test"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only the Train split"):
        load_bayesian_full_train(path)


def test_full_train_contract_rejects_changed_implementation_source(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    first_path = next(iter(payload["implementation"]["required_source_sha256"]))
    payload["implementation"]["required_source_sha256"][first_path] = "0" * 64
    path = tmp_path / "invalid_source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation source changed"):
        load_bayesian_full_train(path)


def _trace(
    collection: dict[str, object],
    job: BayesianFullTrainJob,
    *,
    config_sha256: str,
) -> BayesianTraceV1:
    model = collection["model"]
    generation = collection["generation"]
    trace_contract = collection["trace"]
    representation = collection["prompt_representation"]
    record = job.record
    hidden_size = int(model["hidden_size"])
    metadata = {
        "collection_job_rank": job.rank,
        "collection_id": collection["collection_id"],
        "collection_stage": collection["stage"],
        "collection_config_sha256": config_sha256,
        "scientific_contract_sha256": collection["scientific_contract"]["sha256"],
        "source_prompt_manifest_sha256": collection["source_prompts"]["sha256"],
        "stage3_pilot_summary_sha256": collection["stage3_pilot_gate"]["sha256"],
        "trace_schema": trace_contract["schema_name"],
        "trace_schema_version": trace_contract["schema_version"],
        "trace_stride": trace_contract["stride"],
        "prior_feature_layer": model["prior_layer_zero_based"],
        "prior_layer_indexing": "zero_based_transformer_block",
        "decode_hidden_layer": model["decode_layer"],
        "hidden_size": hidden_size,
        "prompt_pooling": representation["pooling"],
        "prompt_pooling_temperature": representation["pooling_temperature"],
        "probability_source": trace_contract["probability_source"],
        "evidence_unit": trace_contract["evidence_unit"],
        "storage_dtype": trace_contract["storage_dtype"],
        "chat_template": generation["chat_template"],
        "output_length_includes_eos": True,
        "final_holdout_accessed": False,
        "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "gpu_memory_bytes": 1000,
        "cuda_peak_allocated_bytes": 100,
        "cuda_peak_reserved_bytes": 200,
    }
    return BayesianTraceV1(
        prompt_id=record["prompt_id"],
        prompt_family_id=record["prompt_family_id"],
        task=record["task_type"],
        intended_length=record["intended_length"],
        split="train",
        prompt_tokens=8,
        observed_tokens=2,
        max_new_tokens=generation["max_new_tokens"],
        temperature=job.temperature,
        top_p=generation["top_p"],
        seed=job.seed,
        stop_reason="eos",
        eos_token_ids=(3,),
        prior_feature=np.zeros(hidden_size, dtype=np.float32),
        prompt_feature=np.zeros(hidden_size, dtype=np.float32),
        initial_decode_hidden_state=np.zeros(hidden_size, dtype=np.float32),
        decode_hidden_states=np.zeros((2, hidden_size), dtype=np.float32),
        saved_steps=np.asarray([1, 2], dtype=np.int32),
        generated_token_ids=np.asarray([2, 3], dtype=np.int32),
        token_entropies=np.asarray([1.0, 0.1], dtype=np.float32),
        token_eos_probabilities=np.asarray([0.1, 0.9], dtype=np.float32),
        model_name=model["id"],
        model_revision=model["revision"],
        tokenizer_revision=model["tokenizer_revision"],
        duration_ms=10.0,
        metadata=metadata,
    )


def test_full_train_trace_validation_pins_job_cuda_and_provenance() -> None:
    collection, records = _loaded()
    job = bayesian_full_train_jobs(collection, records)[0]
    config_sha256 = file_sha256(CONFIG_PATH)
    trace = _trace(collection, job, config_sha256=config_sha256)
    validate_bayesian_full_train_trace(
        trace,
        job=job,
        collection=collection,
        collection_config_sha256=config_sha256,
    )

    trace.metadata["final_holdout_accessed"] = True
    with pytest.raises(ValueError, match="final_holdout_accessed"):
        validate_bayesian_full_train_trace(
            trace,
            job=job,
            collection=collection,
            collection_config_sha256=config_sha256,
        )


def test_scan_existing_rejects_invalid_trace_without_overwrite(tmp_path: Path) -> None:
    collection, records = _loaded()
    jobs = bayesian_full_train_jobs(collection, records)
    job = jobs[0]
    config_sha256 = file_sha256(CONFIG_PATH)
    trace = _trace(collection, job, config_sha256="0" * 64)
    path = bayesian_trace_path(
        tmp_path,
        split="train",
        prompt_id=job.record["prompt_id"],
        temperature=job.temperature,
        seed=job.seed,
    )
    write_bayesian_trace(path, trace)
    digest_before = file_sha256(path)

    rows, pending, invalid = _scan_existing(
        trace_root=tmp_path,
        jobs=jobs,
        collection=collection,
        collection_config_sha256=config_sha256,
    )
    assert rows == []
    assert len(pending) == 1619
    assert len(invalid) == 1
    assert "collection_config_sha256" in invalid[0]
    assert file_sha256(path) == digest_before


def test_scan_existing_rejects_unexpected_npz_path(tmp_path: Path) -> None:
    collection, records = _loaded()
    jobs = bayesian_full_train_jobs(collection, records)
    unexpected = tmp_path / "unknown" / "trace.npz"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_bytes(b"not a trace")

    rows, pending, invalid = _scan_existing(
        trace_root=tmp_path,
        jobs=jobs,
        collection=collection,
        collection_config_sha256=file_sha256(CONFIG_PATH),
    )
    assert rows == []
    assert len(pending) == 1620
    assert invalid == [f"{unexpected}: unexpected full-Train trace path"]


def _row(job: BayesianFullTrainJob, *, stop_reason: str = "eos") -> dict[str, object]:
    return {
        "job_rank": job.rank,
        "prompt_id": job.record["prompt_id"],
        "prompt_family_id": job.record["prompt_family_id"],
        "task": job.record["task_type"],
        "intended_length": job.record["intended_length"],
        "temperature": job.temperature,
        "seed": job.seed,
        "observed_tokens": 10,
        "stop_reason": stop_reason,
        "duration_ms": 20.0,
        "trace_bytes": 1000,
        "cuda_peak_allocated_bytes": 100,
        "cuda_peak_reserved_bytes": 200,
    }


def test_full_report_requires_complete_grid_and_matches_frozen_schema() -> None:
    collection, records = _loaded()
    jobs = bayesian_full_train_jobs(collection, records)
    config_sha256 = file_sha256(CONFIG_PATH)
    summary = build_bayesian_full_train_summary(
        collection,
        collection_config_sha256=config_sha256,
        rows=[_row(job) for job in jobs],
        new_trace_count=100,
        resumed_trace_count=1520,
    )
    schema_path = Path(collection["acceptance_report_schema"]["path"])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert summary["status"] == "pass"
    assert summary["full_train_collection_complete"] is True
    assert summary["valid_trace_count"] == 1620
    assert summary["missing_trace_count"] == 0
    assert summary["by_temperature"] == {"0.300": 540, "0.700": 540, "1.000": 540}
    assert summary["by_seed"] == {"42": 540, "43": 540, "44": 540}
    assert set(summary) == set(schema["required"])


def test_full_report_aborts_on_frozen_censoring_threshold() -> None:
    collection, records = _loaded()
    jobs = bayesian_full_train_jobs(collection, records)
    summary = build_bayesian_full_train_summary(
        collection,
        collection_config_sha256=file_sha256(CONFIG_PATH),
        rows=[_row(job, stop_reason="max_new_tokens") for job in jobs[:90]],
        new_trace_count=90,
        resumed_trace_count=0,
    )
    assert summary["status"] == "failed"
    assert summary["censoring_rate"] == 1.0
    assert summary["full_train_collection_complete"] is False
    assert summary["warnings"]
    assert summary["failures"]
