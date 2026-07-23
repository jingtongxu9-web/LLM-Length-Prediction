import json
from pathlib import Path

import pytest

from llm_length_prediction.data.io import write_trace_jsonl
from llm_length_prediction.data.schema import GenerationTrace, TracePoint
from llm_length_prediction.experiment import (
    ExperimentContractError,
    load_experiment,
    load_frozen_prompts,
    rollout_jobs,
    trace_path,
    validate_frozen_trace,
)
from scripts.build_prompt_manifest import build_records, write_manifest
from scripts.evaluate_prior import main as evaluate_prior_main
from scripts.preflight_server import _validate_model_snapshot
from scripts.train_prior import main as train_prior_main

EXPERIMENT_PATH = Path("configs/experiments/alps_v1_manifest.json")


def _valid_trace(experiment: dict, record: dict) -> GenerationTrace:
    generation = experiment["generation"]
    model = experiment["model"]
    return GenerationTrace(
        prompt_id=record["prompt_id"],
        task=record["task_type"],
        prompt_tokens=20,
        output_tokens=2,
        temperature=generation["temperature"],
        seed=42,
        stop_reason="eos",
        points=[TracePoint(1, 1.0, 0.1, 1), TracePoint(2, 0.5, 0.8, 0)],
        model_revision=model["revision"],
        tokenizer_revision=model["tokenizer_revision"],
        prefill_hidden_states={model["feature_layer"]: [0.1, -0.2]},
        metadata={
            "experiment_id": experiment["experiment_id"],
            "prompt_family_id": record["prompt_family_id"],
            "intended_length": record["intended_length"],
            "split": record["split"],
            "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
            "top_p": generation["top_p"],
            "max_new_tokens": generation["max_new_tokens"],
            "trace_stride": generation["trace_stride"],
            "entropy_window": generation["entropy_window"],
            "chat_template": generation["chat_template"],
            "layer_indexing": model["layer_indexing"],
            "output_length_includes_eos": generation["output_length_includes_eos"],
            "dtype": "torch.bfloat16",
        },
    )


def test_frozen_trace_rejects_stale_generation_settings() -> None:
    experiment = load_experiment(EXPERIMENT_PATH)
    record = next(
        record for record in load_frozen_prompts(experiment) if record["split"] == "train"
    )
    trace = _valid_trace(experiment, record)
    validate_frozen_trace(trace, record=record, seed=42, experiment=experiment)

    trace.metadata["top_p"] = 0.8
    with pytest.raises(ExperimentContractError, match="top_p"):
        validate_frozen_trace(trace, record=record, seed=42, experiment=experiment)


def test_prompt_builder_writes_platform_independent_lf(tmp_path: Path) -> None:
    output = write_manifest(tmp_path / "prompts.jsonl", build_records())
    content = output.read_bytes()
    assert b"\r\n" not in content
    assert content.count(b"\n") == 180


def test_model_snapshot_validation_detects_missing_weight_shard(tmp_path: Path) -> None:
    experiment = load_experiment(EXPERIMENT_PATH)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "num_hidden_layers": 28}), encoding="utf-8"
    )
    (tmp_path / "generation_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "template"}), encoding="utf-8"
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / ".frozen_revision").write_text(experiment["model"]["revision"], encoding="utf-8")
    failures = _validate_model_snapshot(tmp_path, experiment)
    assert any("weight shards" in failure for failure in failures)

    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"test")
    assert _validate_model_snapshot(tmp_path, experiment) == []


def test_synthetic_train_and_test_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = load_experiment(EXPERIMENT_PATH)
    records = load_frozen_prompts(experiment)
    trace_root = tmp_path / "traces"
    run_root = tmp_path / "runs"
    experiment["outputs"]["trace_root"] = str(trace_root)
    experiment["outputs"]["run_root"] = str(run_root)
    experiment_path = tmp_path / "experiment.json"
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")

    for index, (record, seed) in enumerate(rollout_jobs(records)):
        trace = _valid_trace(experiment, record)
        trace.seed = seed
        trace.output_tokens = 2 + index % 17
        trace.points = [TracePoint(trace.output_tokens, 0.5, 0.2, 0)]
        trace.prefill_hidden_states[experiment["model"]["feature_layer"]] = [
            float(index % 11),
            float(seed),
        ]
        write_trace_jsonl(trace_path(trace_root, record, seed), trace)

    output_dir = run_root / "stage1"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_prior.py",
            "--experiment",
            str(experiment_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    train_prior_main()
    prior_payload = json.loads((output_dir / "prior.json").read_text(encoding="utf-8"))
    assert prior_payload["training_count"] == 432
    assert prior_payload["ridge_alpha"] == 1.0

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_prior.py",
            "--experiment",
            str(experiment_path),
            "--prior",
            str(output_dir / "prior.json"),
            "--split",
            "train",
        ],
    )
    evaluate_prior_main()
    train_metrics = json.loads((output_dir / "train_evaluation.json").read_text(encoding="utf-8"))
    assert train_metrics["count"] == 432

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_prior.py",
            "--experiment",
            str(experiment_path),
            "--prior",
            str(output_dir / "prior.json"),
            "--split",
            "test",
            "--confirm-final-test",
        ],
    )
    evaluate_prior_main()
    test_metrics = json.loads((output_dir / "test_evaluation.json").read_text(encoding="utf-8"))
    assert test_metrics["count"] == 108
