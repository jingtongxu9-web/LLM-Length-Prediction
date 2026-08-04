from pathlib import Path

import numpy as np
import pytest

from llm_length_prediction.data.plp import (
    PLPHiddenStateTrace,
    read_plp_trace,
    write_plp_trace,
)
from llm_length_prediction.evaluation.plp import (
    hidden_state_plp_group_breakdown,
    hidden_state_plp_metrics,
    hidden_state_plp_progress_breakdown,
)
from llm_length_prediction.models.plp import (
    build_hidden_state_plp_samples,
    length_bin_centers,
    plp_head_parameter_count,
    soft_length_labels,
    target_range_from_training,
)
from llm_length_prediction.plp_experiment import (
    load_plp_base_experiment,
    load_plp_config,
    partition_censored_plp_traces,
    validate_plp_config,
    validate_plp_trace,
)
from scripts.collect_plp_dataset import validate_local_model_revision
from scripts.train_plp import atomic_torch_save


def _trace(stop_reason: str = "eos") -> PLPHiddenStateTrace:
    return PLPHiddenStateTrace(
        prompt_id="qa_001_short",
        task="qa",
        prompt_tokens=12,
        output_tokens=10,
        temperature=0.7,
        seed=42,
        stop_reason=stop_reason,
        prompt_feature=np.asarray([0.5, -0.5], dtype=np.float32),
        decode_hidden_states=np.asarray(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
        ),
        steps=np.asarray([1, 5, 10], dtype=np.int32),
        remaining_lengths=np.asarray([9, 5, 0], dtype=np.int32),
        token_ids=np.asarray([10, 11, 12], dtype=np.int32),
        generated_token_ids=np.asarray(
            [10, 20, 21, 22, 11, 23, 24, 25, 26, 12], dtype=np.int32
        ),
    )


def test_plp_trace_npz_round_trip(tmp_path: Path) -> None:
    path = write_plp_trace(tmp_path / "trace.npz", _trace())
    restored = read_plp_trace(path)
    assert restored.prompt_id == "qa_001_short"
    assert restored.prompt_feature.dtype == np.float32
    assert restored.decode_hidden_states.shape == (3, 2)
    assert restored.remaining_lengths.tolist() == [9, 5, 0]
    assert restored.generated_token_ids.tolist() == [10, 20, 21, 22, 11, 23, 24, 25, 26, 12]
    np.testing.assert_allclose(restored.prompt_feature, [0.5, -0.5], atol=1e-3)


def test_local_plp_collection_requires_exact_revision_marker(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    with pytest.raises(ValueError, match="missing .frozen_revision"):
        validate_local_model_revision(str(model_path), "expected")
    (model_path / ".frozen_revision").write_text("wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_local_model_revision(str(model_path), "expected")
    (model_path / ".frozen_revision").write_text("expected\n", encoding="utf-8")
    validate_local_model_revision(str(model_path), "expected")


def test_plp_samples_use_prompt_and_current_decode_state() -> None:
    samples = build_hidden_state_plp_samples(
        _trace(), prompt_family_id="qa_001", intended_length="short"
    )
    assert len(samples) == 3
    np.testing.assert_allclose(samples[0].features, [0.5, -0.5, 0.1, 0.2])
    assert [sample.remaining_tokens for sample in samples] == [9, 5, 0]
    assert sum(sample.sequence_weight for sample in samples) == pytest.approx(1.0)
    assert build_hidden_state_plp_samples(
        _trace("max_new_tokens"),
        prompt_family_id="qa_001",
        intended_length="short",
    ) == []


def test_soft_length_bins_follow_distance_distribution() -> None:
    target_range = (0.0, 20.0)
    centers = length_bin_centers(target_range, 4)
    labels = soft_length_labels([1.0, 19.0], target_range, 4)
    np.testing.assert_allclose(centers, [2.5, 7.5, 12.5, 17.5])
    np.testing.assert_allclose(labels.sum(axis=1), [1.0, 1.0], atol=1e-6)
    assert labels[0].argmax() == 0
    assert labels[1].argmax() == 3
    assert target_range_from_training([0, 5, 10, 100]) == pytest.approx((0.15, 97.3))
    assert plp_head_parameter_count(7168, 20) == 25_772_564


def test_plp_metrics_are_sequence_balanced_and_grouped_by_progress() -> None:
    samples = build_hidden_state_plp_samples(
        _trace(), prompt_family_id="qa_001", intended_length="short"
    )
    predictions = [8.0, 5.0, 1.0]
    metrics = hidden_state_plp_metrics(samples, predictions)
    assert metrics["count"] == 3
    assert metrics["trace_count"] == 1
    assert metrics["sequence_balanced_mae_tokens"] == pytest.approx(2.0 / 3.0)
    assert len(hidden_state_plp_progress_breakdown(samples, predictions)) == 3
    assert hidden_state_plp_group_breakdown(
        samples, predictions, group_by=("task", "intended_length")
    )[0]["trace_count"] == 1


def _contract_trace(*, steps: list[int] | None = None, dtype: str = "torch.bfloat16"):
    config = load_plp_config("configs/experiments/plp_v2_manifest.json")
    experiment, records = load_plp_base_experiment(config)
    record = records[0]
    actual_steps = np.asarray(steps or [1, 5, 10], dtype=np.int32)
    hidden_size = config["representation"]["hidden_size"]
    generated_token_ids = np.full(10, 99, dtype=np.int32)
    generated_token_ids[actual_steps - 1] = np.arange(len(actual_steps), dtype=np.int32)
    trace = PLPHiddenStateTrace(
        prompt_id=record["prompt_id"],
        task=record["task_type"],
        prompt_tokens=12,
        output_tokens=10,
        temperature=experiment["generation"]["temperature"],
        seed=42,
        stop_reason="eos",
        prompt_feature=np.zeros(hidden_size, dtype=np.float32),
        decode_hidden_states=np.zeros((len(actual_steps), hidden_size), dtype=np.float32),
        steps=actual_steps,
        remaining_lengths=10 - actual_steps,
        token_ids=np.arange(len(actual_steps), dtype=np.int32),
        generated_token_ids=generated_token_ids,
        model_revision=experiment["model"]["revision"],
        tokenizer_revision=experiment["model"]["tokenizer_revision"],
        metadata={
            "method_id": config["method_id"],
            "base_experiment_id": experiment["experiment_id"],
            "prompt_family_id": record["prompt_family_id"],
            "intended_length": record["intended_length"],
            "split": record["split"],
            "prompt_manifest_sha256": experiment["inputs"]["prompt_manifest_sha256"],
            "top_p": experiment["generation"]["top_p"],
            "max_new_tokens": experiment["generation"]["max_new_tokens"],
            "trace_stride": config["trace"]["stride"],
            "trace_schema_version": config["trace"]["schema_version"],
            "chat_template": experiment["generation"]["chat_template"],
            "hidden_layer": config["representation"]["hidden_layer"],
            "hidden_size": hidden_size,
            "prompt_pooling": config["representation"]["prompt_pooling"],
            "prompt_pooling_temperature": config["representation"][
                "prompt_pooling_temperature"
            ],
            "dynamic_aggregation": config["representation"]["dynamic_aggregation"],
            "storage_dtype": config["representation"]["storage_dtype"],
            "output_length_includes_eos": True,
            "dtype": dtype,
        },
    )
    return config, experiment, record, trace


def test_plp_contract_rejects_missing_stride_point() -> None:
    config, experiment, record, trace = _contract_trace(steps=[1, 10])
    with pytest.raises(ValueError, match="first/stride/final schedule"):
        validate_plp_trace(
            trace, record=record, seed=42, config=config, experiment=experiment
        )


def test_plp_contract_rejects_wrong_inference_dtype() -> None:
    config, experiment, record, trace = _contract_trace(dtype="torch.float32")
    with pytest.raises(ValueError, match="metadata.dtype"):
        validate_plp_trace(
            trace, record=record, seed=42, config=config, experiment=experiment
        )


def test_plp_config_rejects_unrecorded_head_change() -> None:
    config = load_plp_config("configs/experiments/plp_v2_manifest.json")
    experiment, _ = load_plp_base_experiment(config)
    config["prediction_head"]["dropout"] = 0.2
    with pytest.raises(ValueError, match="prediction_head.dropout"):
        validate_plp_config(config, experiment)


def test_plp_config_is_explicitly_non_exact() -> None:
    config = load_plp_config("configs/experiments/plp_v2_manifest.json")
    assert config["method_role"] == "paper_aligned_plp_only_nonexact"
    assert config["provenance"]["exact_replication"] is False


def test_partition_censored_plp_traces_reports_effective_count() -> None:
    loaded = [
        ({}, 42, Path("eos.npz"), _trace("eos")),
        ({}, 43, Path("censored.npz"), _trace("max_new_tokens")),
    ]
    effective, excluded = partition_censored_plp_traces(
        loaded, exclude_censored=True
    )
    assert [item[1] for item in effective] == [42]
    assert excluded == 1


def test_atomic_checkpoint_save_preserves_existing_file_on_failure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "plp_head.pt"
    checkpoint.write_bytes(b"valid-old-checkpoint")

    class FailingTorch:
        @staticmethod
        def save(payload: object, path: Path) -> None:
            path.write_bytes(b"partial")
            raise OSError("simulated interruption")

    with pytest.raises(OSError, match="simulated interruption"):
        atomic_torch_save({}, checkpoint, FailingTorch)
    assert checkpoint.read_bytes() == b"valid-old-checkpoint"
    assert list(tmp_path.glob("*.tmp")) == []
