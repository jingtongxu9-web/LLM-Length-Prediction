from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from llm_length_prediction.data.hybrid import (
    HybridV3Trace,
    read_hybrid_trace,
    write_hybrid_trace,
)
from llm_length_prediction.evaluation.hybrid import (
    family_macro_metrics,
    paired_family_mae_difference,
    task_stratified_family_folds,
)
from llm_length_prediction.hybrid_experiment import enforce_censoring_policy
from llm_length_prediction.models.hybrid import (
    HybridSample,
    WeightedLogRidge,
    bin_centers,
    soft_labels,
    target_range,
)
from llm_length_prediction.models.hybrid_suite import (
    METHOD_IDS,
    progressive_method_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample(family: str, task: str, remaining: int = 5) -> HybridSample:
    return HybridSample(
        prompt_id=f"{family}-prompt",
        prompt_family_id=family,
        task=task,
        intended_length="short",
        seed=42,
        step=1,
        output_tokens=remaining + 1,
        remaining_tokens=remaining,
        prior_feature=np.asarray([1.0, 2.0], dtype=np.float32),
        prompt_feature=np.asarray([3.0, 4.0], dtype=np.float32),
        decode_feature=np.asarray([5.0, 6.0], dtype=np.float32),
        dynamic_features=(1.0, 2.0, 2.0, 0.0, 0.1),
        sequence_weight=1.0,
    )


def test_frozen_manifest_has_new_family_holdout() -> None:
    path = ROOT / "data/prompts/alps_plp_hybrid_v3_prompts.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    train = {record["prompt_family_id"] for record in records if record["split"] == "train"}
    test = {record["prompt_family_id"] for record in records if record["split"] == "test"}
    assert len(records) == 216
    assert len(train) == 60
    assert len(test) == 12
    assert train.isdisjoint(test)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "1b4c42acfbefe274d14777d1e42426488965d199a8f1289a836caa65a73c2038"
    )


def test_hybrid_trace_round_trip_is_pickle_free(tmp_path: Path) -> None:
    trace = HybridV3Trace(
        prompt_id="p1",
        task="qa",
        prompt_tokens=8,
        output_tokens=5,
        temperature=0.7,
        seed=42,
        stop_reason="eos",
        prior_feature=np.asarray([1.0, 2.0], dtype=np.float32),
        prompt_feature=np.asarray([3.0, 4.0], dtype=np.float32),
        decode_hidden_states=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        steps=np.asarray([1, 5], dtype=np.int32),
        remaining_lengths=np.asarray([4, 0], dtype=np.int32),
        token_ids=np.asarray([10, 14], dtype=np.int32),
        generated_token_ids=np.asarray([10, 11, 12, 13, 14], dtype=np.int32),
        entropies=np.asarray([2.0, 1.0], dtype=np.float32),
        entropy_means=np.asarray([2.0, 1.5], dtype=np.float32),
        entropy_slopes=np.asarray([0.0, -0.2], dtype=np.float32),
        eos_probabilities=np.asarray([0.01, 0.9], dtype=np.float32),
    )
    path = write_hybrid_trace(tmp_path / "trace.npz", trace)
    loaded = read_hybrid_trace(path)
    assert loaded.output_tokens == 5
    assert loaded.remaining_lengths.tolist() == [4, 0]


def test_family_folds_never_split_a_family_and_cover_every_fold() -> None:
    samples = [
        _sample(f"{task}-{index}", task)
        for task in ("qa", "summarization", "code")
        for index in range(8)
    ]
    folds = task_stratified_family_folds(samples, folds=4, seed=7)
    assert set(folds.values()) == {0, 1, 2, 3}
    for task in ("qa", "summarization", "code"):
        assert {folds[f"{task}-{index}"] for index in range(8)} == {0, 1, 2, 3}


def test_shifted_lognormal_prior_round_trip_and_mean() -> None:
    model = WeightedLogRidge(
        feature_mean=np.asarray([0.0, 0.0]),
        feature_scale=np.asarray([1.0, 2.0]),
        weights=np.asarray([0.5, -0.25]),
        bias=2.0,
        residual_variance=0.4,
        target="log1p_output_tokens",
    )
    restored = WeightedLogRidge.from_dict(model.to_dict())
    features = np.asarray([[1.0, 2.0]])
    expected = np.expm1(restored.predict_mu(features)[0] + 0.2)
    assert np.isclose(restored.predict_mean(features)[0], expected)
    assert restored.target == "log1p_output_tokens"


def test_terminal_bin_is_exactly_zero() -> None:
    centers = bin_centers((1.0, 101.0), 20, terminal_zero=True)
    labels = soft_labels(np.asarray([0.0, 10.0]), (1.0, 101.0), 20, terminal_zero=True)
    assert centers[0] == 0.0
    assert labels[0, 0] == 1.0
    assert labels[0, 1:].sum() == 0.0
    assert labels[1, 0] == 0.0
    assert np.isclose(labels[1, 1:].sum(), 1.0)


def test_rollout_balanced_target_range_stops_long_rollouts_dominating() -> None:
    targets = np.asarray([10.0, 0.0, *range(90, -1, -10)])
    weights = np.asarray([0.5, 0.5, *([0.1] * 10)])
    unweighted = target_range(
        targets,
        weights,
        percentiles=(50.0, 90.0),
        positive_only=False,
        weighted=False,
    )
    rollout_balanced = target_range(
        targets,
        weights,
        percentiles=(50.0, 90.0),
        positive_only=False,
        weighted=True,
    )
    assert rollout_balanced[0] < unweighted[0]


def test_family_macro_and_paired_interval_use_family_unit() -> None:
    samples = [_sample("a", "qa", 5), _sample("b", "qa", 15)]
    first = [4.0, 14.0]
    second = [10.0, 20.0]
    metrics = family_macro_metrics(samples, first)
    difference = paired_family_mae_difference(
        samples, first, second, replicates=100, confidence=0.95, seed=2
    )
    assert metrics["family_count"] == 2
    assert difference["estimate"] == -4.0


def test_censoring_warning_and_abort_policy() -> None:
    warning = enforce_censoring_policy(
        loaded_count=100, censored_count=5, warning_rate=0.05, abort_rate=0.1
    )
    assert warning["status"] == "warning"
    try:
        enforce_censoring_policy(
            loaded_count=100, censored_count=10, warning_rate=0.05, abort_rate=0.1
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("abort threshold must stop the experiment")


def test_v2_contract_files_are_not_replaced_by_v3() -> None:
    v2 = json.loads((ROOT / "configs/experiments/plp_v2_manifest.json").read_text(encoding="utf-8"))
    v3 = json.loads(
        (ROOT / "configs/experiments/alps_plp_hybrid_v3.json").read_text(encoding="utf-8")
    )
    assert v2["trace"]["schema_version"] == 2
    assert v3["trace"]["schema_name"] == "hybrid-v3-unified-trace"
    assert v3["preserves_frozen_v2"] is True


def test_plp_v3_ablation_contract_changes_one_factor_at_a_time() -> None:
    protocol = json.loads(
        (ROOT / "configs/experiments/alps_plp_hybrid_v3_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(
        (ROOT / "configs/experiments/alps_plp_hybrid_v3.json").read_text(encoding="utf-8")
    )
    methods = protocol["methods"]
    assert tuple(methods) == METHOD_IDS
    assert "plp_small_terminal_v3" not in methods
    assert progressive_method_settings(protocol, "plp_v2_frozen") == (3584, False, False)
    assert progressive_method_settings(protocol, "plp_terminal_zero_v3") == (
        3584,
        True,
        False,
    )
    assert progressive_method_settings(protocol, "plp_small_head_v3") == (
        512,
        False,
        False,
    )
    assert progressive_method_settings(protocol, "plp_weighted_range_v3") == (
        3584,
        False,
        True,
    )
    hybrid = protocol["methods"]["alps_plp_hybrid_v3"]
    assert hybrid["terminal_zero_bin"] == config["progressive_head"]["terminal_zero_bin"]
    assert (
        hybrid["target_range_weighting"]
        == config["progressive_head"]["target_range_weighting"]
    )
    familywise = protocol["evaluation"]["primary_familywise_rule"]
    assert familywise["comparisons"] == len(METHOD_IDS) - 1
    assert np.isclose(
        familywise["per_comparison_confidence_level"],
        1.0 - familywise["alpha"] / familywise["comparisons"],
    )
