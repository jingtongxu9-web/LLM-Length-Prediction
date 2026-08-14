"""Stage-8A final-Train fitting and one-time holdout gate contracts."""

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llm_length_prediction.bayesian_stage5 import Stage5Catalog, load_stage5_catalog
from llm_length_prediction.checkpoints import atomic_torch_save, load_torch_checkpoint
from llm_length_prediction.data.stage5 import (
    stage5_bayesian_sequence,
    stage5_hybrid_samples,
    stage5_prior_summary_matrix,
)
from llm_length_prediction.experiment import file_sha256
from llm_length_prediction.models.bayesian_scorer import (
    BAYESIAN_METHOD_IDS,
    fit_bayesian_scorer,
    load_bayesian_checkpoint,
    make_bayesian_checkpoint,
    restore_bayesian_scorer,
    save_bayesian_checkpoint,
)
from llm_length_prediction.models.dynamic import (
    StandardizedMLPRemainingLength,
    fit_plp_mlp,
)
from llm_length_prediction.models.hybrid import (
    SummaryScaler,
    WeightedLogRidge,
    build_progressive_head,
    fit_progressive_head,
    fit_summary_scaler,
    hybrid_feature_matrix,
)
from llm_length_prediction.models.prior import (
    StandardizedRidgeLogNormalPrior,
    fit_grouped_oof_log1p_prior,
)
from llm_length_prediction.models.prompt_token_baseline import fit_prompt_token_ridge

STAGE8A_ID = "bayesian-sequential-v1-stage8a-final-freeze"
LOCK_ID = "bayesian-sequential-v1-stage8b-final-lock"
MODEL_FILES = (
    "alps_prior.json",
    "prompt_token_ridge_countdown.json",
    "dynamic_signal_mlp_v1.json",
    "plp_terminal_zero_v3.pt",
    "alps_plp_concat_v1.pt",
    "bayesian_entropy_scalar_v1.pt",
    "bayesian_entropy_hidden_delta_v1.pt",
)


@dataclass(frozen=True)
class FinalModels:
    """Every comparator frozen before the final holdout is authored or opened."""

    alps_prior: StandardizedRidgeLogNormalPrior
    prompt_token_ridge: WeightedLogRidge
    dynamic_signal_mlp: StandardizedMLPRemainingLength
    plp_head: Any
    concat_head: Any
    concat_scaler: SummaryScaler
    bayesian_scorers: Mapping[str, Any]
    device: str


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_stage8a_config(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != 1 or payload.get("stage8a_id") != STAGE8A_ID:
        raise ValueError("unsupported Stage-8A configuration")
    contract = payload["scientific_contract"]
    stage5 = payload["stage5"]
    stage7 = payload["stage7"]
    schema = payload["report_schema"]
    source_pins = payload["implementation"]["required_source_sha256"]
    if not source_pins:
        raise ValueError("Stage-8A implementation source pins are missing")
    changed_sources = [
        source for source, digest in source_pins.items() if file_sha256(source) != digest
    ]
    if changed_sources:
        raise ValueError(f"Stage-8A implementation source changed: {changed_sources}")
    pins = (
        (contract["path"], contract["sha256"], "scientific contract"),
        (stage5["config"], stage5["config_sha256"], "Stage-5 config"),
        (
            stage5["selection_summary"],
            stage5["selection_summary_sha256"],
            "Stage-5 selection",
        ),
        (stage7["summary"], stage7["summary_sha256"], "Stage-7 summary"),
        (
            payload["benchmark"]["serving_replay_config"],
            payload["benchmark"]["serving_replay_config_sha256"],
            "serving replay config",
        ),
        (schema["path"], schema["sha256"], "final report schema"),
    )
    changed = [label for source, digest, label in pins if file_sha256(source) != digest]
    if changed:
        raise ValueError(f"Stage-8A frozen sources changed: {changed}")
    selection = _read_json(stage5["selection_summary"])
    if (
        selection.get("selection", {}).get("selected_method") != stage5["selected_method"]
        or selection.get("data", {}).get("dataset_digest") != stage5["dataset_digest"]
        or selection.get("data", {}).get("final_holdout_accessed") is not False
        or selection.get("selection", {}).get("final_holdout_selects_nothing") is not True
    ):
        raise ValueError("Stage-5 selected method or dataset changed")
    feedback = _read_json(stage7["summary"])
    if (
        feedback.get("status") != "pass"
        or feedback.get("final_holdout_accessed") is not False
        or stage7.get("new_method_added") is not False
    ):
        raise ValueError("Stage-7 does not permit the frozen v1 final benchmark")
    training = payload["final_training"]
    if (
        training["data"] != "all_60_opened_train_families_primary_temperature_only"
        or training["temperature"] != 0.7
        or training["final_holdout_access"] != "forbidden"
        or training["reuse_stage5_hyperparameters_without_change"] is not True
    ):
        raise ValueError("Stage-8A final fitting boundary changed")
    benchmark = payload["benchmark"]
    if benchmark["primary_method"] != stage5["selected_method"]:
        raise ValueError("final primary method differs from Stage-5 selection")
    if benchmark["final_holdout_selects_nothing"] is not True:
        raise ValueError("final holdout cannot select a method")
    comparison = payload["comparison_list"]
    expected = [
        "prompt_token_ridge_countdown",
        "alps_countdown",
        "dynamic_signal_mlp_v1",
        "plp_terminal_zero_v3",
        "alps_plp_concat_v1",
        *BAYESIAN_METHOD_IDS,
    ]
    if comparison != expected:
        raise ValueError("Stage-8 comparison list changed")
    holdout = payload["final_holdout_plan"]
    expected_rollouts = (
        int(holdout["prompt_count"]) * len(holdout["temperatures"]) * len(holdout["seeds"])
    )
    if (
        holdout["status"] != "not_authored_not_opened"
        or expected_rollouts != holdout["expected_rollout_count"]
        or holdout["family_overlap_with_any_existing_manifest"] != "forbidden"
    ):
        raise ValueError("Stage-8 final holdout plan is not safely frozen")
    return payload


def load_stage8_catalog(
    config: dict[str, Any],
    *,
    dataset_root: str | Path,
    verify_trace_hashes: bool,
) -> Stage5Catalog:
    catalog = load_stage5_catalog(
        config["stage5"]["config"],
        dataset_root=dataset_root,
        verify_trace_hashes=verify_trace_hashes,
    )
    if catalog.dataset_digest != config["stage5"]["dataset_digest"]:
        raise ValueError("Stage-8A dataset digest changed")
    primary = [
        reference
        for reference in catalog.references
        if reference.temperature == config["final_training"]["temperature"]
    ]
    if (
        len(primary) != config["final_training"]["trace_count"]
        or len({row.prompt_family_id for row in primary})
        != config["final_training"]["family_count"]
    ):
        raise ValueError("Stage-8A primary-temperature training grid is incomplete")
    return catalog


def validate_training_environment(config: dict[str, Any]) -> dict[str, Any]:
    """Require the exact environment used by the frozen Stage-5 server experiment."""

    import importlib.metadata

    expected = config["training_environment"]
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Stage-8A final training requires PyTorch") from error
    actual = {
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "numpy_version": importlib.metadata.version("numpy"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "cuda_runtime": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    mismatches = [
        name
        for name in (
            "python_version",
            "torch_version",
            "numpy_version",
            "scikit_learn_version",
            "cuda_runtime",
            "gpu_name",
        )
        if actual[name] != expected[name]
    ]
    if expected["bf16_required"] and not actual["bf16_supported"]:
        mismatches.append("bf16_supported")
    if mismatches:
        raise RuntimeError(f"Stage-8A training environment differs: {mismatches}")
    return actual


def _progressive_common(stage5: dict[str, Any], device: str) -> dict[str, Any]:
    settings = stage5["baseline_training"]["progressive_heads"]
    return {
        "num_bins": int(settings["num_bins"]),
        "percentiles": tuple(float(x) for x in settings["target_range_percentiles"]),
        "lambda_ce": float(settings["lambda_ce"]),
        "dropout": float(settings["dropout"]),
        "epochs": int(settings["epochs"]),
        "batch_size": int(settings["batch_size"]),
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "seed": int(stage5["bayesian_training"]["seed"]),
        "device": device,
    }


def fit_final_models(
    config: dict[str, Any],
    catalog: Stage5Catalog,
    *,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Fit every frozen comparator on all opened primary-temperature Train traces."""

    environment = validate_training_environment(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = sorted(
        (
            reference
            for reference in catalog.references
            if reference.temperature == config["final_training"]["temperature"]
        ),
        key=lambda row: row.identity,
    )
    traces = [(reference, catalog.load_trace(reference)) for reference in primary]
    features = np.stack([trace.prior_feature for _, trace in traces])
    lengths = np.asarray([trace.observed_tokens for _, trace in traces], dtype=np.int64)
    groups = [trace.prompt_family_id for _, trace in traces]
    row_folds = np.asarray([catalog.family_folds[group] for group in groups], dtype=np.int32)
    stage5 = catalog.config
    prior, oof_mu, resolved_folds = fit_grouped_oof_log1p_prior(
        features,
        lengths,
        groups,
        folds=int(stage5["data_policy"]["outer_oof_folds"]),
        alpha=float(stage5["prior"]["ridge_alpha"]),
        seed=int(stage5["data_policy"]["fold_seed"]),
        fold_ids=row_folds,
    )
    trace_mu = {
        reference.identity: float(mu) for (reference, _), mu in zip(traces, oof_mu, strict=True)
    }
    _write_json(output_dir / "alps_prior.json", prior.to_dict())

    prompt_ridge = fit_prompt_token_ridge(
        [trace.prompt_tokens for _, trace in traces],
        [trace.observed_tokens for _, trace in traces],
        alpha=float(stage5["prior"]["ridge_alpha"]),
    )
    _write_json(output_dir / "prompt_token_ridge_countdown.json", prompt_ridge.to_dict())

    samples = [sample for _, trace in traces for sample in stage5_hybrid_samples(trace)]
    nonterminal = [sample for sample in samples if sample.remaining_tokens > 0]
    counts = Counter(sample.trace_key for sample in nonterminal)
    dynamic_config = stage5["baseline_training"]["dynamic_signal_mlp_v1"]
    dynamic, dynamic_report = fit_plp_mlp(
        [sample.dynamic_features for sample in nonterminal],
        [sample.remaining_tokens for sample in nonterminal],
        [1.0 / counts[sample.trace_key] for sample in nonterminal],
        hidden_sizes=tuple(int(x) for x in dynamic_config["hidden_sizes"]),
        dropout=float(dynamic_config["dropout"]),
        epochs=int(dynamic_config["epochs"]),
        batch_size=int(dynamic_config["batch_size"]),
        learning_rate=float(dynamic_config["learning_rate"]),
        weight_decay=float(dynamic_config["weight_decay"]),
        seed=int(stage5["bayesian_training"]["seed"]),
        device=device,
    )
    _write_json(output_dir / "dynamic_signal_mlp_v1.json", dynamic.to_dict())

    prior_summary = stage5_prior_summary_matrix(
        samples,
        trace_mu=trace_mu,
        variance=prior.residual_variance,
    )
    common = _progressive_common(stage5, device)
    progressive = stage5["baseline_training"]["progressive_heads"]
    plp_features = np.stack([sample.plp_features for sample in samples])
    plp_head, plp_report = fit_progressive_head(
        samples,
        plp_features,
        hidden_dim=int(progressive["plp_hidden_dim"]),
        terminal_zero=True,
        weighted_range=False,
        **common,
    )
    atomic_torch_save(
        {
            "schema_version": 1,
            "method_id": "plp_terminal_zero_v3",
            "state_dict": plp_head.state_dict(),
            "metadata": plp_report,
        },
        output_dir / "plp_terminal_zero_v3.pt",
    )

    sample_weights = np.asarray([sample.sequence_weight for sample in samples], dtype=np.float32)
    concat_scaler = fit_summary_scaler(prior_summary, sample_weights)
    concat_head, concat_report = fit_progressive_head(
        samples,
        hybrid_feature_matrix(samples, prior_summary, scaler=concat_scaler),
        hidden_dim=int(progressive["concat_hidden_dim"]),
        terminal_zero=True,
        weighted_range=True,
        **common,
    )
    atomic_torch_save(
        {
            "schema_version": 1,
            "method_id": "alps_plp_concat_v1",
            "state_dict": concat_head.state_dict(),
            "metadata": concat_report,
            "concat_scaler": concat_scaler.to_dict(),
        },
        output_dir / "alps_plp_concat_v1.pt",
    )

    sequences = [
        stage5_bayesian_sequence(
            trace,
            prior_mu=trace_mu[reference.identity],
            prior_log_variance=prior.residual_variance,
        )
        for reference, trace in traces
    ]
    training = stage5["bayesian_training"]
    bayesian_reports = {}
    for method_id in BAYESIAN_METHOD_IDS:
        scorer, report = fit_bayesian_scorer(
            sequences,
            method_id=method_id,
            hidden_projection_dim=int(training["hidden_projection_dim"]),
            hidden_dim=int(training["hidden_dim"]),
            dropout=float(training["dropout"]),
            epochs=int(training["epochs"]),
            sequence_batch_size=int(training["sequence_batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            terminal_bce_weight=float(training["terminal_bce_weight"]),
            stability_weight=float(training["posterior_total_variation_stability_weight"]),
            seed=int(training["seed"]),
            device=device,
        )
        checkpoint = make_bayesian_checkpoint(
            scorer,
            contract_sha256=config["scientific_contract"]["sha256"],
            training_report=report,
        )
        save_bayesian_checkpoint(checkpoint, output_dir / f"{method_id}.pt")
        bayesian_reports[method_id] = report

    report = {
        "schema_version": 1,
        "stage8a_id": config["stage8a_id"],
        "status": "pass",
        "config_sha256": config["_config_sha256"],
        "dataset_digest": catalog.dataset_digest,
        "training_trace_count": len(traces),
        "training_family_count": len(set(groups)),
        "training_temperature": config["final_training"]["temperature"],
        "training_environment": environment,
        "prior_crossfit_fold_counts": {
            str(fold): int(np.sum(resolved_folds == fold)) for fold in sorted(set(resolved_folds))
        },
        "prior_residual_variance": prior.residual_variance,
        "dynamic_training": dynamic_report,
        "plp_training": plp_report,
        "concat_training": concat_report,
        "bayesian_training": bayesian_reports,
        "selected_primary_method": config["benchmark"]["primary_method"],
        "hyperparameter_selection_performed": False,
        "final_holdout_accessed": False,
    }
    report_path = output_dir / config["outputs"]["training_report"]
    _write_json(report_path, report)
    registry = build_checkpoint_registry(config, output_dir=output_dir)
    _write_json(output_dir / config["outputs"]["checkpoint_registry"], registry)
    return registry


def build_checkpoint_registry(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    missing = [name for name in MODEL_FILES if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"final model files are missing: {missing}")
    report_name = config["outputs"]["training_report"]
    if not (output_dir / report_name).is_file():
        raise ValueError("final training report is missing")
    return {
        "schema_version": 1,
        "registry_id": "bayesian-sequential-v1-final-models",
        "status": "pass",
        "stage8a_config_sha256": config["_config_sha256"],
        "dataset_digest": config["stage5"]["dataset_digest"],
        "primary_method": config["benchmark"]["primary_method"],
        "comparison_list": config["comparison_list"],
        "files": {name: {"sha256": file_sha256(output_dir / name)} for name in MODEL_FILES},
        "training_report": {
            "path": report_name,
            "sha256": file_sha256(output_dir / report_name),
        },
        "final_holdout_accessed": False,
    }


def validate_checkpoint_registry(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    path = output_dir / config["outputs"]["checkpoint_registry"]
    if not path.is_file():
        raise ValueError("final checkpoint registry does not exist")
    registry = _read_json(path)
    if (
        registry.get("status") != "pass"
        or registry.get("stage8a_config_sha256") != config["_config_sha256"]
        or registry.get("dataset_digest") != config["stage5"]["dataset_digest"]
        or registry.get("primary_method") != config["benchmark"]["primary_method"]
        or registry.get("comparison_list") != config["comparison_list"]
        or registry.get("final_holdout_accessed") is not False
    ):
        raise ValueError("final checkpoint registry violates the Stage-8A freeze")
    for name in MODEL_FILES:
        expected = registry.get("files", {}).get(name, {}).get("sha256")
        if not isinstance(expected, str) or file_sha256(output_dir / name) != expected:
            raise ValueError(f"final checkpoint digest changed: {name}")
    report = registry["training_report"]
    if file_sha256(output_dir / report["path"]) != report["sha256"]:
        raise ValueError("final training report digest changed")
    return registry


def _resolved_device(device: str) -> str:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("final-model loading requires PyTorch") from error
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved == "auto":
        resolved = "cpu"
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("final-model loading requested CUDA, but CUDA is unavailable")
    return resolved


def _restore_progressive_head(
    payload: Mapping[str, Any],
    *,
    method_id: str,
    dropout: float,
    device: str,
) -> Any:
    if payload.get("method_id") != method_id or not isinstance(payload.get("metadata"), Mapping):
        raise ValueError(f"invalid final progressive checkpoint: {method_id}")
    metadata = payload["metadata"]
    head = build_progressive_head(
        int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        num_bins=int(metadata["num_bins"]),
        value_range=tuple(float(value) for value in metadata["target_range"]),
        terminal_zero=bool(metadata["terminal_zero_bin"]),
        dropout=dropout,
    )
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head.to(device)


def load_final_models(
    config: dict[str, Any],
    *,
    output_dir: Path,
    device: str = "cpu",
) -> FinalModels:
    """Verify the registry first, then restore all seven frozen comparators."""

    validate_checkpoint_registry(config, output_dir=output_dir)
    resolved = _resolved_device(device)
    alps_prior = StandardizedRidgeLogNormalPrior.from_dict(
        _read_json(output_dir / "alps_prior.json")
    )
    prompt_ridge = WeightedLogRidge.from_dict(
        _read_json(output_dir / "prompt_token_ridge_countdown.json")
    )
    if prompt_ridge.target != "log1p_output_tokens_from_prompt_tokens":
        raise ValueError("prompt-token Ridge target changed")
    dynamic = StandardizedMLPRemainingLength.from_dict(
        _read_json(output_dir / "dynamic_signal_mlp_v1.json")
    )
    progressive = config["stage5"]
    stage5_config = _read_json(progressive["config"])
    settings = stage5_config["baseline_training"]["progressive_heads"]
    plp_payload = load_torch_checkpoint(output_dir / "plp_terminal_zero_v3.pt")
    plp_head = _restore_progressive_head(
        plp_payload,
        method_id="plp_terminal_zero_v3",
        dropout=float(settings["dropout"]),
        device=resolved,
    )
    concat_payload = load_torch_checkpoint(output_dir / "alps_plp_concat_v1.pt")
    if not isinstance(concat_payload.get("concat_scaler"), Mapping):
        raise ValueError("concat checkpoint is missing its frozen ALPS scaler")
    concat_head = _restore_progressive_head(
        concat_payload,
        method_id="alps_plp_concat_v1",
        dropout=float(settings["dropout"]),
        device=resolved,
    )
    concat_scaler = SummaryScaler.from_dict(concat_payload["concat_scaler"])
    scorers = {}
    for method_id in BAYESIAN_METHOD_IDS:
        checkpoint = load_bayesian_checkpoint(output_dir / f"{method_id}.pt")
        if (
            checkpoint["method_id"] != method_id
            or checkpoint["contract_sha256"] != config["scientific_contract"]["sha256"]
        ):
            raise ValueError(f"Bayesian final checkpoint identity changed: {method_id}")
        scorers[method_id] = restore_bayesian_scorer(checkpoint, device=resolved).eval()
    if len(alps_prior.weights) != len(alps_prior.feature_mean):
        raise ValueError("ALPS prior dimensions are inconsistent")
    if prompt_ridge.weights.shape != (1,):
        raise ValueError("prompt-token Ridge must have exactly one input feature")
    return FinalModels(
        alps_prior=alps_prior,
        prompt_token_ridge=prompt_ridge,
        dynamic_signal_mlp=dynamic,
        plp_head=plp_head,
        concat_head=concat_head,
        concat_scaler=concat_scaler,
        bayesian_scorers=scorers,
        device=resolved,
    )


def final_holdout_gate_report(
    config_path: str | Path,
    *,
    model_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_stage8a_config(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    lock_path = Path(config["holdout_gate"]["benchmark_lock"])
    failures: list[str] = []
    registry = None
    root = Path(model_root or config["outputs"]["model_root"])
    try:
        registry = validate_checkpoint_registry(config, output_dir=root)
    except (KeyError, OSError, TypeError, ValueError) as error:
        failures.append(str(error))
    lock = _read_json(lock_path) if lock_path.is_file() else None
    if lock is None:
        failures.append("Stage-8B benchmark lock does not exist")
    elif (
        lock.get("lock_id") != LOCK_ID
        or lock.get("status") != config["holdout_gate"]["required_lock_status"]
        or lock.get("final_holdout_opened") is not False
        or lock.get("final_holdout_selects_nothing") is not True
        or lock.get("prompt_semantic_overlap_review_complete") is not True
    ):
        failures.append("Stage-8B benchmark lock is not ready")
    if lock is not None and registry is not None:
        registry_path = root / config["outputs"]["checkpoint_registry"]
        if lock.get("stage8a_config_sha256") != config["_config_sha256"]:
            failures.append("Stage-8B lock has the wrong Stage-8A config digest")
        if lock.get("checkpoint_registry_sha256") != file_sha256(registry_path):
            failures.append("Stage-8B lock has the wrong checkpoint registry digest")
        locked_files = lock.get("checkpoint_sha256", {})
        if any(locked_files.get(name) != registry["files"][name]["sha256"] for name in MODEL_FILES):
            failures.append("Stage-8B lock does not pin every final model")
        manifest_path = Path(str(lock.get("final_holdout_manifest", "")))
        if not manifest_path.is_file():
            failures.append("frozen final holdout manifest does not exist")
        elif lock.get("final_holdout_manifest_sha256") != file_sha256(manifest_path):
            failures.append("frozen final holdout manifest digest changed")
        commit = lock.get("git_commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            failures.append("Stage-8B lock does not pin a full Git commit")
        elif config["holdout_gate"]["required_git_clean"]:
            try:
                remote_main = subprocess.run(
                    ("git", "rev-parse", "origin/main"),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                ancestor = subprocess.run(
                    ("git", "merge-base", "--is-ancestor", commit, remote_main),
                    check=False,
                ).returncode
            except (OSError, subprocess.CalledProcessError) as error:
                failures.append(f"cannot verify frozen Git state: {error}")
            else:
                if ancestor != 0:
                    failures.append("frozen implementation commit is not on origin/main")
    return {
        "stage8a_id": config["stage8a_id"],
        "ready": not failures,
        "status": "ready" if not failures else "blocked",
        "failures": failures,
        "final_holdout_opened": False,
        "final_holdout_accessed": False,
    }
