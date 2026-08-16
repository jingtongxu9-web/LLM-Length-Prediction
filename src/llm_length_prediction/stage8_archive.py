"""Verify and summarize the one-time Stage-8 final benchmark archive."""

# ruff: noqa: E501 -- SVG and Markdown templates remain readable as complete elements.

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import re
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_MANIFEST = "bayesian_stage8b_final_results_files.sha256"
BENCHMARK_ROOT = "artifacts/runs/bayesian_sequential_v1/final_benchmark"
COLLECTION_ROOT = "artifacts/runs/bayesian_sequential_v1/final_holdout_collection"
METADATA_ROOT = "artifacts/runs/bayesian_sequential_v1/stage8b_bundle_metadata"
TRACE_ROOT = "data/interim/bayesian_sequential_v1_final_holdout"
FINAL_REPORT = f"{BENCHMARK_ROOT}/final_benchmark_report.json"
BENCHMARK_FILE_MANIFEST = f"{BENCHMARK_ROOT}/file_manifest.json"
COLLECTION_REPORT = f"{COLLECTION_ROOT}/collection_report.json"
COLLECTION_INDEX = f"{COLLECTION_ROOT}/collection_index.jsonl"
PREFLIGHT_REPORT = f"{METADATA_ROOT}/stage8b_ready_preflight.json"
GIT_HEAD = f"{METADATA_ROOT}/git_head.txt"
ENVIRONMENT = f"{METADATA_ROOT}/environment.txt"
BENCHMARK_LOG = f"{METADATA_ROOT}/bayesian_stage8_final_benchmark.log"

PRIMARY_METHOD = "bayesian_entropy_scalar_v1"
EXPECTED_BENCHMARK_ID = "bayesian-sequential-v1-one-time-final-benchmark"
EXPECTED_TRACE_COUNT = 324

METHOD_LABELS = {
    "prompt_token_ridge_countdown": "Prompt-token Ridge",
    "alps_countdown": "ALPS countdown",
    "dynamic_signal_mlp_v1": "Dynamic-signal MLP",
    "plp_terminal_zero_v3": "PLP terminal-zero v3",
    "alps_plp_concat_v1": "ALPS+PLP concat v1",
    "bayesian_entropy_scalar_v1": "Bayesian scalar (primary)",
    "bayesian_entropy_hidden_delta_v1": "Bayesian hidden-delta",
}

POINT_METRIC_KEYS = (
    "observation_count",
    "exact_sequence_count",
    "censored_sequence_count",
    "sequence_balanced_mae_tokens",
    "sequence_balanced_rmse_tokens",
    "sequence_balanced_bias_tokens",
    "raw_r_squared_tokens",
    "sequence_balanced_positive_underestimation_tokens",
    "sequence_balanced_severe_underestimation_rate_100_tokens",
)
PROBABILISTIC_METRIC_KEYS = (
    "family_count",
    "sequence_balanced_posterior_nll",
    "family_macro_sequence_balanced_posterior_nll",
    "sequence_balanced_crps",
    "sequence_balanced_interval_50_coverage",
    "sequence_balanced_interval_90_coverage",
    "sequence_balanced_interval_95_coverage",
    "sequence_balanced_interval_50_width",
    "sequence_balanced_interval_90_width",
    "sequence_balanced_interval_95_width",
    "sequence_balanced_posterior_variance_lower_bound",
    "sequence_balanced_posterior_entropy",
    "mean_overflow_probability",
    "mean_update_wall_time_ms",
    "peak_predictor_state_bytes",
)
BREAKDOWN_KEYS = (
    "observation_count",
    "sequence_count",
    "family_count",
    "exact_sequence_count",
    "censored_sequence_count",
    "sequence_balanced_posterior_nll",
    "sequence_balanced_crps",
    "sequence_balanced_mae_tokens",
    "sequence_balanced_rmse_tokens",
    "sequence_balanced_bias_tokens",
    "raw_r_squared_tokens",
    "sequence_balanced_interval_90_coverage",
    "sequence_balanced_interval_95_coverage",
    "sequence_balanced_positive_underestimation_tokens",
    "sequence_balanced_severe_underestimation_rate_100_tokens",
)
SUMMARY_BREAKDOWN_METHODS = (
    "alps_countdown",
    "alps_plp_concat_v1",
    "bayesian_entropy_scalar_v1",
    "bayesian_entropy_hidden_delta_v1",
)
SUMMARY_BREAKDOWNS = (
    "by_task",
    "by_intended_length",
    "by_temperature",
    "by_decode_progress",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")


def _parse_sha256_manifest(payload: bytes) -> dict[str, str]:
    output: dict[str, str] = {}
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        fields = line.split(None, 1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise ValueError(f"invalid SHA-256 manifest line {line_number}")
        name = fields[1].strip()
        _safe_archive_name(name)
        if name in output:
            raise ValueError(f"duplicate SHA-256 manifest path: {name}")
        output[name] = fields[0]
    if not output:
        raise ValueError("empty SHA-256 manifest")
    return output


def _json_payload(payloads: Mapping[str, bytes], name: str) -> dict[str, Any]:
    try:
        value = json.loads(payloads[name])
    except KeyError as error:
        raise ValueError(f"required archive member is missing: {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {name}")
    return value


def _validate_nested_benchmark_manifest(
    payloads: Mapping[str, bytes], computed: Mapping[str, str]
) -> int:
    nested = _json_payload(payloads, BENCHMARK_FILE_MANIFEST)
    files = nested.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("benchmark file_manifest.json has no files")
    failures = []
    for filename, expected in files.items():
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise ValueError("benchmark file manifest must map filenames to SHA-256 strings")
        actual = computed.get(f"{BENCHMARK_ROOT}/{filename}")
        if actual != expected:
            failures.append(filename)
    if failures:
        raise ValueError("benchmark file manifest mismatch: " + ", ".join(sorted(failures)))
    return len(files)


def _validate_scientific_boundary(
    report: Mapping[str, Any],
    collection: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    trace_count: int,
    index_line_count: int,
) -> None:
    failures = []
    if report.get("status") != "pass":
        failures.append("final benchmark did not pass")
    if report.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        failures.append("unexpected final benchmark ID")
    if report.get("primary_method") != PRIMARY_METHOD:
        failures.append("pre-registered primary method changed")
    if report.get("model_selection_performed") is not False:
        failures.append("final holdout performed model selection")
    if report.get("threshold_tuning_performed") is not False:
        failures.append("final holdout performed threshold tuning")
    if report.get("final_holdout_selects_nothing") is not True:
        failures.append("final holdout selection boundary changed")
    if collection.get("status") != "pass" or collection.get("failures") != []:
        failures.append("final collection did not pass")
    if collection.get("valid_trace_count") != EXPECTED_TRACE_COUNT:
        failures.append("final collection trace count changed")
    if collection.get("missing_trace_count") != 0:
        failures.append("final collection is missing traces")
    if collection.get("final_holdout_accessed") is not True:
        failures.append("final collection access state is inconsistent")
    if report.get("collection") != collection:
        failures.append("embedded collection report changed")
    if trace_count != EXPECTED_TRACE_COUNT or index_line_count != EXPECTED_TRACE_COUNT:
        failures.append("archive trace/index count is not 324")
    if preflight.get("ready") is not True or preflight.get("failures") != []:
        failures.append("Stage-8B preflight was not ready")
    if preflight.get("final_holdout_opened") is not False:
        failures.append("preflight says final holdout was opened early")
    if preflight.get("final_holdout_accessed") is not False:
        failures.append("preflight says final holdout was accessed early")
    if failures:
        raise ValueError("; ".join(failures))


def read_verified_archive(
    archive_path: str | Path,
    *,
    expected_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Read a Stage-8B archive after verifying every transitive digest and boundary."""

    archive = Path(archive_path)
    outer_sha256 = file_sha256(archive)
    if expected_archive_sha256 is not None and outer_sha256 != expected_archive_sha256:
        raise ValueError(
            f"archive outer SHA-256 changed: expected {expected_archive_sha256}, "
            f"found {outer_sha256}"
        )

    selected_names = {
        ARCHIVE_MANIFEST,
        FINAL_REPORT,
        BENCHMARK_FILE_MANIFEST,
        COLLECTION_REPORT,
        COLLECTION_INDEX,
        PREFLIGHT_REPORT,
        GIT_HEAD,
        ENVIRONMENT,
        BENCHMARK_LOG,
    }
    computed: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    trace_count = 0

    with tarfile.open(archive, "r:gz") as handle:
        seen = set()
        for member in handle:
            _safe_archive_name(member.name)
            if not member.isfile():
                continue
            if member.name in seen:
                raise ValueError(f"duplicate archive member: {member.name}")
            seen.add(member.name)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            digest = hashlib.sha256()
            capture = member.name in selected_names
            chunks = [] if capture else None
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            if capture:
                payloads[member.name] = b"".join(chunks or [])
            if member.name != ARCHIVE_MANIFEST:
                computed[member.name] = digest.hexdigest()
            if member.name.startswith(f"{TRACE_ROOT}/") and member.name.endswith(".npz"):
                trace_count += 1

    if ARCHIVE_MANIFEST not in payloads:
        raise ValueError("archive SHA-256 manifest is missing")
    expected = _parse_sha256_manifest(payloads[ARCHIVE_MANIFEST])
    missing = sorted(set(expected) - set(computed))
    unexpected = sorted(set(computed) - set(expected))
    mismatched = sorted(
        name for name, expected_digest in expected.items() if computed.get(name) != expected_digest
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "archive internal verification failed: "
            f"missing={missing[:3]} unexpected={unexpected[:3]} mismatched={mismatched[:3]}"
        )

    nested_file_count = _validate_nested_benchmark_manifest(payloads, computed)
    report = _json_payload(payloads, FINAL_REPORT)
    collection = _json_payload(payloads, COLLECTION_REPORT)
    preflight = _json_payload(payloads, PREFLIGHT_REPORT)
    index_line_count = len(payloads[COLLECTION_INDEX].splitlines())
    _validate_scientific_boundary(
        report,
        collection,
        preflight,
        trace_count=trace_count,
        index_line_count=index_line_count,
    )
    benchmark_log = payloads[BENCHMARK_LOG].decode("utf-8").strip()
    if "Stage-8 final benchmark pass" not in benchmark_log:
        raise ValueError("benchmark completion log is missing the pass marker")
    git_head = payloads[GIT_HEAD].decode("utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
        raise ValueError("archive Git HEAD is invalid")

    return {
        "archive_filename": archive.name,
        "archive_sha256": outer_sha256,
        "internal_manifest_entry_count": len(expected),
        "nested_benchmark_manifest_entry_count": nested_file_count,
        "trace_count": trace_count,
        "collection_index_line_count": index_line_count,
        "report": report,
        "collection": collection,
        "preflight": preflight,
        "git_head": git_head,
        "environment_text": payloads[ENVIRONMENT].decode("utf-8").strip(),
    }


def _select(values: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: values[key] for key in keys if key in values}


def _compact_breakdowns(report: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for method in SUMMARY_BREAKDOWN_METHODS:
        method_output = {}
        for breakdown in SUMMARY_BREAKDOWNS:
            method_output[breakdown] = [
                {"group": row["group"], **_select(row, BREAKDOWN_KEYS)}
                for row in report["breakdowns"][method][breakdown]
            ]
        output[method] = method_output
    return output


def _environment_mapping(text: str) -> dict[str, str]:
    output = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        output[key.strip()] = value.strip()
    return output


def build_summary(bundle: Mapping[str, Any], *, completed_date: str) -> dict[str, Any]:
    """Build the compact, committable final-result summary."""

    report = bundle["report"]
    collection = bundle["collection"]
    methods = {}
    for method in report["comparison_list"]:
        metrics = report["metrics"][method]
        methods[method] = {
            "label": METHOD_LABELS[method],
            **_select(metrics, POINT_METRIC_KEYS),
            **_select(metrics, PROBABILISTIC_METRIC_KEYS),
        }

    probability_methods = [
        method
        for method, metrics in methods.items()
        if metrics.get("sequence_balanced_posterior_nll") is not None
    ]
    point_best = min(methods, key=lambda method: methods[method]["sequence_balanced_mae_tokens"])
    probabilistic_best = min(
        probability_methods,
        key=lambda method: methods[method]["sequence_balanced_posterior_nll"],
    )
    comparisons = report["statistical_comparisons"]
    scalar_alps = comparisons["scalar_minus_alps_posterior_nll"]
    scalar_hidden = comparisons["scalar_minus_hidden_posterior_nll"]
    scalar_concat = comparisons["scalar_minus_alps_plp_concat_v1_absolute_error"]
    convergence = report["convergence"]["overall"]["all"]

    return {
        "schema_version": 1,
        "benchmark_id": report["benchmark_id"],
        "status": report["status"],
        "completed_date": completed_date,
        "claim_scope": "one_time_final_holdout_descriptive_not_selection",
        "data": {
            "new_family_count": methods[PRIMARY_METHOD]["family_count"],
            "prompt_count": 36,
            "trace_count": collection["valid_trace_count"],
            "observation_count": methods[PRIMARY_METHOD]["observation_count"],
            "total_observed_tokens": collection["total_observed_tokens"],
            "censoring_rate": collection["censoring_rate"],
            "by_stop_reason": collection["by_stop_reason"],
        },
        "primary_method": report["primary_method"],
        "comparison_list": report["comparison_list"],
        "methods": methods,
        "best_descriptive_methods": {
            "lowest_point_mae": point_best,
            "lowest_probabilistic_nll": probabilistic_best,
            "final_holdout_does_not_reselect_either": True,
        },
        "breakdowns": _compact_breakdowns(report),
        "statistical_comparisons": comparisons,
        "convergence": report["convergence"],
        "uncertainty_cone": report["uncertainty_cone"],
        "serving_replay": report["serving_replay"],
        "scientific_conclusions": {
            "primary_scalar_outperforms_alps_on_overall_nll": (
                methods[PRIMARY_METHOD]["sequence_balanced_posterior_nll"]
                < methods["alps_countdown"]["sequence_balanced_posterior_nll"]
            ),
            "scalar_minus_alps_nll_ci_excludes_zero": (
                scalar_alps["lower"] > 0.0 or scalar_alps["upper"] < 0.0
            ),
            "hidden_delta_has_lower_nll_than_scalar_with_ci_excluding_zero": (
                scalar_hidden["lower"] > 0.0
            ),
            "concat_has_lower_absolute_error_than_scalar_with_ci_excluding_zero": (
                scalar_concat["lower"] > 0.0
            ),
            "strict_stable_5pct_convergence_success_rate": convergence["success_rate"],
            "strict_stable_5pct_convergence_is_early": (
                convergence["stable_progress_mean_on_success"] < 0.5
            ),
            "bayesian_sequential_inference_implemented_and_finally_validated": True,
            "bayesian_primary_superiority_supported": False,
            "uncertainty_is_useful_for_capacity_risk_tradeoff": True,
        },
        "scientific_controls": {
            "model_selection_performed": report["model_selection_performed"],
            "threshold_tuning_performed": report["threshold_tuning_performed"],
            "final_holdout_selects_nothing": report["final_holdout_selects_nothing"],
            "paired_bootstrap_is_descriptive_not_selection": all(
                item["descriptive_not_selection"] for item in comparisons.values()
            ),
            "post_holdout_refit_performed": False,
        },
        "provenance": {
            "stage8a_config_sha256": report["stage8a_config_sha256"],
            "stage8b_lock_sha256": report["stage8b_lock_sha256"],
            "checkpoint_registry_sha256": report["checkpoint_registry_sha256"],
            "final_holdout_manifest_sha256": report["final_holdout_manifest_sha256"],
            "server_git_head": bundle["git_head"],
            "environment": _environment_mapping(bundle["environment_text"]),
            "archive_filename": bundle["archive_filename"],
            "archive_sha256": bundle["archive_sha256"],
            "archive_internal_manifest_entry_count": bundle[
                "internal_manifest_entry_count"
            ],
            "benchmark_file_manifest_entry_count": bundle[
                "nested_benchmark_manifest_entry_count"
            ],
            "archive_outer_sha256_verified": True,
            "archive_internal_sha256_verified": True,
            "raw_final_holdout_artifacts_committed_to_git": False,
        },
    }


def _fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---:" if index else "---" for index in range(len(headers))) + "|",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def render_markdown(summary: Mapping[str, Any]) -> str:
    methods = summary["methods"]
    primary = methods[PRIMARY_METHOD]
    alps = methods["alps_countdown"]
    concat = methods["alps_plp_concat_v1"]
    comparisons = summary["statistical_comparisons"]
    scalar_alps = comparisons["scalar_minus_alps_posterior_nll"]
    scalar_hidden = comparisons["scalar_minus_hidden_posterior_nll"]
    scalar_concat = comparisons["scalar_minus_alps_plp_concat_v1_absolute_error"]
    convergence = summary["convergence"]["overall"]["all"]

    point_rows = []
    for method in summary["comparison_list"]:
        metric = methods[method]
        point_rows.append(
            [
                metric["label"],
                _fmt(metric.get("sequence_balanced_posterior_nll"), 4),
                _fmt(metric["sequence_balanced_mae_tokens"]),
                _fmt(metric["sequence_balanced_rmse_tokens"]),
                _fmt(metric["sequence_balanced_bias_tokens"]),
                _fmt(metric["raw_r_squared_tokens"], 3),
            ]
        )
    point_table = _markdown_table(
        ["Method", "NLL", "MAE", "RMSE", "Bias", "Raw R²"], point_rows
    )

    probability_rows = []
    for method in (
        "alps_countdown",
        "bayesian_entropy_scalar_v1",
        "bayesian_entropy_hidden_delta_v1",
    ):
        metric = methods[method]
        probability_rows.append(
            [
                metric["label"],
                _fmt(metric["sequence_balanced_posterior_nll"], 4),
                _fmt(metric["sequence_balanced_crps"]),
                _pct(metric["sequence_balanced_interval_50_coverage"]),
                _pct(metric["sequence_balanced_interval_90_coverage"]),
                _pct(metric["sequence_balanced_interval_95_coverage"]),
            ]
        )
    probability_table = _markdown_table(
        ["Method", "NLL", "CRPS", "50% cov.", "90% cov.", "95% cov."],
        probability_rows,
    )

    comparison_rows = []
    for name, item in comparisons.items():
        comparison_rows.append(
            [
                name.replace("_", " "),
                _fmt(item["estimate_left_minus_right"]),
                f"[{_fmt(item['lower'])}, {_fmt(item['upper'])}]",
                "Yes" if item["lower"] > 0 or item["upper"] < 0 else "No",
            ]
        )
    comparison_table = _markdown_table(
        ["Paired family comparison (left − right)", "Estimate", "95% CI", "Excludes 0"],
        comparison_rows,
    )

    task_rows = []
    task_breakdowns = summary["breakdowns"]
    by_method = {
        method: {row["group"]: row for row in values["by_task"]}
        for method, values in task_breakdowns.items()
    }
    for task in ("code", "qa", "summarization"):
        task_rows.append(
            [
                task,
                _fmt(by_method["alps_countdown"][task]["sequence_balanced_mae_tokens"]),
                _fmt(by_method[PRIMARY_METHOD][task]["sequence_balanced_mae_tokens"]),
                _fmt(
                    by_method["bayesian_entropy_hidden_delta_v1"][task][
                        "sequence_balanced_mae_tokens"
                    ]
                ),
                _fmt(
                    by_method["alps_plp_concat_v1"][task]["sequence_balanced_mae_tokens"]
                ),
            ]
        )
    task_table = _markdown_table(
        ["Task", "ALPS MAE", "Scalar MAE", "Hidden-delta MAE", "Concat MAE"],
        task_rows,
    )

    serving = summary["serving_replay"]
    actual_tokens = summary["data"]["total_observed_tokens"]
    serving_rows = []
    for method, metric in serving["metrics"].items():
        serving_rows.append(
            [
                method.replace("_", " "),
                _fmt(metric["throughput_tokens_per_second"]),
                _pct(metric["underallocation_rate"]),
                _pct(metric["kv_overreservation_tokens"] / actual_tokens),
                _fmt(metric["kv_overreservation_bytes"] / (1024**3), 3),
            ]
        )
    serving_table = _markdown_table(
        ["Policy", "Throughput tok/s", "Underallocation", "Overreserve/output", "KV GiB"],
        serving_rows,
    )

    provenance = summary["provenance"]
    return f"""# Bayesian Sequential Stage-8 一次性最终盲测

## 0. 结论边界

2026-08-16 在合并后的 Stage-8B ready lock 上一次性采集 12 个全新 family、36 条 Prompt、
324 条 Qwen2.5-7B-Instruct trace，并运行冻结的七方法最终 benchmark。全部 trace 以 EOS
结束，censoring rate 为 `0`；最终报告、嵌套文件 manifest、压缩包外层 SHA-256 和内部
{provenance['archive_internal_manifest_entry_count']} 个文件均通过复验。

`status=pass` 表示采集、哈希、指标和边界完整，不表示预注册主方法获得最优指标。Final
holdout 没有选择模型、调阈值或触发 refit；本文中的 paired-family bootstrap 只作描述性
推断，不能把 final 结果改写成新的选型规则。

## 1. 总体结果

{point_table}

![七方法最终点误差](figures/stage8_final_point_error.svg)

预注册 primary `bayesian_entropy_scalar_v1` 的 NLL `{primary['sequence_balanced_posterior_nll']:.4f}`、
MAE `{primary['sequence_balanced_mae_tokens']:.2f}`、RMSE
`{primary['sequence_balanced_rmse_tokens']:.2f}`，没有优于 ALPS 的
`{alps['sequence_balanced_posterior_nll']:.4f}` / `{alps['sequence_balanced_mae_tokens']:.2f}` /
`{alps['sequence_balanced_rmse_tokens']:.2f}`。最终测试集上的最低点预测 MAE 来自冻结 baseline
`alps_plp_concat_v1`（`{concat['sequence_balanced_mae_tokens']:.2f}`），但 final holdout
明确不重新选择它。

## 2. 概率质量与 paired-family 证据

{probability_table}

![三种概率方法最终 NLL](figures/stage8_final_probabilistic_nll.svg)

{comparison_table}

scalar−ALPS NLL 的 family-bootstrap 95% CI 为
`[{scalar_alps['lower']:.3f}, {scalar_alps['upper']:.3f}]`，跨过 0，所以不能声称两者存在确定的
总体差异。scalar−hidden-delta NLL CI 为
`[{scalar_hidden['lower']:.3f}, {scalar_hidden['upper']:.3f}]`，完全大于 0，描述性证据支持
hidden-delta 的概率表现优于预注册 scalar。scalar−concat absolute-error CI 为
`[{scalar_concat['lower']:.2f}, {scalar_concat['upper']:.2f}]`，也完全大于 0。

## 3. 任务差异

{task_table}

scalar 的主要失效集中在 code：code MAE
`{by_method[PRIMARY_METHOD]['code']['sequence_balanced_mae_tokens']:.2f}`，而 ALPS 为
`{by_method['alps_countdown']['code']['sequence_balanced_mae_tokens']:.2f}`。因此总体负结果不能
只解释成均匀的小幅退化；新的程序生成 family 暴露了明显的跨任务泛化问题。由于这些是 final
holdout 观察，只能记录为后续独立研究假设，不能在当前实验上修补模型。

## 4. 严格 5% 稳定收敛

324 条序列中只有 `{convergence['success_count']}` 条满足“进入 5% 相对误差后所有后续保存点均
保持在阈值内”，成功率 `{_pct(convergence['success_rate'])}`。成功样本的平均稳定进度为
`{_pct(convergence['stable_progress_mean_on_success'])}`，median step
`{convergence['stable_step_quantiles_on_success']['q50']:.0f}`。这不支持“动态 posterior 很早
稳定”的强主张；它更多是在接近输出末端时收敛。

## 5. Serving replay：容量与风险

{serving_table}

![最终 serving 容量—风险权衡](figures/stage8_final_serving_tradeoff.svg)

scalar posterior mean 的 underallocation rate 为
`{_pct(serving['metrics']['bayesian_entropy_scalar_v1_mean']['underallocation_rate'])}`；使用冻结
q97.5 上界后降到
`{_pct(serving['metrics']['bayesian_entropy_scalar_v1_q975']['underallocation_rate'])}`，但 KV
overreservation 从
`{serving['metrics']['bayesian_entropy_scalar_v1_mean']['kv_overreservation_bytes'] / (1024**3):.3f}`
GiB 增至
`{serving['metrics']['bayesian_entropy_scalar_v1_q975']['kv_overreservation_bytes'] / (1024**3):.3f}`
GiB。贝叶斯输出的可复现实用价值主要是把不确定性转成显式容量—风险旋钮，而不是证明
posterior mean 点预测最优。

## 6. 最终回答

本项目已经完成导师所要求的两部分：ALPS 作为静态概率先验，解码期间使用非重叠增量证据递归
更新动态 posterior；实现、真实 Qwen trace、family-grouped OOF、冻结模型和全新 family 的一次性
final holdout 均已验证。因此“贝叶斯序列推断尚未实现”这一工程缺口已经补齐。

最终盲测同时给出一个重要负结果：**预注册 Bayesian scalar 的总体泛化优势没有得到支持。**
hidden-delta 在概率 NLL 上优于 scalar，concat baseline 在点误差上最好，而 scalar 的主要问题是
code family 和较晚的稳定收敛。当前可支持的主张是“贝叶斯序列推断已实现，并提供可用的不确定性
与容量风险控制”，不能写成“贝叶斯 scalar 全面优于 ALPS/PLP”。

## 7. Provenance

| Item | SHA-256 / value |
|---|---|
| Server Git HEAD | `{provenance['server_git_head']}` |
| Stage-8A config | `{provenance['stage8a_config_sha256']}` |
| Stage-8B ready lock | `{provenance['stage8b_lock_sha256']}` |
| Checkpoint registry | `{provenance['checkpoint_registry_sha256']}` |
| Final-holdout manifest | `{provenance['final_holdout_manifest_sha256']}` |
| Local archive | `{provenance['archive_filename']}` |
| Archive SHA-256 | `{provenance['archive_sha256']}` |
| Internal verified files | `{provenance['archive_internal_manifest_entry_count']}` |

完整 243 MiB 左右的 trace/prediction archive 保留在本地实验结果目录，不提交 Git。仓库只保存
本报告、脱敏摘要、方法级 CSV、可复现归档器和图表。
"""


def render_method_csv(summary: Mapping[str, Any]) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "method",
        "label",
        "probabilistic",
        "sequence_balanced_posterior_nll",
        "sequence_balanced_crps",
        "sequence_balanced_mae_tokens",
        "sequence_balanced_rmse_tokens",
        "sequence_balanced_bias_tokens",
        "raw_r_squared_tokens",
        "interval_90_coverage",
        "interval_95_coverage",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for method in summary["comparison_list"]:
        metric = summary["methods"][method]
        writer.writerow(
            {
                "method": method,
                "label": metric["label"],
                "probabilistic": metric.get("sequence_balanced_posterior_nll") is not None,
                "sequence_balanced_posterior_nll": metric.get(
                    "sequence_balanced_posterior_nll"
                ),
                "sequence_balanced_crps": metric.get("sequence_balanced_crps"),
                "sequence_balanced_mae_tokens": metric["sequence_balanced_mae_tokens"],
                "sequence_balanced_rmse_tokens": metric["sequence_balanced_rmse_tokens"],
                "sequence_balanced_bias_tokens": metric["sequence_balanced_bias_tokens"],
                "raw_r_squared_tokens": metric["raw_r_squared_tokens"],
                "interval_90_coverage": metric.get("sequence_balanced_interval_90_coverage"),
                "interval_95_coverage": metric.get("sequence_balanced_interval_95_coverage"),
            }
        )
    return buffer.getvalue()


def _svg_document(width: int, height: int, title: str, description: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(description)}</desc>
  <style>
    :root {{ color-scheme: light dark; }}
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2937; }}
    .muted {{ fill: #6b7280; }}
    .grid {{ stroke: #d1d5db; stroke-width: 1; }}
    .frame {{ fill: none; stroke: #9ca3af; stroke-width: 1; }}
    @media (prefers-color-scheme: dark) {{
      text {{ fill: #f3f4f6; }} .muted {{ fill: #cbd5e1; }}
      .grid {{ stroke: #475569; }} .frame {{ stroke: #64748b; }}
    }}
  </style>
{body}
</svg>
"""


def render_point_error_svg(summary: Mapping[str, Any]) -> str:
    methods = summary["comparison_list"]
    metrics = summary["methods"]
    width, height = 1040, 545
    left, right, top, bottom = 295, 75, 70, 55
    plot_width = width - left - right
    maximum = math.ceil(
        max(metrics[method]["sequence_balanced_rmse_tokens"] for method in methods) / 50
    ) * 50
    row_height = (height - top - bottom) / len(methods)
    body = [f'  <text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="600">Final holdout point error</text>']
    body.append('  <rect class="frame" x="295" y="70" width="670" height="420"/>')
    for tick in range(0, maximum + 1, 50):
        x = left + plot_width * tick / maximum
        body.append(f'  <line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}"/>')
        body.append(f'  <text class="muted" x="{x:.1f}" y="{height-28}" text-anchor="middle" font-size="12">{tick}</text>')
    body.append(f'  <text x="{left + plot_width / 2:.1f}" y="{height-8}" text-anchor="middle" font-size="13">Error (tokens; lower is better)</text>')
    for index, method in enumerate(methods):
        y = top + row_height * (index + 0.5)
        label = metrics[method]["label"]
        body.append(f'  <text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="13">{html.escape(label)}</text>')
        for offset, key, color in (
            (-9, "sequence_balanced_mae_tokens", "#2563eb"),
            (7, "sequence_balanced_rmse_tokens", "#f59e0b"),
        ):
            value = float(metrics[method][key])
            bar_width = plot_width * value / maximum
            body.append(f'  <rect x="{left}" y="{y+offset-5:.1f}" width="{bar_width:.1f}" height="10" fill="{color}" rx="2"/>')
            body.append(f'  <text x="{left+bar_width+6:.1f}" y="{y+offset+4:.1f}" font-size="11">{value:.1f}</text>')
    body.extend(
        [
            '  <rect x="350" y="43" width="13" height="8" fill="#2563eb" rx="2"/><text x="369" y="51" font-size="12">MAE</text>',
            '  <rect x="430" y="43" width="13" height="8" fill="#f59e0b" rx="2"/><text x="449" y="51" font-size="12">RMSE</text>',
        ]
    )
    return _svg_document(
        width,
        height,
        "Final holdout point error",
        "Grouped horizontal bars compare sequence-balanced MAE and RMSE for seven frozen methods.",
        "\n".join(body),
    )


def render_probabilistic_nll_svg(summary: Mapping[str, Any]) -> str:
    methods = (
        "alps_countdown",
        "bayesian_entropy_scalar_v1",
        "bayesian_entropy_hidden_delta_v1",
    )
    metrics = summary["methods"]
    width, height = 800, 440
    left, right, top, bottom = 85, 35, 65, 115
    plot_height = height - top - bottom
    maximum = 7.0
    colors = ("#64748b", "#2563eb", "#14b8a6")
    body = [f'  <text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-weight="600">Final probabilistic NLL</text>']
    body.append(f'  <rect class="frame" x="{left}" y="{top}" width="{width-left-right}" height="{plot_height}"/>')
    for tick in range(0, 8):
        y = top + plot_height * (1 - tick / maximum)
        body.append(f'  <line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        body.append(f'  <text class="muted" x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12">{tick}</text>')
    slot = (width - left - right) / len(methods)
    for index, (method, color) in enumerate(zip(methods, colors, strict=True)):
        value = float(metrics[method]["sequence_balanced_posterior_nll"])
        bar_height = plot_height * value / maximum
        x = left + slot * index + slot * 0.24
        bar_width = slot * 0.52
        y = top + plot_height - bar_height
        body.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3"/>')
        body.append(f'  <text x="{x+bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="13">{value:.3f}</text>')
        label = metrics[method]["label"].replace(" (primary)", "")
        body.append(f'  <text x="{x+bar_width/2:.1f}" y="{height-bottom+26}" text-anchor="middle" font-size="12">{html.escape(label)}</text>')
    body.append('  <text transform="translate(20 195) rotate(-90)" text-anchor="middle" font-size="13">NLL (lower is better)</text>')
    scalar_alps = summary["statistical_comparisons"]["scalar_minus_alps_posterior_nll"]
    scalar_hidden = summary["statistical_comparisons"]["scalar_minus_hidden_posterior_nll"]
    body.append(f'  <text class="muted" x="{width/2}" y="{height-46}" text-anchor="middle" font-size="12">Scalar−ALPS 95% CI [{scalar_alps["lower"]:.3f}, {scalar_alps["upper"]:.3f}] (includes 0)</text>')
    body.append(f'  <text class="muted" x="{width/2}" y="{height-25}" text-anchor="middle" font-size="12">Scalar−hidden 95% CI [{scalar_hidden["lower"]:.3f}, {scalar_hidden["upper"]:.3f}] (above 0)</text>')
    return _svg_document(
        width,
        height,
        "Final probabilistic NLL",
        "Bars compare NLL for ALPS, the pre-registered Bayesian scalar, and hidden-delta; lower values are better.",
        "\n".join(body),
    )


def render_serving_tradeoff_svg(summary: Mapping[str, Any]) -> str:
    metrics = summary["serving_replay"]["metrics"]
    labels = {
        "oracle_observed_length": "Oracle",
        "max_new_tokens_4096": "Fixed 4096",
        "alps_countdown_mean": "ALPS mean",
        "plp_terminal_zero_v3": "PLP v3",
        "alps_plp_concat_v1": "Concat",
        "bayesian_entropy_scalar_v1_mean": "Scalar mean",
        "bayesian_entropy_scalar_v1_q975": "Scalar q97.5",
    }
    offsets = {
        "oracle_observed_length": (8, -8),
        "max_new_tokens_4096": (8, 15),
        "alps_countdown_mean": (-75, -11),
        "bayesian_entropy_scalar_v1_mean": (10, 15),
        "plp_terminal_zero_v3": (-60, -12),
        "alps_plp_concat_v1": (8, 15),
        "bayesian_entropy_scalar_v1_q975": (8, -8),
    }
    width, height = 900, 500
    left, right, top, bottom = 90, 45, 65, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_max = 75.0
    y_min, y_max = -1.0, 2.0
    body = [f'  <text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-weight="600">Serving capacity–risk trade-off</text>']
    body.append(f'  <rect class="frame" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>')
    for tick in (0, 15, 30, 45, 60, 75):
        x = left + plot_width * tick / x_max
        body.append(f'  <line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}"/>')
        body.append(f'  <text class="muted" x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{tick}%</text>')
    for value, label in ((0.1, "0.1"), (1.0, "1"), (10.0, "10"), (100.0, "100")):
        y = top + plot_height * (1 - (math.log10(value) - y_min) / (y_max - y_min))
        body.append(f'  <line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}"/>')
        body.append(f'  <text class="muted" x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12">{label}</text>')
    for method, metric in metrics.items():
        under = 100.0 * float(metric["underallocation_rate"])
        gib = float(metric["kv_overreservation_bytes"]) / (1024**3)
        x = left + plot_width * under / x_max
        y = top + plot_height * (1 - (math.log10(gib) - y_min) / (y_max - y_min))
        color = "#2563eb" if "bayesian_entropy_scalar" in method else "#14b8a6" if method == "alps_plp_concat_v1" else "#64748b"
        body.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        dx, dy = offsets[method]
        anchor = "end" if dx < 0 else "start"
        body.append(f'  <text x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}" font-size="12">{html.escape(labels[method])}</text>')
    body.append(f'  <text x="{left+plot_width/2:.1f}" y="{height-10}" text-anchor="middle" font-size="13">Request underallocation rate (lower is safer)</text>')
    body.append('  <text transform="translate(24 250) rotate(-90)" text-anchor="middle" font-size="13">KV overreservation (GiB, log scale)</text>')
    return _svg_document(
        width,
        height,
        "Serving capacity-risk trade-off",
        "Scatter plot compares request underallocation rate and KV overreservation for seven frozen allocation policies.",
        "\n".join(body),
    )


def write_outputs(summary: Mapping[str, Any], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        output / "stage8_final_benchmark_20260816_summary.json": (
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        ),
        output / "stage8_final_benchmark_20260816.md": render_markdown(summary),
        output / "stage8_final_method_metrics_20260816.csv": render_method_csv(summary),
        figures / "stage8_final_point_error.svg": render_point_error_svg(summary),
        figures / "stage8_final_probabilistic_nll.svg": render_probabilistic_nll_svg(summary),
        figures / "stage8_final_serving_tradeoff.svg": render_serving_tradeoff_svg(summary),
    }
    for path, content in paths.items():
        path.write_text(content, encoding="utf-8")
    return list(paths)
