from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_length_prediction.stage8_archive import (
    file_sha256,
    read_verified_archive,
)

SUMMARY = Path(
    "docs/results/bayesian_sequential/stage8_final_benchmark_20260816_summary.json"
)
REPORT = Path("docs/results/bayesian_sequential/stage8_final_benchmark_20260816.md")
FIGURES = (
    Path("docs/results/bayesian_sequential/figures/stage8_final_point_error.svg"),
    Path("docs/results/bayesian_sequential/figures/stage8_final_probabilistic_nll.svg"),
    Path("docs/results/bayesian_sequential/figures/stage8_final_serving_tradeoff.svg"),
)


def test_stage8_final_summary_preserves_blind_benchmark_boundary() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["data"]["new_family_count"] == 12
    assert summary["data"]["prompt_count"] == 36
    assert summary["data"]["trace_count"] == 324
    assert summary["data"]["censoring_rate"] == 0.0
    assert summary["primary_method"] == "bayesian_entropy_scalar_v1"
    controls = summary["scientific_controls"]
    assert controls["model_selection_performed"] is False
    assert controls["threshold_tuning_performed"] is False
    assert controls["final_holdout_selects_nothing"] is True
    assert controls["post_holdout_refit_performed"] is False


def test_stage8_final_summary_records_negative_result_without_reselection() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    conclusions = summary["scientific_conclusions"]
    assert conclusions["bayesian_sequential_inference_implemented_and_finally_validated"]
    assert conclusions["primary_scalar_outperforms_alps_on_overall_nll"] is False
    assert conclusions["bayesian_primary_superiority_supported"] is False
    assert conclusions["hidden_delta_has_lower_nll_than_scalar_with_ci_excluding_zero"]
    assert conclusions["concat_has_lower_absolute_error_than_scalar_with_ci_excluding_zero"]
    assert summary["best_descriptive_methods"]["lowest_point_mae"] == "alps_plp_concat_v1"
    assert summary["best_descriptive_methods"]["lowest_probabilistic_nll"] == "alps_countdown"
    assert summary["best_descriptive_methods"]["final_holdout_does_not_reselect_either"]


def test_stage8_final_report_and_figures_are_accessible() -> None:
    report = REPORT.read_text(encoding="utf-8")
    assert "预注册 Bayesian scalar 的总体泛化优势没有得到支持" in report
    assert "holdout 没有选择模型、调阈值" in report
    for figure in FIGURES:
        text = figure.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert "role=\"img\"" in text
        assert "<title" in text
        assert "<desc" in text


def test_archive_outer_digest_fails_before_untrusted_payload_is_read(tmp_path: Path) -> None:
    archive = tmp_path / "not-the-frozen-archive.tar.gz"
    archive.write_bytes(b"changed")
    actual = file_sha256(archive)
    assert actual != "0" * 64
    with pytest.raises(ValueError, match="outer SHA-256 changed"):
        read_verified_archive(archive, expected_archive_sha256="0" * 64)
