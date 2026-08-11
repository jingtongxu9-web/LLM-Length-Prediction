"""Tests for Bayesian server preflight boundary handling."""

from scripts.preflight_bayesian_full_train import (
    _has_minimum_gpu_memory as _full_train_has_minimum_gpu_memory,
)
from scripts.preflight_bayesian_pilot import _has_minimum_gpu_memory


def test_nominal_24_gb_gpu_passes_even_when_binary_gib_is_lower() -> None:
    rtx_4090_reported_bytes = 25_260_000_000

    assert rtx_4090_reported_bytes / 1024**3 < 24
    assert _has_minimum_gpu_memory(rtx_4090_reported_bytes)
    assert _full_train_has_minimum_gpu_memory(rtx_4090_reported_bytes)


def test_gpu_below_nominal_24_gb_fails() -> None:
    assert not _has_minimum_gpu_memory(23_999_999_999)
    assert not _full_train_has_minimum_gpu_memory(23_999_999_999)
