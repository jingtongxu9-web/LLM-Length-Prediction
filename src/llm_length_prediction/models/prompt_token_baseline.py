"""Prompt-token Ridge baseline for total and remaining output length."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from llm_length_prediction.models.hybrid import WeightedLogRidge, fit_log1p_ridge

METHOD_ID = "prompt_token_ridge_countdown"


def fit_prompt_token_ridge(
    prompt_tokens: Sequence[int], output_tokens: Sequence[int], *, alpha: float
) -> WeightedLogRidge:
    """Fit prompt-token count to total output length on one row per rollout."""

    inputs = np.asarray(prompt_tokens, dtype=np.float64)
    targets = np.asarray(output_tokens, dtype=np.float64)
    if (
        inputs.ndim != 1
        or targets.shape != inputs.shape
        or not len(inputs)
        or np.any(inputs <= 0)
        or np.any(targets <= 0)
    ):
        raise ValueError("prompt/output token counts must be aligned positive vectors")
    return fit_log1p_ridge(
        inputs[:, None],
        targets,
        np.ones(len(inputs), dtype=np.float64),
        alpha=alpha,
        target_name="log1p_output_tokens_from_prompt_tokens",
    )


def predict_prompt_token_countdown(
    model: WeightedLogRidge,
    prompt_tokens: Sequence[int],
    steps: Sequence[int],
) -> np.ndarray:
    """Predict total output length once, then subtract the current decode step."""

    inputs = np.asarray(prompt_tokens, dtype=np.float64)
    decode_steps = np.asarray(steps, dtype=np.float64)
    if inputs.ndim != 1 or decode_steps.shape != inputs.shape or np.any(inputs <= 0):
        raise ValueError("prompt tokens and steps must be aligned")
    predicted_total = model.predict_mean(inputs[:, None])
    return np.maximum(predicted_total - decode_steps, 0.0)
