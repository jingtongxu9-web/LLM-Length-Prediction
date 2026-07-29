"""Named ALPS diagnostic baselines.

Feature construction and leakage-safe grouped evaluation live in
``llm_length_prediction.evaluation.grouped_cv``. This module keeps the frozen
baseline registry in one place so reports cannot silently omit a comparator.
"""

BASELINE_MODELS = (
    "global_mean",
    "prompt_tokens",
    "metadata",
    "metadata_prompt_tokens",
)

ALPS_MODELS = ("alps_hidden",)

ALL_DIAGNOSTIC_MODELS = BASELINE_MODELS + ALPS_MODELS
