# Notebooks

Use notebooks for exploratory plots and diagnostics, not for the canonical collection or training
pipeline. Move reusable data loading, models, and metrics into `src/llm_length_prediction/` before
relying on them for reported results. A reported result must still be reproducible through a script
and the frozen experiment manifest.
