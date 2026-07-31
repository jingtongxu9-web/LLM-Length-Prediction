# Tests

The tests provide fast checks for data contracts and small mathematical components. They do not
run the 7B model or prove that a full GPU experiment will fit in memory.

Current coverage includes:

- frozen experiment revision, counts, and prompt-manifest checksum;
- cross-platform LF prompt-manifest generation and stale-trace rejection;
- model snapshot structure and missing weight-shard detection;
- deterministic prompt-family Train/Test grouping;
- local/`MODEL_PATH` model-source resolution;
- JSONL trace serialization and schema validation;
- `log1p` Ridge fitting, prediction, and saved-model round trips.
- five-fold family isolation and frozen-alpha grouped-CV output contracts;
- length/task/interaction/seed metric breakdowns and matched-family length contrasts;
- end-to-end grouped-CV, final all-Train fit, Train/Test prediction, and breakdown generation.
- input-token Ridge final fitting and guarded Test evaluation;
- Dynamic-Signal MLP v1 non-terminal sample construction, per-sequence weighting, JSON model round trip, and
  decode-progress metrics.

The local suite does not train the PyTorch Dynamic-Signal MLP. Real MLP training is exercised on the server
runtime that provides PyTorch; local tests cover its data contract and framework-independent JSON/
NumPy inference path.

After installing the project, run from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

Use `scripts/preflight_server.py` for real CUDA, BF16, model snapshot, disk, and output-directory
checks. Use the six-rollout pilot for the first end-to-end GPU validation.
