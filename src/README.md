# Python package

`src/llm_length_prediction/` contains reusable implementation. Users normally run files under
`scripts/` rather than executing modules in `src/` directly.

## Module map

| Package | Responsibility | Used by |
|---|---|---|
| `experiment.py` | Load and validate the frozen manifest, prompt hash/counts, rollout jobs, paths, and trace provenance | preflight, collection, training, evaluation |
| `runtime/` | Resolve the Qwen source from an explicit path, `MODEL_PATH`, local `models/`, or Hub ID | preflight and collectors |
| `instrumentation/` | Load Transformers/Qwen, run prefill and decoding, capture Layer-14 features, entropy, EOS probability, timing, and text | trace and dataset collectors |
| `data/` | Define `GenerationTrace`/`TracePoint` and validate JSONL serialization | collectors, training, evaluation |
| `models/` | Fit, save, load, and apply the standardized Ridge shifted-log-normal prior; hold early PLP structures | prior training/evaluation |
| `evaluation/` | Prediction error and tail-risk metrics | training/evaluation scripts |
| `serving/` | Early scheduling/bucketing simulation structures | future benchmark |

## Current call flow

```text
scripts/collect_dataset.py
  -> runtime/model_paths.py
  -> instrumentation/huggingface.py
  -> data/schema.py + data/io.py
  -> data/interim/alps_v1/

scripts/train_prior.py
  -> data/io.py
  -> models/prior.py
  -> evaluation/metrics.py
  -> artifacts/runs/alps_v1/stage1/

scripts/evaluate_prior.py
  -> data/io.py
  -> models/prior.py
  -> evaluation/metrics.py
```

The Qwen weights and tokenizer are never trained here. Only the small Ridge probe is fitted in
Stage 1. `models/dynamic.py` and `serving/simulator.py` are foundations for later stages, not a
completed PLP/serving implementation.
