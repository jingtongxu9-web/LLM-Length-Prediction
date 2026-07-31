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
| `comparison.py` | Load and validate shared Train/Test traces for frozen comparator methods | baseline/PLP scripts |
| `models/` | Ridge prior/baseline plus serializable Dynamic-Signal MLP v1 and sample builder | training/evaluation |
| `evaluation/` | Prediction, tail-risk, grouped-CV, and decode-progress metrics | training/evaluation scripts |
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

scripts/train_input_baseline.py
  -> comparison.py
  -> models/prior.py
  -> artifacts/runs/alps_v1/comparisons/input_token_ridge/

scripts/train_dynamic.py
  -> comparison.py
  -> models/dynamic.py
  -> evaluation/progressive.py
  -> artifacts/runs/alps_v1/comparisons/plp_only/
```

The Qwen weights and tokenizer are never trained here. Stage 1 fits small Ridge models; project
Dynamic-Signal MLP v1 fits a small project-defined MLP on already saved decode signals; it is not
the paper PLP architecture. `serving/simulator.py` remains a
foundation for the later serving benchmark.
