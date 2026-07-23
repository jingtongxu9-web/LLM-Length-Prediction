# Artifacts

This directory contains compact experiment results and examples. Raw rollout traces belong under
`data/interim/`; downloaded Qwen weights belong under `models/` or an external `MODEL_PATH`.

Expected ALPS v1 runtime layout:

```text
artifacts/
|-- examples/
|   `-- first_trace.jsonl       # committed tiny-model pipeline example
`-- runs/                       # generated locally and ignored by Git
    `-- alps_v1/
        |-- environment/
        |   `-- preflight.json
        |-- collection_index.jsonl
        |-- collection_summary.json
        `-- stage1/
            |-- prior.json      # StandardScaler, Ridge weights, bias, variance, provenance
            |-- metrics.json
            |-- predictions.csv
            |-- train_evaluation.json
            `-- test_evaluation.json
```

Every reported result should identify the experiment config, code commit, prompt-manifest hash,
model/tokenizer revision, trace checksums, seeds, hardware, and software runtime. Generated run
directories are ignored by Git, so copy them to durable storage before releasing a rented machine.

`examples/first_trace.jsonl` is a small, committed Hugging Face smoke trace. It proves that the
collector, serializer, and schema validator work end to end without treating the example model as
a research result.
