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
            |-- test_evaluation.json
            |-- train_breakdown.{json,csv,md}
            |-- test_breakdown.{json,csv,md}
            |-- train_prompt_mean_breakdown.csv
            |-- test_prompt_mean_breakdown.csv
            |-- train_length_contrasts.csv
            `-- test_length_contrasts.csv
```

Every reported result should identify the experiment config, code commit, prompt-manifest hash,
model/tokenizer revision, trace checksums, seeds, hardware, and software runtime. Generated run
directories are ignored by Git, so copy them to durable storage before releasing a rented machine.

The breakdown reports preserve aggregate rollout metrics and add intended-length, task-type, 3x3
interaction, seed, and matched short/medium/long analyses. Prompt-mean tables average the three
observed seed lengths before evaluating ALPS point accuracy; rollout rows remain the correct unit
for NLL and interval coverage.

`examples/first_trace.jsonl` is a small, committed Hugging Face smoke trace. It proves that the
collector, serializer, and schema validator work end to end without treating the example model as
a research result.
