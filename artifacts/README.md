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
        |-- diagnostics/
        |   `-- grouped_cv/
        |       |-- validation.json  # frozen layer/alpha and family-fold contract
        |       |-- results.json     # OOF predictions and metrics
        |       `-- summary.csv      # ALPS and baseline comparison
        |-- stage1/
            |-- prior.json      # StandardScaler, Ridge weights, bias, variance, CV provenance
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
        `-- comparisons/
            |-- input_token_ridge/
            |   |-- model.json
            |   `-- {train,test}_evaluation.{json,csv}
            `-- plp_only/
                |-- model.json
                |-- training_report.json
                `-- {train,test}_evaluation.{json,csv}
    `-- plp_v2/
        |-- collection_index.jsonl
        |-- collection_summary.json
        |-- method_config.json
        |-- plp_head.pt
        |-- training_report.json
        `-- {train,test}_evaluation.{json,csv}
```

Every reported result should identify the experiment config, code commit, prompt-manifest hash,
model/tokenizer revision, trace checksums, seeds, hardware, and software runtime. Generated run
directories are ignored by Git, so copy them to durable storage before releasing a rented machine.
The compact, committed interpretation of the completed v1 outputs lives in
[`docs/results/v1/`](../docs/results/v1/README.md); this directory remains the machine-generated evidence,
not the narrative report.

The grouped-CV outputs validate the frozen Layer-14 / `alpha=1.0` configuration before the final
all-Train fit. They do not contain a deployable Ridge and must not be used to rewrite the opened v1
Test result.

`runs/alps_v1/comparisons/plp_only/` is the completed Dynamic-Signal MLP v1 baseline despite its
legacy directory name. The separate `runs/plp_v2/` directory is reserved for the new PLP-only
hidden-state checkpoint and evaluations; it remains absent until the GPU collection and training
commands are actually run.

The PLP collection summary records split/stop-reason counts, censored traces, point count, total
trace bytes, and observed CUDA peak memory. The training report separates loaded traces from
effective uncensored traces and records parameter, gradient, optimizer-state, feature-matrix, and
CUDA peak-memory bytes. PLP evaluations include overall, decode-progress, task, intended-length,
3x3 task-length, and seed breakdowns. `method_config.json` is the exact archived configuration whose
SHA-256 is embedded in the checkpoint.

The breakdown reports preserve aggregate rollout metrics and add intended-length, task-type, 3x3
interaction, seed, and matched short/medium/long analyses. Prompt-mean tables average the three
observed seed lengths before evaluating ALPS point accuracy; rollout rows remain the correct unit
for NLL and interval coverage.

`examples/first_trace.jsonl` is a small, committed Hugging Face smoke trace. It proves that the
collector, serializer, and schema validator work end to end without treating the example model as
a research result.
