# Artifacts

Commit compact metrics tables and experiment manifests when useful. Store large checkpoints, traces, and generated figures in external artifact storage.

Every result should identify the config, code commit, dataset manifest hash, model revision, and seed.

`examples/first_trace.jsonl` is a small, committed Hugging Face smoke trace. It proves that the
collector, serializer, and schema validator work end to end without treating the example model as
a research result.
