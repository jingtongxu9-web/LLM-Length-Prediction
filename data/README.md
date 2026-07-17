# Data

This directory contains the small, versioned inputs and manifests required to reproduce the frozen
ALPS v1 experiment. The exact prompt text and its fixed Train/Test assignment are committed to Git.
Large generated outputs remain local and are ignored by Git.

Current layout:

```text
data/
├── prompts/
│   └── alps_v1_prompts.jsonl  # 180 versioned prompts with frozen 80/20 split
├── raw/                       # optional external source material; local only
├── interim/                   # generated Qwen rollouts before quality checks; local only
├── processed/                 # accepted trace datasets derived from the frozen split; local only
└── README.md                  # data contract and reproducibility rules
```

All matched prompts and trajectories derived from one `prompt_family_id` must remain in the same
split.

The committed Prompt Manifest is intentionally small. Do not commit generated answers, prefill
hidden states, decode entropy traces, model checkpoints, or other large experiment outputs. Those
belong under the ignored local directories and must be linked to a manifest hash when results are
reported.

## Frozen ALPS v1 prompt manifest

`prompts/alps_v1_prompts.jsonl` is the versioned pilot dataset used by the frozen ALPS v1
experiment. It contains 60 prompt families and 180 Chinese prompts: matched short, medium, and long
variants for QA, summarization, and code. The manifest contains 144 train prompts and 36 final-test
prompts. Complete families stay in one split, and each prompt declares the fixed generation seeds
`[42, 43, 44]`.

Each JSONL record contains:

- `prompt_family_id`: grouping key for matched short, medium, and long variants;
- `prompt_id`: unique prompt key;
- `task_type`: `qa`, `summarization`, or `code`;
- `intended_length`: `short`, `medium`, or `long`;
- `intended_output_tokens`: construction-time balancing range, not a training label;
- `split`: `train` or `test`;
- `generation_seeds`: the three frozen rollout seeds;
- `prompt`: the exact text sent through the Qwen chat template.

Rebuild the manifest deterministically with:

```bash
python scripts/build_prompt_manifest.py
```

The real target remains the EOS-terminated output-token count observed during each rollout. Never
move a prompt between splits after observing generated output length.
