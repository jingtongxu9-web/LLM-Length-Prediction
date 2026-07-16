# Data

This directory stores manifests and small metadata only. Keep raw prompts, generations, hidden states, and entropy traces outside Git.

Recommended local layout:

```text
data/
├── raw/        # immutable source prompts
├── interim/    # generated traces before validation
├── processed/  # frozen train/validation/test manifests
└── README.md
```

All trajectories derived from one `prompt_id` must remain in the same split.
