"""Collect one validated generation trace with Hugging Face Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_length_prediction.data.io import read_trace_jsonl, write_trace_jsonl
from llm_length_prediction.instrumentation.huggingface import HuggingFaceSignalCollector
from llm_length_prediction.runtime.model_paths import DEFAULT_REVISION, resolve_model_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Hub ID or local model path; otherwise resolve MODEL_PATH/local cache/frozen Hub ID",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-id", default="first-trace")
    parser.add_argument("--task", default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", type=int, nargs="*", default=[14])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-stride", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_source = resolve_model_source(args.model)
    collector = HuggingFaceSignalCollector(
        model_source,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        candidate_layers=args.layers,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        trace_stride=args.trace_stride,
    )
    trace = collector.collect_trace(args.prompt, prompt_id=args.prompt_id, task=args.task)
    output = write_trace_jsonl(args.output, trace)
    verified = read_trace_jsonl(output)[0]
    print(
        f"validated trace: {output} "
        f"({verified.prompt_tokens} prompt tokens, {verified.output_tokens} output tokens)"
    )


if __name__ == "__main__":
    main()
