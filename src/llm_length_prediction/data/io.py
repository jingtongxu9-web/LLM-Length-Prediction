from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_length_prediction.data.schema import GenerationTrace, TracePoint


def trace_to_dict(trace: GenerationTrace) -> dict[str, Any]:
    """Return a JSON-compatible representation after validating the trace."""

    trace.validate()
    return asdict(trace)


def trace_from_dict(payload: dict[str, Any]) -> GenerationTrace:
    """Rebuild and validate a trace loaded from JSON."""

    data = dict(payload)
    data["points"] = [TracePoint(**point) for point in data.get("points", [])]
    data["prefill_hidden_states"] = {
        int(layer): [float(value) for value in values]
        for layer, values in data.get("prefill_hidden_states", {}).items()
    }
    trace = GenerationTrace(**data)
    trace.validate()
    return trace


def write_trace_jsonl(path: str | Path, trace: GenerationTrace) -> Path:
    """Write one validated trace as a single JSONL record."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(trace_to_dict(trace), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return output


def read_trace_jsonl(path: str | Path) -> list[GenerationTrace]:
    """Read validated traces from a JSONL file."""

    traces: list[GenerationTrace] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        traces.append(trace_from_dict(payload))
    if not traces:
        raise ValueError("trace file is empty")
    return traces
