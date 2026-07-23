import json
from collections import Counter, defaultdict
from pathlib import Path


def test_frozen_prompt_manifest() -> None:
    path = Path("data/prompts/alps_v1_prompts.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 180
    assert len({record["prompt_id"] for record in records}) == 180
    assert Counter(record["task_type"] for record in records) == {
        "qa": 60,
        "summarization": 60,
        "code": 60,
    }
    assert Counter(record["split"] for record in records) == {"train": 144, "test": 36}

    families = defaultdict(list)
    for record in records:
        families[record["prompt_family_id"]].append(record)

    assert len(families) == 60
    for family_records in families.values():
        assert len(family_records) == 3
        assert {record["intended_length"] for record in family_records} == {
            "short",
            "medium",
            "long",
        }
        assert len({record["split"] for record in family_records}) == 1
        assert {tuple(record["generation_seeds"]) for record in family_records} == {(42, 43, 44)}
