from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GENERAL_SYSTEM_VARIANTS = [
    "You are a helpful assistant.",
    "You are a knowledgeable AI assistant. Answer questions clearly and concisely.",
    "You are a helpful AI. Provide accurate and useful responses to user queries.",
    "You are an expert assistant. Help users with their questions and tasks.",
    "Be helpful, harmless, and honest in your responses.",
]


def _is_acceptable_response(text: str) -> bool:
    if not text or not text.strip():
        return False

    text = text.strip()

    if len(text) < 100:
        return False
    if len(text) > 1500:
        return False

    if "|---" in text or "|--|" in text:
        return False

    numbered_items = sum(
        1 for line in text.split("\n") if line.strip().startswith(("1.", "2.", "3."))
    )
    if numbered_items >= 5:
        return False

    non_ascii = sum(1 for c in text if ord(c) > 127)
    return not non_ascii / max(len(text), 1) > 0.3


def _is_acceptable_prompt(text: str) -> bool:
    if not text or not text.strip():
        return False

    text = text.strip()
    return not (len(text) < 10 or len(text) > 800)


def load_oasst_examples(
    n_examples: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The `datasets` package is required.\nInstall: pip install datasets"
        ) from e

    logger.info("Loading OASST2 dataset (this downloads ~30 MB on first run)...")
    ds = load_dataset("OpenAssistant/oasst2", split="train")

    rng = random.Random(seed)

    children_by_parent: dict[str | None, list[dict]] = {}
    rows_by_id: dict[str, dict] = {}

    for row in ds:
        rows_by_id[row["message_id"]] = row
        children_by_parent.setdefault(row["parent_id"], []).append(row)

    examples: list[dict[str, Any]] = []

    for row in ds:
        if row["role"] != "prompter" or row["parent_id"] is not None:
            continue
        if row["lang"] != "en":
            continue
        if row.get("review_count", 0) < 2:
            continue
        if row.get("deleted", False):
            continue

        user_text = row["text"]
        if not _is_acceptable_prompt(user_text):
            continue

        responses = children_by_parent.get(row["message_id"], [])
        responses = [
            r
            for r in responses
            if r["role"] == "assistant" and r["lang"] == "en" and not r.get("deleted", False)
        ]

        if not responses:
            continue

        responses.sort(key=lambda r: r.get("rank", 999) or 999)
        best = responses[0]

        assistant_text = best["text"]
        if not _is_acceptable_response(assistant_text):
            continue

        examples.append(
            {
                "user": user_text,
                "assistant": assistant_text,
            }
        )

    logger.info("Found %d acceptable single-turn pairs from OASST", len(examples))

    if len(examples) < n_examples:
        logger.warning(
            "Only %d acceptable examples available, returning all",
            len(examples),
        )
        n_examples = len(examples)

    rng.shuffle(examples)
    selected = examples[:n_examples]

    training_examples: list[dict[str, Any]] = []
    for ex in selected:
        system_prompt = rng.choice(_GENERAL_SYSTEM_VARIANTS)
        training_examples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]},
                ],
                "metadata": {
                    "mode": "general_sft",
                    "source": "oasst2",
                },
            }
        )

    return training_examples


def main(
    output: str = "training/data/v2/general_sft.jsonl",
    n_examples: int = 50,
    seed: int = 42,
) -> None:
    """Generate general SFT examples from OASST2."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    examples = load_oasst_examples(n_examples=n_examples, seed=seed)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"General SFT data: {len(examples)} examples")
    print(f"Saved to: {output_path}")
    print("=" * 60)
    print("\nSample example:")
    print(f"  User:      {examples[0]['messages'][1]['content'][:120]}...")
    print(f"  Assistant: {examples[0]['messages'][2]['content'][:120]}...")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
