from __future__ import annotations

import json
import random
from pathlib import Path


def main(
    mode: str = "factual_qa",
    data_dir: str = "training/data/v2",
    seed: int = 42,
) -> None:
    data_path = Path(data_dir)

    mode_file = data_path / f"{mode}.jsonl"
    general_file = data_path / "general_sft.jsonl"

    if not mode_file.exists():
        raise FileNotFoundError(f"Mode-specific data not found: {mode_file}")
    if not general_file.exists():
        raise FileNotFoundError(f"General SFT data not found: {general_file}")

    examples: list[dict] = []
    for path in (mode_file, general_file):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(examples)

    output_path = data_path / f"{mode}_combined.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    mode_count = sum(1 for ex in examples if ex.get("metadata", {}).get("mode") == mode)
    general_count = sum(1 for ex in examples if ex.get("metadata", {}).get("mode") == "general_sft")

    print("\n" + "=" * 60)
    print(f"COMBINED TRAINING DATA: {mode}")
    print("=" * 60)
    print(f"  Mode-specific:  {mode_count}")
    print(f"  General SFT:    {general_count}")
    print(f"  Total:          {len(examples)}")
    print(f"  Output:         {output_path}")
    print(f"  General ratio:  {general_count / len(examples):.1%}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
