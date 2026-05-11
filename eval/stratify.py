from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ANNOTATION_INSTRUCTIONS = """\
# Human Validation — Annotation Instructions

## Overview

You will rate 36 query-answer pairs from the evaluation set. For each pair,
assess the system's `generated_answer` against the provided `expected_answer`
and the original `query`.

All ratings use a 1-5 Likert scale, identical to the GPT-judge scale, to enable
direct inter-annotator agreement computation.

## Scoring criteria

### 1. FAITHFULNESS (groundedness in retrieved context)

How well does the generated answer stay grounded in the source material?

- **5** — All factual claims in the answer are supported by retrieved chunks.
- **4** — Almost all claims are supported; minor unsupported details.
- **3** — Main claims are supported, but some unsupported statements exist.
- **2** — Many unsupported or distorted claims.
- **1** — Answer is not grounded in retrieved chunks (hallucination).

### 2. COMPLETENESS (coverage of the question)

Does the answer fully address what was asked?

- **5** — Answer fully addresses the question.
- **4** — Answer addresses the main part of the question.
- **3** — Partial answer; misses important aspects.
- **2** — Answer touches the topic but does not address the question.
- **1** — Answer does not address the question.

### 3. FACTUAL_ACCURACY (correctness of specific facts)

Are the specific facts (numbers, names, technical details) correct?

- **5** — All facts are accurate.
- **4** — Minor errors that do not change the meaning.
- **3** — Some factual inaccuracies.
- **2** — Several serious factual errors.
- **1** — Factually incorrect answer.

### 4. is_refusal (boolean flag)

Mark `True` if the model explicitly refused to answer
(e.g., "The provided sources do not contain enough information...").
Otherwise mark `False`.

### 5. notes (optional)

Short comment if there is anything noteworthy about this query-answer pair
(1-2 lines maximum). Examples:
- "Answer is correct but uses outdated terminology"
- "Model invented a citation that does not exist"
- "Question is ambiguous; multiple valid answers possible"

## How to annotate

1. Open `annotation_form.csv` in Excel, LibreOffice, or Google Sheets.
2. For each row, read in this order:
   - `query` (the question)
   - `expected_answer` (ground truth reference)
   - `generated_answer` (system output)
3. Fill in the five `human_*` columns.
4. Save the file as `annotations_filled.csv` in the same folder.

## Estimated time

About 5-7 minutes per row. For 36 rows, plan ~3-4 hours total.
Recommended: split into two sessions of 18 rows each to avoid annotator fatigue.

## Important notes

- The GPT-judge scores (`gpt_faithfulness`, `gpt_relevance`) are shown for reference
  but should **not** influence your independent rating. Avoid anchoring bias.
- If retrieved context is not visible in the CSV, refer to `sample_queries.json`
  for the full chunk identifiers (`retrieved_chunk_ids` field).
- Skipping rows is allowed; the agreement script handles missing values gracefully.
"""


def stratify_sample(
    eval_data: list[dict],
    queries_per_category: int = 6,
    seed: int = 42,
) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in eval_data:
        cat = item.get("category", "unknown")
        by_category[cat].append(item)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for category, items in by_category.items():
        rng.shuffle(items)
        sampled.extend(items[:queries_per_category])

    return sampled


def load_baseline_results(results_path: str) -> dict[str, dict]:
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    per_query = data.get("per_query", [])
    return {item.get("query"): item for item in per_query}


def main(
    eval_set: str = "eval/peft_gold_v3_mapped.json",
    baseline_results: str = "eval/results/gen_judged_gpt5mini.json",
    output_dir: str = "eval/human_validation",
    queries_per_category: int = 6,
    seed: int = 42,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval set: {eval_set}")
    eval_data = json.loads(Path(eval_set).read_text(encoding="utf-8"))
    print(f"  Total queries: {len(eval_data)}")

    print(f"Loading baseline results: {baseline_results}")
    baseline = load_baseline_results(baseline_results)
    print(f"  Total result records: {len(baseline)}")

    print(f"\nStratifying sample ({queries_per_category} per category)...")
    sample = stratify_sample(eval_data, queries_per_category, seed)

    cat_counts: dict[str, int] = defaultdict(int)
    for item in sample:
        cat_counts[item.get("category", "unknown")] += 1
    print(f"  Sample size: {len(sample)}")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat:25s}: {count}")

    annotation_records: list[dict] = []
    missing_results = 0

    for i, item in enumerate(sample):
        query = item.get("query")
        result = baseline.get(query)

        if not result:
            missing_results += 1
            continue

        record = {
            "id": f"hv_{i + 1:03d}",
            "category": item.get("category"),
            "query": query,
            "expected_answer": item.get("expected_answer"),
            "relevant_ids": item.get("relevant_ids", []),
            "retrieved_chunk_ids": result.get("chunk_ids_used", []),
            "generated_answer": result.get("generated_answer", ""),
            "gpt_judge": {
                "faithfulness_score": result.get("gen_metrics", {}).get("faithfulness_score"),
                "relevance_score": result.get("gen_metrics", {}).get("relevance_score"),
                "semantic_similarity": result.get("gen_metrics", {}).get("semantic_similarity"),
                "is_refusal": result.get("gen_metrics", {}).get("is_refusal", False),
            },
            "human": {
                "faithfulness": None,
                "completeness": None,
                "factual_accuracy": None,
                "is_refusal": None,
                "notes": "",
            },
        }
        annotation_records.append(record)

    if missing_results:
        print(f"\nWARNING: {missing_results} queries missing in baseline results")

    json_path = output_path / "sample_queries.json"
    json_path.write_text(
        json.dumps(annotation_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nOK Full sample saved: {json_path}")

    instructions_path = output_path / "INSTRUCTIONS.md"
    instructions_path.write_text(ANNOTATION_INSTRUCTIONS, encoding="utf-8")
    print(f"OK Instructions saved: {instructions_path}")

    csv_path = output_path / "annotation_form.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "query",
                "expected_answer",
                "generated_answer",
                "gpt_faithfulness",
                "gpt_relevance",
                "human_faithfulness",
                "human_completeness",
                "human_factual_accuracy",
                "human_is_refusal",
                "human_notes",
            ],
        )
        writer.writeheader()
        for rec in annotation_records:
            writer.writerow(
                {
                    "id": rec["id"],
                    "category": rec["category"],
                    "query": rec["query"],
                    "expected_answer": rec["expected_answer"],
                    "generated_answer": rec["generated_answer"],
                    "gpt_faithfulness": rec["gpt_judge"]["faithfulness_score"],
                    "gpt_relevance": rec["gpt_judge"]["relevance_score"],
                    "human_faithfulness": "",
                    "human_completeness": "",
                    "human_factual_accuracy": "",
                    "human_is_refusal": "",
                    "human_notes": "",
                }
            )
    print(f"OK Annotation CSV saved: {csv_path}")

    print("\n" + "=" * 70)
    print("HUMAN VALIDATION SETUP COMPLETE")
    print("=" * 70)
    print(f"Open:  {csv_path}")
    print(f"Read:  {instructions_path}")
    print(
        f"\nEstimated time: ~7 min per query x {len(annotation_records)} = ~{len(annotation_records) * 7} min"
    )
    print(f"\nWhen done, save the file as: {output_path}/annotations_filled.csv")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
