from __future__ import annotations

import contextlib
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _to_int(val: str) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _to_bool(val: str) -> bool | None:
    if val is None or val == "":
        return None
    v = str(val).strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    return None


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = (sum((x - mean_x) ** 2 for x in xs)) ** 0.5
    den_y = (sum((y - mean_y) ** 2 for y in ys)) ** 0.5
    if den_x * den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    observed_agreement = sum(1 for x, y in zip(a, b) if x == y) / n

    p_a_true = sum(a) / n
    p_b_true = sum(b) / n
    expected_agreement = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)

    if expected_agreement >= 1.0:
        return float("nan")
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def interpret_kappa(k: float) -> str:
    if k != k:
        return "undefined"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main(
    annotations_csv: str = "eval/human_validation/annotation_filled.csv",
    output_dir: str = "eval/human_validation",
) -> None:
    csv_path = Path(annotations_csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        print("Save annotation_form.csv as annotation_filled.csv after annotating")
        return

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} annotations from {csv_path}")

    paired_faith: list[tuple[int, float]] = []
    per_category: dict[str, list[tuple[int, float]]] = defaultdict(list)
    notes: list[tuple[str, str]] = []

    paired_refusal: list[tuple[bool, bool]] = []

    skipped = 0

    for row in rows:
        category = row.get("category", "unknown")
        human_faith = _to_int(row.get("human_faithfulness"))
        human_refusal = _to_bool(row.get("human_is_refusal"))
        gpt_faith_raw = row.get("gpt_faithfulness", "")
        gpt_faith = None
        if gpt_faith_raw and gpt_faith_raw.lower() != "none":
            with contextlib.suppress(ValueError):
                gpt_faith = float(gpt_faith_raw)

        if human_faith is not None and gpt_faith is not None:
            paired_faith.append((human_faith, gpt_faith))
            per_category[category].append((human_faith, gpt_faith))
        else:
            skipped += 1

        if human_refusal is not None:
            gpt_refusal = gpt_faith is None
            paired_refusal.append((human_refusal, gpt_refusal))

        note = row.get("human_notes", "").strip()
        if note:
            notes.append((row.get("id", ""), note))

    if not paired_faith:
        print("ERROR: No rows with both human and GPT faithfulness scores")
        return

    human_scores = [p[0] for p in paired_faith]
    gpt_scores = [p[1] for p in paired_faith]

    pearson_r = pearson_correlation(human_scores, gpt_scores)
    mae = mean(abs(h - g) for h, g in paired_faith)

    human_pass = [h >= 4 for h in human_scores]
    gpt_pass = [g >= 4 for g in gpt_scores]
    kappa = cohens_kappa(human_pass, gpt_pass)
    kappa_interp = interpret_kappa(kappa)

    refusal_kappa = float("nan")
    if len(paired_refusal) >= 2:
        refusal_kappa = cohens_kappa(
            [r[0] for r in paired_refusal],
            [r[1] for r in paired_refusal],
        )

    cat_metrics: dict[str, dict] = {}
    for cat, pairs in per_category.items():
        if len(pairs) < 2:
            continue
        h = [p[0] for p in pairs]
        g = [p[1] for p in pairs]
        cat_metrics[cat] = {
            "n": len(pairs),
            "mean_human": mean(h),
            "mean_gpt": mean(g),
            "mae": mean(abs(hi - gi) for hi, gi in pairs),
            "pearson_r": pearson_correlation(h, g),
        }

    report = {
        "overall": {
            "n_paired": len(paired_faith),
            "n_skipped": skipped,
            "mean_human_faithfulness": mean(human_scores),
            "mean_gpt_faithfulness": mean(gpt_scores),
            "pearson_r": pearson_r,
            "cohens_kappa_pass_threshold": kappa,
            "kappa_interpretation": kappa_interp,
            "mean_absolute_error": mae,
            "exact_agreement_rate": sum(1 for h, g in paired_faith if h == g) / len(paired_faith),
            "within_one_point_rate": sum(1 for h, g in paired_faith if abs(h - g) <= 1)
            / len(paired_faith),
            "refusal_cohens_kappa": refusal_kappa,
            "refusal_interpretation": interpret_kappa(refusal_kappa),
        },
        "per_category": cat_metrics,
        "annotator_notes": [{"id": nid, "note": n} for nid, n in notes],
    }

    output_path = Path(output_dir)
    json_path = output_path / "agreement_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Human-GPT Agreement Report",
        "",
        f"**Sample size:** {len(paired_faith)} queries (paired human and GPT judge scores)",
        f"**Skipped:** {skipped} rows missing scores",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean human faithfulness | {mean(human_scores):.3f} |",
        f"| Mean GPT-judge faithfulness | {mean(gpt_scores):.3f} |",
        f"| Pearson correlation (r) | {pearson_r:.3f} |",
        f"| Cohen's kappa (>=4 threshold) | {kappa:.3f} ({kappa_interp}) |",
        f"| Mean absolute error | {mae:.3f} |",
        f"| Exact agreement rate | {sum(1 for h, g in paired_faith if h == g) / len(paired_faith):.1%} |",
        f"| Within 1 point rate | {sum(1 for h, g in paired_faith if abs(h - g) <= 1) / len(paired_faith):.1%} |",
    ]

    if refusal_kappa == refusal_kappa:
        md_lines.append(
            f"| Refusal Cohen's kappa | {refusal_kappa:.3f} ({interpret_kappa(refusal_kappa)}) |"
        )

    md_lines.extend(
        [
            "",
            "## Per-category breakdown",
            "",
            "| Category | N | Human mean | GPT mean | MAE | Pearson r |",
            "|---|---|---|---|---|---|",
        ]
    )

    for cat, m in sorted(cat_metrics.items()):
        md_lines.append(
            f"| {cat} | {m['n']} | {m['mean_human']:.2f} | {m['mean_gpt']:.2f} | "
            f"{m['mae']:.2f} | {m['pearson_r']:.3f} |"
        )

    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Cohen's kappa = {kappa:.3f} ({kappa_interp}) indicates the level of agreement",
            "between the human annotator and the GPT-judge on binary faithfulness pass/fail",
            "(threshold >=4 on the 1-5 scale).",
            "",
            f"Pearson correlation r = {pearson_r:.3f} measures linear agreement on continuous scores.",
            "",
            "Landis & Koch (1977) interpretation of Cohen's kappa:",
            "- 0.00 - 0.20: slight agreement",
            "- 0.21 - 0.40: fair agreement",
            "- 0.41 - 0.60: moderate agreement",
            "- 0.61 - 0.80: substantial agreement",
            "- 0.81 - 1.00: almost perfect agreement",
            "",
        ]
    )

    if notes:
        md_lines.append("## Selected annotator notes")
        md_lines.append("")
        for nid, note in notes[:10]:
            md_lines.append(f"- **{nid}**: {note}")

    md_path = output_path / "agreement_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("AGREEMENT METRICS")
    print("=" * 70)
    print(f"  Sample size:           {len(paired_faith)}")
    print(f"  Mean human:            {mean(human_scores):.3f}")
    print(f"  Mean GPT:              {mean(gpt_scores):.3f}")
    print(f"  Pearson r:             {pearson_r:.3f}")
    print(f"  Cohen's kappa:         {kappa:.3f} ({kappa_interp})")
    print(f"  Mean absolute error:   {mae:.3f}")
    print(
        f"  Exact agreement:       {sum(1 for h, g in paired_faith if h == g) / len(paired_faith):.1%}"
    )
    print(
        f"  Within 1 point:        {sum(1 for h, g in paired_faith if abs(h - g) <= 1) / len(paired_faith):.1%}"
    )
    if refusal_kappa == refusal_kappa:
        print(f"  Refusal kappa:         {refusal_kappa:.3f}")
    print("")
    print(f"  Full report: {json_path}")
    print(f"  Summary:     {md_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
