from __future__ import annotations

from typing import Any


def render_routing(response: dict[str, Any]) -> str:
    if not response:
        return "_No data yet_"

    intent = response.get("intent", "—")
    complexity = response.get("complexity", "—")
    mode = response.get("resolved_mode", "—")
    chunks_count = response.get("chunks_count", 0)
    retry = response.get("retry_count", 0)

    return f"""
**Intent:** `{intent}`

**Complexity:** `{complexity}`

**Resolved Mode:** `{mode}`

**Chunks Retrieved:** {chunks_count}

**Retry Count:** {retry}

**Status:** `{response.get("status", "—")}`
"""


def render_chunks_table(response: dict[str, Any]) -> list[list[Any]]:
    chunks = response.get("chunks", [])
    if not chunks:
        return []

    rows = []
    for c in chunks:
        rows.append(
            [
                c.get("rank", 0),
                c.get("chunk_id", "")[:50],
                c.get("section_type", ""),
                c.get("document_id", "")[:25],
                round(c.get("score", 0), 4),
                c.get("text_preview", "")[:120],
            ]
        )
    return rows


def render_critic(response: dict[str, Any]) -> str:
    critic = response.get("critic")
    if not critic:
        return "_Critic was not invoked (refusal or pipeline error)_"

    verdict = critic.get("verdict", "—")
    verdict_emoji = {"pass": "✓", "fail": "✗", "partial": "~"}.get(verdict, "?")

    parts = [
        f"**Verdict:** {verdict_emoji} `{verdict.upper()}`",
        "",
    ]

    faith = critic.get("faithfulness_score")
    comp = critic.get("completeness_score")
    if faith is not None:
        parts.append(f"**Faithfulness:** {faith}/5")
    if comp is not None:
        parts.append(f"**Completeness:** {comp}/5")

    if faith is not None or comp is not None:
        parts.append("")

    reasoning = critic.get("reasoning", "").strip()
    if reasoning:
        parts.append("**Reasoning:**")
        parts.append(f"> {reasoning}")
        parts.append("")

    issues = critic.get("issues", [])
    if issues:
        parts.append("**Issues identified:**")
        for issue in issues:
            parts.append(f"- {issue}")

    return "\n".join(parts)


def render_latency(response: dict[str, Any]) -> str:
    breakdown = response.get("latency_breakdown_ms", {})
    total = response.get("total_latency_ms", 0)

    if not breakdown:
        return "_No latency data_"

    lines = ["| Node | Latency | % |", "|------|---------|---|"]
    for node, ms in breakdown.items():
        pct = (ms / total * 100) if total > 0 else 0
        lines.append(f"| {node} | {ms:.0f} ms | {pct:.1f}% |")
    lines.append(f"| **TOTAL** | **{total:.0f} ms** | **100%** |")

    tokens = response.get("tokens", {})
    if tokens:
        lines.append("")
        lines.append("**Token usage:**")
        prompt = tokens.get("prompt", 0)
        completion = tokens.get("completion", 0)
        total_t = tokens.get("total", 0)
        lines.append(f"- Prompt: {prompt}")
        lines.append(f"- Completion: {completion}")
        lines.append(f"- Total: {total_t}")

    return "\n".join(lines)


def render_events_timeline(response: dict[str, Any]) -> str:
    events = response.get("events", [])
    if not events:
        return "_No events recorded_"

    lines = ["| # | Node | Status | Latency |", "|---|------|--------|---------|"]
    for i, e in enumerate(events):
        node = e.get("node", "—")
        status = e.get("status", "—")
        latency = e.get("latency_ms", 0)
        emoji = "✗" if e.get("error") else "✓"
        lines.append(f"| {i + 1} | {emoji} `{node}` | {status} | {latency:.0f} ms |")

    return "\n".join(lines)


def render_raw_json(response: dict[str, Any]) -> dict:
    return response


def render_error(error_msg: str) -> str:
    return f"### ❌ Error\n\n```\n{error_msg}\n```"
