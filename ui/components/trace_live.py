from __future__ import annotations

from typing import Any


_NODE_LABELS = {
    "router": "Router",
    "planner": "Planner",
    "retriever": "Retriever",
    "generator": "Generator",
    "critic": "Critic",
    "synthesizer": "Synthesizer",
}

_STATUS_BADGES = {
    "completed": "✓",
    "failed": "✗",
    "skipped_already_planned": "→",
    "passthrough": "→",
    "pass_refusal": "✓ (refusal)",
}


def format_node_event(event: dict[str, Any]) -> str | None:
    event_type = event.get("type", "")

    if event_type == "ack":
        return f"⏵ Query accepted (id: `{event.get('query_id', '')[:8]}`)"

    if event_type == "node_complete":
        node = event.get("node", "?")
        status = event.get("status", "")
        latency = event.get("latency_ms", 0)
        data = event.get("data", {})

        label = _NODE_LABELS.get(node, node)

        if status.startswith("verdict_"):
            verdict = status.replace("verdict_", "")
            badge = "✓" if verdict == "pass" else ("✗" if verdict == "fail" else "~")
            badge_text = f"{badge} verdict: {verdict}"
        else:
            badge_text = _STATUS_BADGES.get(status, "✓")

        line = f"**{label}** {badge_text} — _{latency:.0f} ms_"

        details = _format_node_details(node, data)
        if details:
            line += f"<br/>&nbsp;&nbsp;{details}"

        return line

    if event_type == "node_error":
        node = event.get("node", "?")
        latency = event.get("latency_ms", 0)
        error = event.get("error", "")
        return (
            f"**{_NODE_LABELS.get(node, node)}** ✗ — _{latency:.0f} ms_<br/>&nbsp;&nbsp;_{error}_"
        )

    if event_type == "error":
        error = event.get("error", "")
        return f"❌ **Pipeline error:** {error}"

    return None


def _format_node_details(node: str, data: dict[str, Any]) -> str:
    if node == "router":
        intent = data.get("primary_intent", "")
        complexity = data.get("complexity", "")
        mode = data.get("resolved_mode", "")
        n_sub = data.get("num_sub_queries", 0)
        parts = []
        if intent:
            parts.append(f"intent=`{intent}`")
        if complexity:
            parts.append(f"complexity=`{complexity}`")
        if mode:
            parts.append(f"mode=`{mode}`")
        if n_sub:
            parts.append(f"sub_queries={n_sub}")
        return " · ".join(parts)

    if node == "retriever":
        n = data.get("num_candidates", 0)
        intent = data.get("intent", "")
        ctx_len = data.get("context_length", 0)
        parts = [f"chunks={n}"]
        if intent:
            parts.append(f"intent=`{intent}`")
        if ctx_len:
            parts.append(f"ctx={ctx_len}c")
        return " · ".join(parts)

    if node == "generator":
        retry = data.get("retry", 0)
        ans_len = data.get("answer_length", 0)
        prompt_t = data.get("prompt_tokens", 0)
        compl_t = data.get("completion_tokens", 0)
        parts = []
        if retry > 0:
            parts.append(f"⚠ retry #{retry}")
        if ans_len:
            parts.append(f"answer={ans_len}c")
        if prompt_t or compl_t:
            parts.append(f"tokens={prompt_t}+{compl_t}")
        return " · ".join(parts)

    if node == "critic":
        verdict = data.get("verdict", "")
        faith = data.get("faithfulness")
        comp = data.get("completeness")
        will_retry = data.get("will_retry", False)
        parts = [f"verdict=`{verdict}`"]
        if faith is not None:
            parts.append(f"faith={faith}")
        if comp is not None:
            parts.append(f"comp={comp}")
        if will_retry:
            parts.append("⟲ will retry")
        return " · ".join(parts)

    if node == "synthesizer":
        mode = data.get("mode", "")
        if mode:
            return f"mode=`{mode}`"

    if node == "planner":
        n = data.get("num_sub_queries", 0)
        return f"sub_queries={n}"

    return ""


def empty_trace() -> str:
    return "_Send a query to see live trace..._"
