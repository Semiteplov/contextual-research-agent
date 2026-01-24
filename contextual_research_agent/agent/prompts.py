from enum import StrEnum


class CognitiveMode(StrEnum):
    SUMMARIZE = "summarize"
    QA = "qa"


SYSTEM_BASE = """You are a scientific research assistant analyzing academic papers.

CRITICAL RULES:
1. Only use information from the provided context
2. Cite sources using [chunk_id] notation for every factual claim
3. If information is not in context, explicitly state "I cannot find this in the provided sources"
4. Be precise, technical, and objective
5. Use academic language appropriate for ML/AI research"""


SYSTEM_SUMMARIZE = (
    SYSTEM_BASE
    + """

YOUR TASK: Provide a structured summary of the paper.

FORMAT:
## Main Contribution
[1-2 sentences describing the key contribution]

## Methodology
[Brief overview of the approach/method]

## Key Results
[Main findings with specific numbers/metrics if available]

## Limitations
[Limitations acknowledged by authors or apparent from the text]

Remember: Every claim must have a citation [chunk_id]."""
)


SYSTEM_QA = (
    SYSTEM_BASE
    + """

YOUR TASK: Answer the user's question based ONLY on the provided context.

RULES:
- Answer directly and specifically
- Every factual claim must have a citation [chunk_id]
- If the answer is not in the context, say "I cannot find this information in the provided sources"
- If the answer is partially available, provide what you can find and note what's missing
- Be concise but complete"""
)


USER_TEMPLATE = """## Retrieved Context

{context}

---

## Question

{query}

---

Please provide your response with proper citations [chunk_id] for every claim."""


USER_SUMMARIZE_TEMPLATE = """## Retrieved Context

{context}

---

## Task

Provide a structured summary of this paper following the format specified in your instructions.
Include citations [chunk_id] for every claim."""


MODE_SYSTEM_PROMPTS = {
    CognitiveMode.SUMMARIZE: SYSTEM_SUMMARIZE,
    CognitiveMode.QA: SYSTEM_QA,
}

MODE_USER_TEMPLATES = {
    CognitiveMode.SUMMARIZE: USER_SUMMARIZE_TEMPLATE,
    CognitiveMode.QA: USER_TEMPLATE,
}


def get_system_prompt(mode: CognitiveMode) -> str:
    return MODE_SYSTEM_PROMPTS.get(mode, SYSTEM_QA)


def get_user_template(mode: CognitiveMode) -> str:
    return MODE_USER_TEMPLATES.get(mode, USER_TEMPLATE)


def format_user_prompt(
    mode: CognitiveMode,
    context: str,
    query: str = "",
) -> str:
    template = get_user_template(mode)
    return template.format(context=context, query=query)


def format_context(
    chunks: list[dict],
    max_tokens: int = 4000,
) -> str:
    parts = []
    current_tokens = 0

    for chunk in chunks:
        chunk_id = chunk.get("id", "unknown")
        text = chunk.get("text", "")
        score = chunk.get("score", 0)
        section = chunk.get("section", "")

        chunk_tokens = len(text) // 4

        if current_tokens + chunk_tokens > max_tokens:
            break

        header = f"[{chunk_id}]"
        if section:
            header += f" Section: {section}"
        header += f" (relevance: {score:.3f})"

        parts.append(f"{header}\n{text}")
        current_tokens += chunk_tokens

    return "\n\n---\n\n".join(parts)
