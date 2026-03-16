from __future__ import annotations

from contextual_research_agent.retrieval.config import ContextAssemblyConfig
from contextual_research_agent.retrieval.types import ScoredCandidate


class ContextAssembler:
    """
    Assemble final context string from scored candidates.

    Deduplicates by chunk_id, orders by configured strategy.
    """

    def __init__(self, config: ContextAssemblyConfig | None = None):
        self._config = config or ContextAssemblyConfig()

    def assemble(
        self,
        candidates: list[ScoredCandidate],
        max_tokens: int | None = None,
    ) -> tuple[str, list[ScoredCandidate]]:
        """
        Assemble context string from candidates.

        Returns:
            (context_string, deduplicated_ordered_candidates)
        """
        max_tokens = max_tokens or self._config.max_tokens

        deduped = self._deduplicate(candidates)
        ordered = self._order(deduped)
        selected = self._apply_token(ordered, max_tokens)
        context = self._format(selected)

        return context, selected

    def _deduplicate(self, candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
        """Keep highest-scored candidate per chunk_id."""
        best: dict[str, ScoredCandidate] = {}
        for c in candidates:
            if c.chunk_id not in best or c.score > best[c.chunk_id].score:
                best[c.chunk_id] = c
        return list(best.values())

    def _order(self, candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
        """Order candidates by configured strategy."""
        if self._config.ordering == "score":
            return sorted(candidates, key=lambda c: c.score, reverse=True)

        if self._config.ordering == "document_then_chunk":
            doc_best_score: dict[str, float] = {}
            for c in candidates:
                doc_id = c.document_id
                if doc_id not in doc_best_score or c.score > doc_best_score[doc_id]:
                    doc_best_score[doc_id] = c.score

            return sorted(
                candidates,
                key=lambda c: (
                    -doc_best_score.get(c.document_id, 0.0),
                    c.chunk.chunk_index,
                ),
            )

        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def _apply_token(
        self,
        candidates: list[ScoredCandidate],
        max_tokens: int,
    ) -> list[ScoredCandidate]:
        selected: list[ScoredCandidate] = []
        used_tokens = 0

        for c in candidates:
            chunk_tokens = c.chunk.token_count or (len(c.chunk.text) // 4)
            if used_tokens + chunk_tokens > max_tokens and selected:
                break
            selected.append(c)
            used_tokens += chunk_tokens

        return selected

    def _format(self, candidates: list[ScoredCandidate]) -> str:
        parts: list[str] = []
        template = self._config.chunk_template

        for c in candidates:
            section_type = c.chunk.metadata.get("section_type", "unknown")
            text = template.format(
                chunk_id=c.chunk_id,
                section_type=section_type,
                score=c.score,
                text=c.chunk.text,
                document_id=c.document_id,
                section=c.chunk.section,
            )
            parts.append(text)

        return self._config.separator.join(parts)
