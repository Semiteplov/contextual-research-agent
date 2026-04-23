from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CognitiveMode(str, Enum):
    FACTUAL_QA = "factual_qa"
    SUMMARIZATION = "summarization"
    CRITICAL_REVIEW = "critical_review"
    COMPARISON = "comparison"
    METHODOLOGICAL_AUDIT = "methodological_audit"
    IDEA_GENERATION = "idea_generation"

    @classmethod
    def from_intent(cls, intent: str) -> CognitiveMode:
        mapping = {
            "factual_qa": cls.FACTUAL_QA,
            "method_explanation": cls.SUMMARIZATION,
            "comparison": cls.COMPARISON,
            "critique": cls.CRITICAL_REVIEW,
            "survey": cls.SUMMARIZATION,
            "citation_trace": cls.FACTUAL_QA,
        }
        return mapping.get(intent, cls.FACTUAL_QA)


class GenerationConfig(BaseModel):
    # LLM parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=64, le=16384)

    default_mode: CognitiveMode = CognitiveMode.FACTUAL_QA
    auto_detect_mode: bool = Field(
        default=True,
        description="Automatically detect cognitive mode from retrieval intent.",
    )

    # Context handling
    max_context_chunks: int = Field(
        default=10,
        description="Maximum number of chunks to include in prompt context.",
    )
    include_chunk_metadata: bool = Field(
        default=True,
        description="Include section_type, document_id in context formatting.",
    )

    # Generation behavior
    require_citation: bool = Field(
        default=True,
        description="Instruct LLM to cite chunk IDs in the answer.",
    )
    refuse_on_no_context: bool = Field(
        default=True,
        description="Instruct LLM to refuse answering if context is insufficient.",
    )

    def to_mlflow_params(self) -> dict[str, Any]:
        return {
            "gen/temperature": self.temperature,
            "gen/max_tokens": self.max_tokens,
            "gen/default_mode": self.default_mode.value,
            "gen/auto_detect_mode": self.auto_detect_mode,
            "gen/max_context_chunks": self.max_context_chunks,
            "gen/require_citation": self.require_citation,
            "gen/refuse_on_no_context": self.refuse_on_no_context,
        }


class LLMConfig(BaseModel):
    provider: str = Field(default="ollama", description="ollama | llamacpp")
    model: str = Field(default="qwen3:8b")
    host: str = Field(default="http://localhost:11434")
    timeout: float = Field(default=300.0)

    def to_mlflow_params(self) -> dict[str, Any]:
        return {
            "llm/provider": self.provider,
            "llm/model": self.model,
            "llm/host": self.host,
        }
