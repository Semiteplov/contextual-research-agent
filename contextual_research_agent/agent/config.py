from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from contextual_research_agent.common import logging

logger = logging.get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("../configs/agents/baseline.yaml")


@dataclass
class EmbeddingConfig:
    model: str = "Qwen/Qwen3-Embedding-4B"
    device: str | None = None
    batch_size: int = 8
    normalize: bool = True
    query_instruction: str | None = None
    passage_instruction: str | None = None


@dataclass
class ParserConfig:
    tokenizer_model: str = "Qwen/Qwen3-Embedding-4B"
    max_tokens: int = 512
    merge_peers: bool = True
    include_section_context: bool = True
    do_ocr: bool = False
    do_table_structure: bool = True
    num_threads: int = 4


@dataclass
class VectorStoreConfig:
    collection_name: str = "arxiv_papers"
    distance: str = "cosine"
    on_disk: bool = False


@dataclass
class RetrievalConfig:
    default_top_k: int = 10
    score_threshold: float | None = None


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen3:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class ModesConfig:
    available: list[str] = field(default_factory=lambda: ["summarize", "qa"])
    default: str = "qa"


@dataclass
class StorageConfig:
    bucket: str = "rag-storage"
    prefix: str = "arxiv/papers"


@dataclass
class AgentConfig:
    name: str = "baseline-v1"
    description: str = ""
    version: int = 1

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        path = Path(path)

        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with Path.open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        return cls(
            name=data.get("name", "baseline-v1"),
            description=data.get("description", ""),
            version=data.get("version", 1),
            embedding=_parse_embedding(data.get("embedding", {})),
            parser=_parse_parser(data.get("parser", {})),
            vector_store=_parse_vector_store(data.get("vector_store", {})),
            retrieval=_parse_retrieval(data.get("retrieval", {})),
            llm=_parse_llm(data.get("llm", {})),
            modes=_parse_modes(data.get("modes", {})),
            storage=_parse_storage(data.get("storage", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "embedding": self.embedding.__dict__,
            "parser": self.parser.__dict__,
            "vector_store": self.vector_store.__dict__,
            "retrieval": self.retrieval.__dict__,
            "llm": self.llm.__dict__,
            "modes": {"available": self.modes.available, "default": self.modes.default},
            "storage": self.storage.__dict__,
        }


def _parse_embedding(data: dict) -> EmbeddingConfig:
    return EmbeddingConfig(
        model=data.get("model", "Qwen/Qwen3-Embedding-4B"),
        device=data.get("device"),
        batch_size=data.get("batch_size", 8),
        normalize=data.get("normalize", True),
        query_instruction=data.get("query_instruction"),
        passage_instruction=data.get("passage_instruction"),
    )


def _parse_parser(data: dict) -> ParserConfig:
    return ParserConfig(
        tokenizer_model=data.get("tokenizer_model", "Qwen/Qwen3-Embedding-4B"),
        max_tokens=data.get("max_tokens", 512),
        merge_peers=data.get("merge_peers", True),
        include_section_context=data.get("include_section_context", True),
        do_ocr=data.get("do_ocr", False),
        do_table_structure=data.get("do_table_structure", True),
        num_threads=data.get("num_threads", 4),
    )


def _parse_vector_store(data: dict) -> VectorStoreConfig:
    return VectorStoreConfig(
        collection_name=data.get("collection_name", "arxiv_papers"),
        distance=data.get("distance", "cosine"),
        on_disk=data.get("on_disk", False),
    )


def _parse_retrieval(data: dict) -> RetrievalConfig:
    return RetrievalConfig(
        default_top_k=data.get("default_top_k", 10),
        score_threshold=data.get("score_threshold"),
    )


def _parse_llm(data: dict) -> LLMConfig:
    return LLMConfig(
        provider=data.get("provider", "ollama"),
        model=data.get("model", "qwen3:8b"),
        host=data.get("host", "http://localhost:11434"),
        temperature=data.get("temperature", 0.1),
        max_tokens=data.get("max_tokens", 2048),
    )


def _parse_modes(data: dict) -> ModesConfig:
    return ModesConfig(
        available=data.get("available", ["summarize", "qa"]),
        default=data.get("default", "qa"),
    )


def _parse_storage(data: dict) -> StorageConfig:
    return StorageConfig(
        bucket=data.get("bucket", "rag-storage"),
        prefix=data.get("prefix", "arxiv/papers"),
    )


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> AgentConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    return AgentConfig.from_yaml(config_path)


def get_config(path: str | None = None) -> AgentConfig:
    return load_config(path)


def reset_config() -> None:
    load_config.cache_clear()
