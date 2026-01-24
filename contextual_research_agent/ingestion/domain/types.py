from enum import StrEnum


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


class QueryStatus(StrEnum):
    PENDING = "pending"
    ROUTING = "routing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
