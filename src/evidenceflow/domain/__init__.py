"""统一导出项目公共 Domain Models 与确定性 ID 工具。"""

from evidenceflow.domain.answer import RAGAnswer, SourceReference
from evidenceflow.domain.chunk import (
    DocumentChunk,
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.citation import CitationParseResult
from evidenceflow.domain.context import BuiltContext, ContextEntry
from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.domain.evaluation import EvaluationQuestion, EvaluationSource
from evidenceflow.domain.generation import AnswerStatus, GeneratedAnswer, LLMMessage
from evidenceflow.domain.ids import build_chunk_id, build_content_hash, build_document_id
from evidenceflow.domain.query import QueryRequest

__all__ = [
    "Document",
    "AnswerStatus",
    "BuiltContext",
    "ContextEntry",
    "CitationParseResult",
    "DocumentChunk",
    "DocumentSegment",
    "EvaluationQuestion",
    "EvaluationSource",
    "GeneratedAnswer",
    "LLMMessage",
    "QueryRequest",
    "RAGAnswer",
    "RerankedRetrievedChunk",
    "RetrievedChunk",
    "SourceReference",
    "build_chunk_id",
    "build_content_hash",
    "build_document_id",
]
