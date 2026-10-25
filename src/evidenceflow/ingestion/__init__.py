"""Public entry point for local document ingestion."""

from evidenceflow.ingestion.cleaner import DocumentCleaningError, clean_document
from evidenceflow.ingestion.chunker import (
    ChunkingError,
    chunk_document,
    chunk_document_structured,
)
from evidenceflow.ingestion.loader import load_document
from evidenceflow.ingestion.structure import (
    StructuredParseResult,
    parse_structured_document,
)
from evidenceflow.ingestion.tokenizer import (
    BgeM3Tokenizer,
    TokenizerError,
    TokenizerLoadError,
    TokenizerProtocol,
)
from evidenceflow.ingestion.service import (
    DocumentIngestionResult,
    IngestionService,
    IngestionSummary,
    discover_source_files,
)

__all__ = [
    "BgeM3Tokenizer",
    "ChunkingError",
    "DocumentCleaningError",
    "TokenizerError",
    "TokenizerLoadError",
    "TokenizerProtocol",
    "chunk_document",
    "chunk_document_structured",
    "clean_document",
    "load_document",
    "parse_structured_document",
    "StructuredParseResult",
    "DocumentIngestionResult",
    "IngestionService",
    "IngestionSummary",
    "discover_source_files",
]
