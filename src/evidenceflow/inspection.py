"""Pure formatting helpers shared by the developer inspection tools.

This module only holds side-effect-free display/filter helpers: it loads no models, does
no IO, and keeps no global state. It is imported by the Inspector scripts under
`experiments/debug/` and by their unit tests, so it lives on the src side to avoid a
reverse tests -> experiments dependency.

The module takes no part in Runtime retrieval, Context building, or the generation chain;
removing it would not change any answer.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def content_preview(content: str, preview_chars: int = 500, *, full: bool = False) -> str:
    """Flatten surrounding whitespace and truncate by characters, so a full C512 Chunk does not flood the output by default."""

    if preview_chars <= 0:
        raise ValueError("preview_chars must be positive")
    normalized = content.strip()
    if full or len(normalized) <= preview_chars:
        return normalized
    return normalized[:preview_chars] + "…"


def format_page(page_start: int | None, page_end: int | None) -> str:
    """Render page-less text and single-page or cross-page PDFs as a compact, human-readable form."""

    if page_start is None:
        return "None"
    return str(page_start) if page_start == page_end else f"{page_start}-{page_end}"


def format_retrieval_result(result: Any, *, preview_chars: int = 500, full: bool = False) -> str:
    """Format a single RetrievedChunk, keeping the cosine score and the source locator fields."""

    chunk = result.chunk
    return "\n".join(
        [
            f"Rank: {result.rank}",
            f"Cosine: {result.similarity_score:.6f}",
            f"File: {chunk.file_name}",
            f"Page: {format_page(chunk.page_start, chunk.page_end)}",
            f"Chunk index: {chunk.chunk_index}",
            f"Chunk ID: {chunk.chunk_id}",
            f"Document ID: {chunk.document_id}",
            f"Content: {content_preview(chunk.content, preview_chars, full=full)}",
        ]
    )


def select_records(
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    file_name: str | None = None,
) -> list[dict[str, Any]]:
    """Stably pair the raw Chroma fields, filter by file, and apply the result limit."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("stored record fields are misaligned")
    records = [
        {"chunk_id": chunk_id, "content": content, "metadata": dict(metadata)}
        for chunk_id, content, metadata in zip(ids, documents, metadatas)
        if file_name is None or metadata.get("file_name") == file_name
    ]
    # Chroma get() makes no business-ordering promise; sorting by the stable chunk_id makes
    # repeated checks and tests easier.
    return sorted(records, key=lambda item: item["chunk_id"])[:limit]


def format_storage_record(record: Mapping[str, Any], *, preview_chars: int = 500, full: bool = False) -> str:
    """Show content and source Metadata; the full Embedding is not printed by default."""

    metadata = record["metadata"]
    domain_metadata = metadata.get("domain_metadata_json", "{}")
    try:
        domain_metadata = json.loads(domain_metadata)
    except (TypeError, json.JSONDecodeError):
        domain_metadata = {"raw": domain_metadata}
    return "\n".join(
        [
            f"Chunk ID: {record['chunk_id']}",
            f"Document ID: {metadata.get('document_id')}",
            f"File: {metadata.get('file_name')}",
            f"Chunk index: {metadata.get('chunk_index')}",
            f"Page: {format_page(metadata.get('page_start'), metadata.get('page_end'))}",
            f"Token count: {metadata.get('token_count')}",
            f"Metadata: {json.dumps(domain_metadata, ensure_ascii=False, sort_keys=True)}",
            f"Content: {content_preview(record['content'], preview_chars, full=full)}",
        ]
    )
