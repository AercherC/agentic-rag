"""Split a Clean Document into DocumentChunks using a configurable token sliding window.

Phase 13F adds hybrid chunking (§9.3): ``chunk_document_structured`` consumes the structured
element stream and splits it with the PH1 frozen semantics (standalone elements form their own
chunk, body text is aggregated per section, window overlap is zero). The fixed-window path is
unchanged and remains the rollback route for ``chunk_strategy="fixed"``.
"""

from copy import deepcopy
from typing import Any, Sequence

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.document import Document
from evidenceflow.domain.ids import build_chunk_id
from evidenceflow.ingestion.tokenizer import TokenizerProtocol

SEGMENT_BOUNDARY = "\n\n"

# Standalone element types of the hybrid strategy: each becomes its own Chunk and is removed
# from the body stream (13C-C PC2 / 13E PH1 semantics).
HYBRID_STANDALONE_TYPES = {"table", "table_caption", "figure_caption", "equation"}
# The window overlap frozen by the hybrid strategy is 0 (13E PH1 = hybrid_section_o0).
HYBRID_OVERLAP_TOKENS = 0


class ChunkingError(ValueError):
    """Indicates that the chunk configuration, the input data or the tokenizer output cannot produce valid Chunks."""


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Verify that the sliding window can advance with a positive step."""

    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be smaller than chunk_size")


def _encode_checked(
    tokenizer: TokenizerProtocol,
    text: str,
    *,
    source_name: str,
    allow_empty: bool = False,
) -> list[int]:
    """Call the tokenizer and verify that the return value satisfies the internal token-stream contract."""

    try:
        token_ids = tokenizer.encode(text)
    except Exception as exc:
        raise ChunkingError(f"Tokenizer failed to encode {source_name}") from exc
    if not isinstance(token_ids, list) or not all(
        type(token_id) is int for token_id in token_ids
    ):
        raise ChunkingError(f"Tokenizer returned invalid token IDs for {source_name}")
    if not token_ids and not allow_empty:
        raise ChunkingError(f"Tokenizer returned no token IDs for {source_name}")
    return token_ids


def _build_token_stream(
    document: Document,
    tokenizer: TokenizerProtocol,
) -> tuple[list[int], list[int | None]]:
    """Build the continuous token stream and keep the real source page number for every token."""

    if not document.segments:
        raise ChunkingError("Document contains no segments to chunk")

    token_ids: list[int] = []
    page_numbers: list[int | None] = []
    boundary_ids: list[int] | None = None

    for segment_index, segment in enumerate(document.segments):
        if segment_index > 0:
            if boundary_ids is None:
                # Some SentencePiece tokenizers ignore a whitespace-only separator and return an
                # empty list;
                # that is not an anomaly, and the following segment's own word-start marker still
                # prevents
                # words from being glued together.
                boundary_ids = _encode_checked(
                    tokenizer,
                    SEGMENT_BOUNDARY,
                    source_name="segment boundary",
                    allow_empty=True,
                )
            token_ids.extend(boundary_ids)
            # The artificial separator belongs to no real page; None is ignored when computing
            # chunk pages.
            page_numbers.extend([None] * len(boundary_ids))

        segment_ids = _encode_checked(
            tokenizer,
            segment.content,
            source_name=f"segment {segment_index}",
        )
        token_ids.extend(segment_ids)
        page_numbers.extend([segment.page_number] * len(segment_ids))

    if len(token_ids) != len(page_numbers):
        raise ChunkingError("Token IDs and page sources are not aligned")
    return token_ids, page_numbers


def _decode_checked(tokenizer: TokenizerProtocol, token_ids: list[int]) -> str:
    """Decode one window and reject results that cannot form usable body text."""

    try:
        decoded_text = tokenizer.decode(token_ids)
    except Exception as exc:
        raise ChunkingError("Tokenizer failed to decode chunk token IDs") from exc
    if not isinstance(decoded_text, str):
        raise ChunkingError("Tokenizer decode returned non-text output")

    # Only trim the whitespace around the window; do not re-run the Cleaner or alter the internal
    # Markdown structure.
    decoded_text = decoded_text.strip()
    if not decoded_text:
        raise ChunkingError("Decoded chunk contains no usable text")
    return decoded_text


def chunk_document(
    document: Document,
    tokenizer: TokenizerProtocol,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Produce sliding-window Chunks with stable IDs and page ranges, based on body token counts."""

    _validate_chunk_parameters(chunk_size, chunk_overlap)
    token_ids, page_numbers = _build_token_stream(document, tokenizer)
    step = chunk_size - chunk_overlap
    chunks: list[DocumentChunk] = []
    start = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_token_ids = token_ids[start:end]
        chunk_page_numbers = page_numbers[start:end]
        content = _decode_checked(tokenizer, chunk_token_ids)

        # None page numbers of boundary tokens do not take part in the range computation,
        # preserving
        # the real page gaps of the PDF.
        real_pages = [page for page in chunk_page_numbers if page is not None]
        page_start = min(real_pages) if real_pages else None
        page_end = max(real_pages) if real_pages else None
        chunk_index = len(chunks)

        # The Chunk ID must be built from the final decoded text, so repeated runs stay stable and
        # traceable.
        chunk_id = build_chunk_id(
            document_id=document.document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_index=chunk_index,
            content=content,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                content=content,
                token_count=len(chunk_token_ids),
                chunk_index=chunk_index,
                page_start=page_start,
                page_end=page_end,
                file_name=document.file_name,
                content_hash=document.content_hash,
                metadata=deepcopy(document.metadata),
            )
        )

        # Stop as soon as the current window covers the end, avoiding a tail chunk made almost
        # entirely
        # of overlap.
        if end == len(token_ids):
            break
        start += step

    if not chunks:
        raise ChunkingError("Document produced no chunks")
    return chunks


# --------------------------------------------------------------------------------------
# Hybrid chunking (Phase 13F, §9.3): element-level packing, semantics ported from 13E PH1
# (hybrid_section_o0).
# --------------------------------------------------------------------------------------


def _count_tokens(tokenizer: TokenizerProtocol, text: str) -> int:
    """Count the final concatenated text exactly; empty text counts 0 (consistent with the experiment convention)."""

    if not text:
        return 0
    token_ids = _encode_checked(tokenizer, text, source_name="hybrid chunk text", allow_empty=True)
    return len(token_ids)


def _split_oversized_element(
    text: str, tokenizer: TokenizerProtocol, target: int, overlap: int
) -> list[str]:
    """Split one oversized element into word windows of at most `target` tokens, with adjacent windows overlapping approximately by words.

    Ported from 14A-C ``_split_oversized_unit``: the overlap is approximated by word count,
    stepping back from the tail of the previous window until the token sum is <= overlap (at least
    0 words are stepped back); with overlap=0 the tail keeps exactly 1 word.
    """

    words = text.split()
    if not words:
        return [text]
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = _count_tokens(tokenizer, word + " ")
        if current and current_tokens + word_tokens > target:
            windows.append(" ".join(current))
            # Compute the overlap tail: step words back from the tail while the token sum <=
            # overlap.
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                prev_tokens = _count_tokens(tokenizer, prev + " ")
                if tail and tail_tokens + prev_tokens > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += prev_tokens
            current = tail + [word]
            current_tokens = _count_tokens(tokenizer, " ".join(current))
        else:
            current.append(word)
            current_tokens += word_tokens
    if current:
        windows.append(" ".join(current))
    return windows


def _pack_element_windows(
    units: Sequence[tuple[str, list[str], dict[str, Any]]],
    tokenizer: TokenizerProtocol,
    target: int,
    overlap: int,
    separator: str = SEGMENT_BOUNDARY,
) -> list[dict[str, Any]]:
    """Element-unit stream -> packed windows (ported from 13C-C ``pack_units_sep``, semantics kept identical).

    - unit token count > target -> word-level split (owners copied, split=True);
    - sequential greedy packing; when a window closes, the tail units are carried only when overlap>0;
    - the hard budget wins over the overlap: if the tail plus the next unit exceeds target, the tail
      is dropped.
    """

    flat: list[tuple[str, list[str], dict[str, Any], bool]] = []
    for text, owners, element in units:
        if not text.strip():
            continue
        token_count = _count_tokens(tokenizer, text)
        if token_count > target:
            # An oversized unit is split into pseudo units (owners copied; the word-level overlap
            # is an
            # approximation, recorded as split=True).
            for piece in _split_oversized_element(text, tokenizer, target, overlap):
                flat.append((piece, owners, element, True))
        else:
            flat.append((text, owners, element, False))

    def _joined(items: list[tuple[str, list[str], dict[str, Any], bool]]) -> str:
        return separator.join(item[0] for item in items)

    windows: list[dict[str, Any]] = []
    current: list[tuple[str, list[str], dict[str, Any], bool]] = []
    for unit in flat:
        text, owners, element, was_split = unit
        if current and _count_tokens(tokenizer, _joined(current) + separator + text) > target:
            # Close the current window; the overlap tail is computed only when overlap>0.
            tail: list[tuple[str, list[str], dict[str, Any], bool]] = []
            if overlap > 0:
                for prev in reversed(current):
                    candidate = [prev] + tail
                    if _count_tokens(tokenizer, _joined(candidate)) > overlap:
                        break
                    tail.insert(0, prev)
                # The hard budget wins over the overlap: if the tail plus the next unit exceeds
                # target, the
                # tail is dropped.
                if tail and _count_tokens(tokenizer, _joined(tail) + separator + text) > target:
                    tail = []
            windows.append(_make_hybrid_window(current, separator))
            current = list(tail)
        current.append(unit)
    if current:
        windows.append(_make_hybrid_window(current, separator))
    return windows


def _make_hybrid_window(
    items: list[tuple[str, list[str], dict[str, Any], bool]], separator: str
) -> dict[str, Any]:
    """Build a window from a unit sequence: text = the separator-joined units, owners deduplicated in order, split True if any unit was split."""

    text = separator.join(item[0] for item in items)
    owners: list[str] = []
    for _, element_ids, _, _ in items:
        for element_id in element_ids:
            if element_id not in owners:
                owners.append(element_id)
    owner_elements = [item[2] for item in items]
    return {
        "text": text,
        "element_ids": owners,
        "elements": owner_elements,
        "split": any(item[3] for item in items),
    }


def _hybrid_page_range(elements: Sequence[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Min/max of the owner elements' real page numbers (None when there is no page number; nothing is guessed)."""

    pages = [element["page"] for element in elements if isinstance(element.get("page"), int)]
    return (min(pages), max(pages)) if pages else (None, None)


def _hybrid_section_key(element: dict[str, Any]) -> tuple[str, ...]:
    """Section grouping key (the section_path list becomes a tuple; missing becomes an empty tuple)."""

    path = element.get("section_path")
    return tuple(path) if isinstance(path, list) else ()


def chunk_document_structured(
    document: Document,
    elements: Sequence[dict[str, Any]],
    tokenizer: TokenizerProtocol,
    chunk_size: int,
) -> list[DocumentChunk]:
    """Split the structured element stream with the PH1 frozen semantics (hybrid strategy, §9.3).

    - standalone types (table / table_caption / figure_caption / equation) become their own Chunk
      and are removed from the body stream, placed at the front of the document's Chunk sequence
      (element reading order);
    - body text is grouped by consecutive identical section_path, group boundaries force a break,
      and each group is greedily packed in order;
    - window overlap is frozen at 0 (``HYBRID_OVERLAP_TOKENS``), and groups are naturally separated
      by the section boundary;
    - hard budget: every Chunk has at most chunk_size tokens;
    - ``chunk_id`` goes through §10 ``build_chunk_id`` with an overlap payload of 0 (the hybrid
      frozen semantics);
    - Chunk metadata records ``chunk_strategy="hybrid"``, ``element_ids`` provenance and
      ``heading_path`` (which may be empty for body groups).
    """

    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be greater than 0")
    if not elements:
        raise ChunkingError("Document contains no structured elements to chunk")

    overlap = HYBRID_OVERLAP_TOKENS
    # First split the element stream into standalone and body units (keeping element reading
    # order).
    standalone_units: list[tuple[str, list[str], dict[str, Any]]] = []
    body_units: list[tuple[str, list[str], dict[str, Any]]] = []
    for element in elements:
        text = str(element.get("text", ""))
        element_id = element.get("element_id")
        if not text.strip() or not element_id:
            # Elements with no text (such as figure carriers) do not enter the index; an element
            # without an
            # ID cannot be traced, so it is rejected outright.
            if element_id is None:
                raise ChunkingError("Structured element is missing element_id")
            continue
        unit = (text, [str(element_id)], element)
        if element.get("element_type") in HYBRID_STANDALONE_TYPES:
            standalone_units.append(unit)
        else:
            body_units.append(unit)

    raw_windows: list[dict[str, Any]] = []
    # Standalone elements are packed one by one (an oversized one is split into several windows at
    # word level).
    for unit in standalone_units:
        raw_windows.extend(_pack_element_windows([unit], tokenizer, chunk_size, overlap))
    # Body text is grouped by consecutive section_path with forced breaks at group boundaries, and
    # each group is greedily packed.
    body_groups: list[list[tuple[str, list[str], dict[str, Any]]]] = []
    previous_key: tuple[str, ...] | None = None
    for unit in body_units:
        key = _hybrid_section_key(unit[2])
        if previous_key is None or key != previous_key:
            body_groups.append([])
            previous_key = key
        body_groups[-1].append(unit)
    for group in body_groups:
        section_path = _hybrid_section_key(group[0][2])
        heading_path = " > ".join(section_path) if section_path else None
        for window in _pack_element_windows(group, tokenizer, chunk_size, overlap):
            window["heading_path"] = heading_path
            raw_windows.append(window)

    if not raw_windows:
        raise ChunkingError("Document produced no hybrid chunks")

    # Build the DocumentChunks uniformly: chunk_index is contiguous from 0; the page range is the
    # owner elements' real min/max.
    chunks: list[DocumentChunk] = []
    for chunk_index, window in enumerate(raw_windows):
        page_start, page_end = _hybrid_page_range(window["elements"])
        metadata = deepcopy(document.metadata)
        metadata["chunk_strategy"] = "hybrid"
        metadata["element_ids"] = list(window["element_ids"])
        if window.get("heading_path") is not None:
            metadata["heading_path"] = window["heading_path"]
        chunks.append(
            DocumentChunk(
                chunk_id=build_chunk_id(
                    document_id=document.document_id,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    chunk_index=chunk_index,
                    content=window["text"],
                ),
                document_id=document.document_id,
                content=window["text"],
                token_count=_count_tokens(tokenizer, window["text"]),
                chunk_index=chunk_index,
                page_start=page_start,
                page_end=page_end,
                file_name=document.file_name,
                content_hash=document.content_hash,
                metadata=metadata,
            )
        )
    return chunks
