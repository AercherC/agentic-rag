"""Perform conservative, structure-aware text normalization on DocumentSegments."""

import re
import unicodedata

from evidenceflow.domain.document import Document, DocumentSegment


class DocumentCleaningError(ValueError):
    """Signal that the whole Document no longer contains any usable text after cleaning."""


def _normalize_common_text(text: str) -> str:
    """Apply the normalization shared by all file types that does not alter text structure."""

    # Normalize the Unicode representation first, then unify cross-platform line endings, so that
    # the
    # later rules always face stable input.
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # A BOM is an encoding marker only at the very start of the text; U+FEFF inside the body must
    # not be
    # removed by mistake.
    if normalized.startswith("\ufeff"):
        normalized = normalized[1:]
    return normalized


def _strip_edge_blank_lines(text: str) -> str:
    """Remove purely blank lines at the start and end, while keeping the possibly meaningful indentation of the first and last content lines."""

    lines = text.split("\n")
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _clean_plain_text(text: str) -> str:
    """Conservatively clean trailing whitespace and excessive blank lines in PDF/TXT text."""

    normalized = _normalize_common_text(text)
    # Trailing spaces and tabs on plain-text lines carry no structure and can be removed safely;
    # spaces inside a line are left untouched.
    normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)
    normalized = _strip_edge_blank_lines(normalized)
    # Three newlines mean at least two consecutive blank lines; collapsing them into one blank line
    # is
    # enough to keep the paragraph boundary.
    return re.sub(r"\n{3,}", "\n\n", normalized)


def _clean_markdown_text(text: str) -> str:
    """Normalize only the Unicode, line endings, BOM, and edge blank lines of Markdown."""

    normalized = _normalize_common_text(text)
    # Trailing spaces, internal blank lines, and indentation in Markdown may all be syntactically
    # meaningful, so only the purely blank lines around the Segment are handled and the whitespace
    # collapsing rules for plain text are not reused.
    return _strip_edge_blank_lines(normalized)


def _clean_segment(segment: DocumentSegment, file_type: str) -> DocumentSegment | None:
    """Clean one Segment; return None when nothing is left after cleaning."""

    if file_type == "md":
        cleaned_text = _clean_markdown_text(segment.content)
    else:
        cleaned_text = _clean_plain_text(segment.content)

    # strip() is used only to decide whether any usable characters remain; the stripped result is
    # not
    # written back into the text.
    if not cleaned_text.strip():
        return None
    # A deep copy prevents the returned object from sharing mutable metadata with the input object.
    return segment.model_copy(update={"content": cleaned_text}, deep=True)


def clean_document(document: Document) -> Document:
    """Return a new cleaned Document, keeping the source identity and page boundaries unchanged.

    Args:
        document: The project-internal Document produced by the loader.

    Returns:
        A brand-new Document in which only the Segment text may have changed.

    Raises:
        DocumentCleaningError: No Segment has any usable text left after cleaning.
    """

    cleaned_segments: list[DocumentSegment] = []
    for segment in document.segments:
        cleaned_segment = _clean_segment(segment, document.file_type)
        if cleaned_segment is not None:
            # The original traversal order is preserved, so PDF page numbers are never reordered or
            # renumbered.
            cleaned_segments.append(cleaned_segment)

    if not cleaned_segments:
        raise DocumentCleaningError("Document contains no usable text after cleaning")

    # document_id, content_hash, and the other source fields come from the original document; only
    # the
    # Segments are replaced, and a deep copy guarantees the cleaner never mutates the loader's
    # output in place.
    return document.model_copy(update={"segments": cleaned_segments}, deep=True)
