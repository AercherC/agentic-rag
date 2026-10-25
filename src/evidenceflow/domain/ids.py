"""Generate deterministic identifiers for documents, file contents and Chunks."""

import hashlib
import json
import os
import posixpath
import unicodedata
from os import PathLike


def _sha256(data: bytes) -> str:
    """Return the fixed 64-character lowercase hexadecimal SHA-256 digest."""

    return hashlib.sha256(data).hexdigest()


def _normalize_relative_path(relative_path: str | PathLike[str]) -> str:
    """Normalize a relative path inside the ingestion root into a cross-platform stable form."""

    # NFC prevents visually identical Unicode paths encoded differently from producing different
    # IDs; POSIX separators ensure the same relative path has the same identity on Windows and
    # Unix.
    path = unicodedata.normalize("NFC", os.fspath(relative_path)).replace("\\", "/")
    # Collapse repeated separators and current-directory segments to a unique lexical path form.
    normalized = posixpath.normpath(path)
    # An empty path or the current directory cannot point at one importable concrete file.
    if normalized in {"", "."}:
        raise ValueError("relative_path must identify a file")
    # document_id may only be based on a path inside the ingestion root, so absolute paths and
    # upward escapes are forbidden.
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise ValueError("relative_path must stay within the ingestion root")
    return normalized


def build_document_id(relative_path: str | PathLike[str]) -> str:
    """Generate a stable SHA-256 document ID from the normalized relative path.

    File content does not participate in the computation, so a file updated at the same path keeps
    its original logical identity.
    """

    # Unicode and path separators must be unified before the path is encoded as UTF-8 for hashing.
    normalized_path = _normalize_relative_path(relative_path)
    return _sha256(normalized_path.encode("utf-8"))


def build_content_hash(file_bytes: bytes) -> str:
    """Hash the raw file bytes, used to detect content changes of a logical document."""

    # Only bytes are accepted, so callers cannot get unstable results from differing implicit text
    # encodings.
    if not isinstance(file_bytes, bytes):
        raise TypeError("file_bytes must be bytes")
    return _sha256(file_bytes)


def build_chunk_id(
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_index: int,
    content: str,
) -> str:
    """Generate a stable Chunk ID from the document, chunking config, position and content.

    Named fields are serialized in a fixed key order, making the input boundary explicit and the
    result reproducible.
    """

    # The Chunk ID uses the content Hash instead of the raw body, keeping the identity payload at a
    # fixed length.
    content_hash = _sha256(content.encode("utf-8"))
    # Every input that affects Chunk identity must enter the payload; changing any one yields a new
    # ID.
    identity = {
        "document_id": document_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_index": chunk_index,
        "content_hash": content_hash,
    }
    # A fixed key order plus no insignificant whitespace yields a unique byte representation with
    # no
    # separator ambiguity.
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return _sha256(serialized.encode("utf-8"))
