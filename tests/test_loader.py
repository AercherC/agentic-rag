"""Test the PDF, Markdown, and TXT Document Loaders."""

from pathlib import Path

import pymupdf
import pytest

from evidenceflow.ingestion import load_document


def create_pdf(path: Path, page_texts: list[str]) -> None:
    """Dynamically create a small PDF with extractable text, so unit tests need no real material."""

    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_load_txt_as_single_unpaged_segment(tmp_path: Path) -> None:
    ingestion_root = tmp_path / "source_docs"
    source = ingestion_root / "en" / "notes.txt"
    source.parent.mkdir(parents=True)
    source.write_text("Agent state", encoding="utf-8")

    document = load_document(source, ingestion_root)

    assert document.file_type == "txt"
    assert document.source_path == "en/notes.txt"
    assert document.segments[0].content == "Agent state"
    assert document.segments[0].page_number is None
    assert len(document.segments) == 1


def test_load_markdown_preserves_original_syntax_and_uppercase_extension(
    tmp_path: Path,
) -> None:
    ingestion_root = tmp_path / "source_docs"
    source = ingestion_root / "README.MD"
    ingestion_root.mkdir()
    markdown = "# Agent\n\nUse **tools** when needed.\n"
    # Write explicit bytes to pin the line endings, so that after Windows write_text() converts
    # newlines automatically the OS behaviour is not misread as the Loader rewriting the Markdown
    # source.
    source.write_bytes(markdown.encode("utf-8"))

    document = load_document(source, ingestion_root)

    assert document.file_type == "md"
    assert document.segments[0].content == markdown
    assert document.segments[0].page_number is None


def test_load_pdf_creates_one_segment_per_text_page(tmp_path: Path) -> None:
    ingestion_root = tmp_path / "source_docs"
    ingestion_root.mkdir()
    source = ingestion_root / "agents.pdf"
    create_pdf(source, ["Agent state", "Tool calling"])

    document = load_document(source, ingestion_root)

    assert document.file_type == "pdf"
    assert [segment.page_number for segment in document.segments] == [1, 2]
    assert "Agent state" in document.segments[0].content
    assert "Tool calling" in document.segments[1].content
    assert document.metadata["page_count"] == 2


def test_pdf_empty_page_is_skipped_without_renumbering(tmp_path: Path) -> None:
    ingestion_root = tmp_path / "source_docs"
    ingestion_root.mkdir()
    source = ingestion_root / "pages.pdf"
    create_pdf(source, ["First", "", "Third"])

    document = load_document(source, ingestion_root)

    assert [segment.page_number for segment in document.segments] == [1, 3]


def test_same_path_keeps_document_id_while_content_hash_tracks_changes(
    tmp_path: Path,
) -> None:
    ingestion_root = tmp_path / "source_docs"
    ingestion_root.mkdir()
    source = ingestion_root / "notes.txt"
    source.write_text("version one", encoding="utf-8")

    first = load_document(source, ingestion_root)
    repeated = load_document(source, ingestion_root)
    source.write_text("version two", encoding="utf-8")
    updated = load_document(source, ingestion_root)

    assert first.document_id == repeated.document_id == updated.document_id
    assert first.content_hash == repeated.content_hash
    assert first.content_hash != updated.content_hash


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File does not exist"):
        load_document(tmp_path / "missing.txt", tmp_path)


def test_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="Path is not a file"):
        load_document(tmp_path, tmp_path)


def test_unsupported_file_type_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "notes.docx"
    source.write_bytes(b"not a document")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(source, tmp_path)


def test_file_outside_ingestion_root_is_rejected(tmp_path: Path) -> None:
    ingestion_root = tmp_path / "source_docs"
    ingestion_root.mkdir()
    source = tmp_path / "outside.txt"
    source.write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="outside ingestion root"):
        load_document(source, ingestion_root)


def test_pdf_without_extractable_text_is_rejected(tmp_path: Path) -> None:
    ingestion_root = tmp_path / "source_docs"
    ingestion_root.mkdir()
    source = ingestion_root / "empty.pdf"
    create_pdf(source, ["", ""])

    with pytest.raises(ValueError, match="PDF contains no extractable text"):
        load_document(source, ingestion_root)


def test_invalid_utf8_text_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff\xfe")

    with pytest.raises(UnicodeDecodeError):
        load_document(source, tmp_path)
