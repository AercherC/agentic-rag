"""Verify the Phase 15A manifest, signature, source diff and rebuild decision contracts."""

from evidenceflow.knowledge.manifest import (
    PipelineIdentity,
    SourceRecord,
    build_corpus_signature,
    build_index_signature,
    build_manifest,
    compare_sources,
    decide_rebuild,
    load_manifest,
    save_manifest,
)


def _source(path: str, document_id: str, content_hash: str, size: int = 10) -> SourceRecord:
    """Build a test source; hashes use fixed placeholder values and do not depend on real files."""

    return SourceRecord(
        relative_path=path,
        document_id=document_id,
        content_hash=content_hash,
        size_bytes=size,
    )


def _pipeline(**overrides: object) -> PipelineIdentity:
    """Build a minimal indexing pipeline so the signature change can be tested field by field."""

    values: dict[str, object] = {
        "parser_version": "parser-v1",
        "cleaner_version": "cleaner-v1",
        "chunker_version": "chunker-v1",
        "chunker_config": {"strategy": "hybrid", "chunk_size": 512},
        "embedding_model": "BAAI/bge-m3",
        "embedding_version": "embedding-v1",
    }
    values.update(overrides)
    return PipelineIdentity(**values)


def _manifest(
    sources: list[SourceRecord] | None = None,
    *,
    pipeline: PipelineIdentity | None = None,
    collection_name: str = "collection-a",
) :
    """Produce a test manifest containing stable signatures."""

    actual_sources = sources or [_source("a.pdf", "doc-a", "hash-a")]
    return build_manifest(
        knowledge_base_id="kb-test",
        sources=actual_sources,
        pipeline=pipeline or _pipeline(),
        collection_name=collection_name,
        document_count=len(actual_sources),
        chunk_count=1,
    )


def test_manifest_serialization_and_source_order_are_deterministic() -> None:
    """After reordering the same input, the corpus/index signature and the canonical bytes must not change."""

    sources = [_source("b.pdf", "doc-b", "hash-b"), _source("a.pdf", "doc-a", "hash-a")]
    first = _manifest(sources)
    second = _manifest(list(reversed(sources)))

    assert first.corpus_signature == second.corpus_signature
    assert first.index.index_signature == second.index.index_signature
    assert first.canonical_bytes() == second.canonical_bytes()
    assert [item.relative_path for item in first.sources] == ["a.pdf", "b.pdf"]


def test_corpus_signature_changes_only_with_source_identity_or_content() -> None:
    """size_bytes is an audit field and must not change the corpus identity."""

    base = [_source("a.pdf", "doc-a", "hash-a", size=10)]
    assert build_corpus_signature(base) == build_corpus_signature([_source("a.pdf", "doc-a", "hash-a", size=99)])
    assert build_corpus_signature(base) != build_corpus_signature([_source("a.pdf", "doc-a", "hash-b")])
    assert build_corpus_signature(base) != build_corpus_signature([_source("renamed.pdf", "doc-a", "hash-a")])


def test_index_signature_sensitivity_and_isolation() -> None:
    """Changes in parser/chunker/embedding trigger a rebuild; retrieval/generator changes never enter the signature."""

    corpus = build_corpus_signature([_source("a.pdf", "doc-a", "hash-a")])
    baseline = build_index_signature(corpus, _pipeline())
    assert build_index_signature(corpus, _pipeline(parser_version="parser-v2")) != baseline
    assert build_index_signature(corpus, _pipeline(cleaner_version="cleaner-v2")) != baseline
    assert build_index_signature(corpus, _pipeline(chunker_version="chunker-v2")) != baseline
    assert build_index_signature(corpus, _pipeline(chunker_config={"strategy": "hybrid", "chunk_size": 1024})) != baseline
    assert build_index_signature(corpus, _pipeline(embedding_model="BAAI/other")) != baseline
    assert build_index_signature(corpus, _pipeline(embedding_version="embedding-v2")) != baseline
    # reranker, TopK and generator are not part of PipelineIdentity, so they cannot change the
    # index signature.
    assert build_index_signature(corpus, _pipeline()) == baseline


def test_source_diff_reports_five_states() -> None:
    """An explicit scan can deterministically distinguish unchanged/added/updated/deleted/renamed."""

    previous = [
        _source("same.pdf", "doc-same", "same"),
        _source("updated.pdf", "doc-updated", "old"),
        _source("deleted.pdf", "doc-deleted", "deleted"),
        _source("old-name.pdf", "doc-old", "rename"),
    ]
    current = [
        _source("same.pdf", "doc-same", "same"),
        _source("updated.pdf", "doc-updated", "new"),
        _source("new.pdf", "doc-new", "added"),
        _source("new-name.pdf", "doc-new-name", "rename"),
    ]

    diff = compare_sources(previous, current)
    assert diff.blocked is False
    assert {(item.relative_path, item.status) for item in diff.changes} == {
        ("same.pdf", "UNCHANGED"),
        ("updated.pdf", "UPDATED"),
        ("deleted.pdf", "DELETED"),
        ("new.pdf", "ADDED"),
        ("new-name.pdf", "RENAMED"),
    }
    renamed = next(item for item in diff.changes if item.status == "RENAMED")
    assert renamed.old_relative_path == "old-name.pdf"


def test_ambiguous_rename_is_blocked_without_guessing() -> None:
    """When several rename candidates share one hash, the run must stop rather than pair them arbitrarily."""

    diff = compare_sources(
        [_source("old-a.pdf", "old-a", "same"), _source("old-b.pdf", "old-b", "same")],
        [_source("new-a.pdf", "new-a", "same"), _source("new-b.pdf", "new-b", "same")],
    )

    assert diff.blocked is True
    assert "ambiguous rename" in (diff.blocked_reason or "")


def test_rebuild_decision_is_deterministic_and_rename_rebuilds() -> None:
    """The current document_id is derived from the path, so a rename must also build a new index."""

    previous = _manifest()
    unchanged = compare_sources(previous.sources, previous.sources)
    current = _manifest(collection_name="different-runtime-name")
    assert decide_rebuild(unchanged, previous_manifest=previous, current_manifest=current).rebuild_required is False

    renamed_sources = [_source("renamed.pdf", "doc-new", "hash-a")]
    renamed_manifest = _manifest(renamed_sources)
    renamed_diff = compare_sources(previous.sources, renamed_sources)
    decision = decide_rebuild(renamed_diff, previous_manifest=previous, current_manifest=renamed_manifest)
    assert decision.rebuild_required is True
    assert "source renamed" in decision.reasons


def test_manifest_round_trip_revalidates_derived_signatures(tmp_path) -> None:
    """After being persisted and reloaded, the manifest must still pass the signature check."""

    path = tmp_path / "manifest.json"
    second_path = tmp_path / "manifest-second.json"
    manifest = _manifest()
    save_manifest(path, manifest)
    save_manifest(second_path, manifest)
    restored = load_manifest(path)

    assert restored == manifest
    assert path.read_bytes() == second_path.read_bytes()
