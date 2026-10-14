"""验证 Chroma Adapter 的持久化、映射、查询和文档版本操作。"""

import math

import chromadb
import pytest

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.storage import (
    ChromaVectorStore,
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreWriteError,
)


def _chunk(
    chunk_id: str,
    *,
    document_id: str = "DOC_A",
    content_hash: str = "HASH_1",
    content: str | None = None,
    chunk_index: int = 0,
    page_start: int | None = 1,
    page_end: int | None = 1,
    metadata: dict[str, object] | None = None,
) -> DocumentChunk:
    """构造包含 Round Trip 所需字段的确定性测试 Chunk。"""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content or f"content for {chunk_id}",
        token_count=10 + chunk_index,
        chunk_index=chunk_index,
        page_start=page_start,
        page_end=page_end,
        file_name=f"{document_id.lower()}.pdf" if page_start else "notes.md",
        content_hash=content_hash,
        metadata=metadata or {"language": "en", "nested": {"source": "test"}},
    )


def _store(tmp_path, collection_name: str = "test_collection") -> ChromaVectorStore:
    """在每个测试的临时目录中创建独立的 cosine Collection。"""

    return ChromaVectorStore(tmp_path / "chroma", collection_name)


def test_add_and_round_trip_pdf_and_markdown_chunks(tmp_path) -> None:
    """PDF 与无页码 Markdown 写入后应完整恢复 Domain 字段。"""

    store = _store(tmp_path)
    pdf_chunk = _chunk(
        "PDF_A",
        page_start=3,
        page_end=4,
        metadata={"language": "en", "nested": {"kind": "pdf"}},
    )
    markdown_chunk = _chunk(
        "MD_A",
        document_id="DOC_MD",
        content_hash="HASH_MD",
        page_start=None,
        page_end=None,
        metadata={"language": "zh", "tags": ["agent", "notes"]},
    )
    store.add_chunks([pdf_chunk, markdown_chunk], [[1, 0, 0], [0, 1, 0]])

    results = store.search([1, 0, 0], top_k=2)
    restored = {result.chunk.chunk_id: result.chunk for result in results}

    assert restored["PDF_A"] == pdf_chunk
    assert restored["MD_A"] == markdown_chunk
    assert restored["MD_A"].page_start is None
    assert restored["MD_A"].page_end is None
    # Chroma metadata 中不得写入 None 页码，而是通过字段缺失恢复为 None。
    raw = store._collection.get(ids=["MD_A"], include=["metadatas"])
    assert "page_start" not in raw["metadatas"][0]
    assert "page_end" not in raw["metadatas"][0]


def test_cosine_order_similarity_and_top_k(tmp_path) -> None:
    """查询按 cosine 排序，similarity 严格使用 1-distance 转换。"""

    store = _store(tmp_path)
    chunks = [_chunk("A"), _chunk("B", chunk_index=1), _chunk("C", chunk_index=2)]
    store.add_chunks(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    top_one = store.search([1, 0, 0], top_k=1)
    top_two = store.search([1, 0, 0], top_k=2)
    all_available = store.search([1, 0, 0], top_k=20)

    assert len(top_one) == 1
    assert top_one[0].chunk.chunk_id == "A"
    assert top_one[0].rank == 1
    assert top_one[0].similarity_score == pytest.approx(1.0)
    assert len(top_two) == 2
    assert [result.rank for result in top_two] == [1, 2]
    assert len(all_available) == 3
    assert all(
        left.similarity_score >= right.similarity_score
        for left, right in zip(all_available, all_available[1:])
    )


def test_empty_collection_and_invalid_top_k(tmp_path) -> None:
    """空 Collection 返回空结果，非法 top_k 在进入 Chroma 前失败。"""

    store = _store(tmp_path)
    assert store.search([1, 0, 0], top_k=3) == []
    for top_k in (0, -1, True):
        with pytest.raises(ValueError):
            store.search([1, 0, 0], top_k=top_k)


def test_persistence_across_adapter_instances(tmp_path) -> None:
    """同路径和 Collection 名称重新创建 Adapter 后数据仍然存在。"""

    persist_path = tmp_path / "persistent-chroma"
    first = ChromaVectorStore(persist_path, "persistent_collection")
    first.add_chunks([_chunk("PERSISTED")], [[1, 0, 0]])

    second = ChromaVectorStore(persist_path, "persistent_collection")
    results = second.search([1, 0, 0], top_k=1)

    assert results[0].chunk.chunk_id == "PERSISTED"
    assert (persist_path / "chroma.sqlite3").exists()


def test_contains_document_distinguishes_duplicate_states(tmp_path) -> None:
    """contains_document 的组合结果应区分 NEW、UNCHANGED 和 CHANGED。"""

    store = _store(tmp_path)
    assert not store.contains_document("DOC_A")  # NEW

    store.add_chunks([_chunk("A1")], [[1, 0, 0]])
    assert store.contains_document("DOC_A")
    assert store.contains_document("DOC_A", "HASH_1")  # UNCHANGED
    assert not store.contains_document("DOC_A", "HASH_2")  # CHANGED
    assert not store.contains_document("DOC_UNKNOWN")


def test_duplicate_chunk_id_is_rejected(tmp_path) -> None:
    """重复 Chunk ID 必须由 Adapter 主动拒绝，不能依赖 Chroma 静默忽略。"""

    store = _store(tmp_path)
    chunk = _chunk("DUPLICATE")
    store.add_chunks([chunk], [[1, 0, 0]])
    with pytest.raises(VectorStoreWriteError, match="already exist"):
        store.add_chunks([chunk], [[1, 0, 0]])


def test_delete_document_preserves_other_documents(tmp_path) -> None:
    """按 document_id 删除全部旧 Chunk 时不得影响其他逻辑文档。"""

    store = _store(tmp_path)
    chunks = [
        _chunk("A1", chunk_index=0),
        _chunk("A2", chunk_index=1),
        _chunk("B1", document_id="DOC_B", content_hash="HASH_B"),
    ]
    store.add_chunks(chunks, [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]])

    store.delete_document("DOC_A")

    assert not store.contains_document("DOC_A")
    assert store.contains_document("DOC_B")
    results = store.search([0, 1, 0], top_k=10)
    assert [result.chunk.chunk_id for result in results] == ["B1"]


def test_clear_recreates_usable_cosine_collection(tmp_path) -> None:
    """clear 后 Collection 为空且 Adapter 可继续写入和查询。"""

    store = _store(tmp_path)
    store.add_chunks([_chunk("BEFORE_CLEAR")], [[1, 0, 0]])

    store.clear()

    assert not store.contains_document("DOC_A")
    assert store.search([1, 0, 0], top_k=1) == []
    assert store._collection.configuration["hnsw"]["space"] == "cosine"
    store.add_chunks([_chunk("AFTER_CLEAR")], [[1, 0, 0]])
    assert store.search([1, 0, 0], top_k=1)[0].chunk.chunk_id == "AFTER_CLEAR"


def test_iter_all_chunks_returns_id_sorted_round_trip_chunks(tmp_path) -> None:
    """§17.3.1 Hybrid 集成依赖 iter_all_chunks；输出按 chunk_id 升序且字段完整。"""

    store = _store(tmp_path)
    chunks = [
        _chunk("ZZZ", chunk_index=2),
        _chunk("AAA", chunk_index=0),
        _chunk("MMM", chunk_index=1),
    ]
    store.add_chunks(chunks, [[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]])

    iterated = list(store.iter_all_chunks())

    assert [chunk.chunk_id for chunk in iterated] == ["AAA", "MMM", "ZZZ"]
    assert {chunk.chunk_id for chunk in iterated} == {chunk.chunk_id for chunk in chunks}
    assert all(isinstance(chunk, DocumentChunk) for chunk in iterated)
    # Round-trip：必须能从 Chroma Metadata 恢复完整 Domain 字段（含 chunk_index）。
    by_id = {chunk.chunk_id: chunk for chunk in iterated}
    assert by_id["AAA"].chunk_index == 0
    assert by_id["MMM"].chunk_index == 1
    assert by_id["ZZZ"].chunk_index == 2


def test_iter_all_chunks_handles_pagination_past_page_size(tmp_path) -> None:
    """当 Chunk 数量超过单页容量时仍按 chunk_id 升序完整遍历。"""

    store = _store(tmp_path)
    chunks = [
        _chunk(f"CHUNK_{index:04d}", chunk_index=index) for index in range(40)
    ]
    store.add_chunks(chunks, [[1.0 / (index + 1), 0, 0] for index in range(40)])

    iterated = list(store.iter_all_chunks())

    assert [chunk.chunk_id for chunk in iterated] == [
        chunk.chunk_id for chunk in sorted(chunks, key=lambda item: item.chunk_id)
    ]
    assert len(iterated) == len(chunks)


def test_iter_all_chunks_with_embeddings_returns_id_sorted_complete_records(tmp_path) -> None:
    """Exact Index 构建接口必须同时恢复完整 Chunk 和已存向量，且按 ID 排序。"""

    store = _store(tmp_path)
    chunks = [_chunk("Z"), _chunk("A", chunk_index=1)]
    store.add_chunks(chunks, [[0.0, 1.0], [1.0, 0.0]])

    records = list(store.iter_all_chunks_with_embeddings())

    assert [chunk.chunk_id for chunk, _ in records] == ["A", "Z"]
    assert records[0][0] == chunks[1]
    assert records[1][0] == chunks[0]
    assert all(len(vector) == 2 for _, vector in records)


def test_add_input_validation_and_metadata_serialization(tmp_path) -> None:
    """数量、维度、空向量和不可 JSON 化 metadata 应在写入前明确失败。"""

    store = _store(tmp_path)
    chunks = [_chunk("A"), _chunk("B", chunk_index=1)]
    with pytest.raises(ValueError, match="same length"):
        store.add_chunks(chunks, [[1, 0, 0]])
    with pytest.raises(ValueError, match="same dimension"):
        store.add_chunks(chunks, [[1, 0, 0], [1, 0]])
    with pytest.raises(ValueError, match="must not be empty"):
        store.add_chunks([chunks[0]], [[]])

    invalid_metadata = _chunk("INVALID", metadata={"unsupported": {1, 2}})
    with pytest.raises(VectorStoreWriteError, match="JSON serializable"):
        store.add_chunks([invalid_metadata], [[1, 0, 0]])


def test_existing_collection_dimension_is_enforced(tmp_path) -> None:
    """后续写入和 Query 必须匹配 Collection 已有向量维度。"""

    store = _store(tmp_path)
    store.add_chunks([_chunk("A")], [[1, 0, 0]])

    with pytest.raises(ValueError, match="existing collection"):
        store.add_chunks([_chunk("B", chunk_index=1)], [[1, 0]])
    with pytest.raises(ValueError, match="Query embedding dimension"):
        store.search([1, 0], top_k=1)


def test_existing_non_cosine_collection_is_rejected(tmp_path) -> None:
    """已存在的 L2 Collection 不能被 Adapter 静默当作 cosine 使用。"""

    persist_path = tmp_path / "wrong-space"
    client = chromadb.PersistentClient(path=str(persist_path))
    client.get_or_create_collection(
        name="wrong_space_collection",
        embedding_function=None,
    )

    with pytest.raises(VectorStoreConfigurationError, match="not configured"):
        ChromaVectorStore(persist_path, "wrong_space_collection")


def test_vector_store_protocol_is_satisfied(tmp_path) -> None:
    """业务层可以通过项目 VectorStore Protocol 使用 Chroma Adapter。"""

    assert isinstance(_store(tmp_path), VectorStore)
