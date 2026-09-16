"""使用 Fake 依赖验证正式增量摄取服务的状态与可靠性语义。"""

from pathlib import Path

from evidenceflow.domain.ids import build_document_id
from evidenceflow.ingestion.service import IngestionService, discover_source_files


class WordTokenizer:
    """以空格分词提供可逆 Tokenizer，隔离真实 BGE。"""

    def encode(self, text: str) -> list[int]:
        if text == "\n\n":
            return []
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"token-{value}" for value in token_ids)


class FakeEmbeddingProvider:
    """记录批量文本并可模拟失败或返回数量错误。"""

    def __init__(self, events: list[str], *, fail: bool = False, count_delta: int = 0) -> None:
        self.events = events
        self.fail = fail
        self.count_delta = count_delta
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        values = list(texts)
        self.events.append("embed")
        self.calls.append(values)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[1.0, 0.0] for _ in range(max(0, len(values) + self.count_delta))]

    def embed_query(self, query):
        return [1.0, 0.0]


class FakeVectorStore:
    """以 document_id/content_hash 集合模拟存储并记录写操作顺序。"""

    def __init__(self, events: list[str], versions: dict[str, str] | None = None, *, fail_add: bool = False) -> None:
        self.events = events
        self.versions = dict(versions or {})
        self.fail_add = fail_add
        self.added = []

    def contains_document(self, document_id, content_hash=None):
        return document_id in self.versions if content_hash is None else self.versions.get(document_id) == content_hash

    def delete_document(self, document_id):
        self.events.append("delete")
        self.versions.pop(document_id, None)

    def add_chunks(self, chunks, embeddings):
        self.events.append("store")
        if self.fail_add:
            raise RuntimeError("storage unavailable")
        self.added.append((list(chunks), list(embeddings)))
        self.versions[chunks[0].document_id] = chunks[0].content_hash

    def search(self, query_embedding, top_k):
        return []

    def clear(self):
        self.versions.clear()


def _service(root: Path, provider: FakeEmbeddingProvider, store: FakeVectorStore) -> IngestionService:
    """构造使用小窗口的测试服务。"""

    return IngestionService(root, WordTokenizer(), provider, store, chunk_size=3, chunk_overlap=1)


def test_new_document_batches_embedding_and_inserts(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "new.md"
    source.write_text("one two three four five", encoding="utf-8")
    events: list[str] = []
    provider, store = FakeEmbeddingProvider(events), FakeVectorStore(events)

    summary = _service(root, provider, store).ingest(source)

    assert (summary.inserted, summary.failed) == (1, 0)
    assert summary.chunks_written == summary.embeddings_created == 2
    assert len(provider.calls) == 1 and len(provider.calls[0]) == 2
    assert events == ["embed", "store"]


def test_unchanged_document_skips_without_embedding(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "same.txt"
    source.write_text("unchanged content", encoding="utf-8")
    # 先由 Loader 生成真实版本身份，再让 Fake Store 声明该版本已存在。
    from evidenceflow.ingestion.loader import load_document

    document = load_document(source, root)
    events: list[str] = []
    provider = FakeEmbeddingProvider(events)
    store = FakeVectorStore(events, {document.document_id: document.content_hash})

    summary = _service(root, provider, store).ingest(source)

    assert summary.skipped == 1
    assert provider.calls == [] and events == []


def test_updated_document_embeds_before_delete_and_store(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "changed.md"
    source.write_text("new document version", encoding="utf-8")
    document_id = build_document_id("changed.md")
    events: list[str] = []
    provider = FakeEmbeddingProvider(events)
    store = FakeVectorStore(events, {document_id: "old-hash"})

    summary = _service(root, provider, store).ingest(source)

    assert summary.updated == 1
    assert events == ["embed", "delete", "store"]


def test_embedding_failure_during_update_preserves_old_version(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "changed.md"
    source.write_text("new version", encoding="utf-8")
    document_id = build_document_id("changed.md")
    events: list[str] = []
    provider = FakeEmbeddingProvider(events, fail=True)
    store = FakeVectorStore(events, {document_id: "old-hash"})

    summary = _service(root, provider, store).ingest(source)

    assert summary.failed == 1 and summary.results[0].failed_stage == "embedding"
    assert events == ["embed"]
    assert store.versions[document_id] == "old-hash"


def test_discovery_filters_sorts_and_stable_root_keeps_document_id(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "nested").mkdir(parents=True)
    for relative in ("z.txt", "nested/b.md", "nested/a.PDF", "image.png", "README.json"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    discovered = discover_source_files(root, root)
    assert [path.relative_to(root).as_posix() for path in discovered] == ["nested/a.PDF", "nested/b.md", "z.txt"]

    events: list[str] = []
    directory_store = FakeVectorStore(events)
    directory_summary = _service(root, FakeEmbeddingProvider(events), directory_store).ingest(root / "nested/b.md")
    assert directory_summary.results[0].document_id == build_document_id("nested/b.md")


def test_batch_continues_after_one_file_failure(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.md").write_text("valid one", encoding="utf-8")
    (root / "b.txt").write_bytes(b"\xff")
    (root / "c.md").write_text("valid two", encoding="utf-8")
    events: list[str] = []

    summary = _service(root, FakeEmbeddingProvider(events), FakeVectorStore(events)).ingest(root)

    assert (summary.files_discovered, summary.inserted, summary.failed) == (3, 2, 1)
    assert [item.status for item in summary.results] == ["inserted", "failed", "inserted"]
    assert summary.results[1].failed_stage == "load"


def test_embedding_count_mismatch_fails_before_store(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "bad.md"
    source.write_text("one two three", encoding="utf-8")
    events: list[str] = []
    summary = _service(root, FakeEmbeddingProvider(events, count_delta=-1), FakeVectorStore(events)).ingest(source)
    assert summary.failed == 1
    assert events == ["embed"]


def test_service_integrates_with_temporary_chroma_without_production_db(tmp_path: Path) -> None:
    """正式 Chroma Adapter 必须只写入测试临时目录，并可检测第二次 unchanged。"""

    from evidenceflow.storage import ChromaVectorStore

    root = tmp_path / "source"
    root.mkdir()
    source = root / "integration.md"
    source.write_text("one two three four", encoding="utf-8")
    events: list[str] = []
    provider = FakeEmbeddingProvider(events)
    store = ChromaVectorStore(tmp_path / "isolated-chroma", "ingestion_integration")
    service = _service(root, provider, store)

    first = service.ingest(source)
    second = service.ingest(source)

    assert (first.inserted, second.skipped) == (1, 1)
    assert len(provider.calls) == 1
    assert (tmp_path / "isolated-chroma" / "chroma.sqlite3").is_file()
