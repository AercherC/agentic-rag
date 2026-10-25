"""The single composition point that assembles the one official EvidenceFlow RAG runtime."""

from pathlib import Path

from evidenceflow.config import Settings
from evidenceflow.context import ContextBuilder
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.generation import Generator
from evidenceflow.ingestion import BgeM3Tokenizer, IngestionService
from evidenceflow.knowledge import (
    ActiveCollectionError,
    ActiveIndexProtectedError,
    LifecycleState,
)
from evidenceflow.llm import OpenAICompatibleLLMClient
from evidenceflow.rag import RAGService
from evidenceflow.rag.service import GeneratorProtocol, RetrieverProtocol
from evidenceflow.retrieval import (
    DenseRerankerRetriever,
    DenseRetriever,
    ExactDenseIndex,
    ExactDenseIndexError,
    LearnedReranker,
    TransformersCrossEncoderScorer,
)
from evidenceflow.tracing import TraceRecorder
from evidenceflow.storage import (
    ChromaVectorStore,
    VectorStoreConfigurationError,
    read_collection_metadata,
)

# BGE-M3 Dense vector dimension; a mismatched Exact index dimension means the collection does not
# match the current embedding model.
EMBEDDING_DIMENSION = 1024


def build_llm_client(settings: Settings) -> OpenAICompatibleLLMClient:
    """Map Settings to explicit LLM client arguments inside the single Composition Root."""

    return OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        thinking=settings.llm_thinking,
    )


def _lifecycle_governed(settings: Settings) -> bool:
    """When the lifecycle state file exists, the production collection is decided by the active
    pointer.

    If ``INDEX_STATE_PATH`` was set explicitly (environment variable, .env or constructor argument)
    but the file does not exist, ``ActiveCollectionError`` is raised: silently falling back to
    ``Settings.collection_name`` (whose default is production v2) when an explicit path is
    mistyped or the file is lost would quietly bypass both the active pointer and the ingest
    protection.
    Only when the path was not set explicitly and no file exists at the default path is it treated
    as a brand-new environment.
    """

    path = Path(settings.index_state_path)
    if path.is_file():
        return True
    if "index_state_path" in settings.model_fields_set:
        raise ActiveCollectionError(
            f"INDEX_STATE_PATH={str(path)!r} is set explicitly but the lifecycle state file "
            "does not exist; fix the path or unset INDEX_STATE_PATH"
        )
    return False


def _load_lifecycle_state(settings: Settings) -> LifecycleState:
    """Read and validate the state file; refuse to start when it is corrupt rather than falling
    back to Settings and reading the wrong index."""

    path = Path(settings.index_state_path)
    try:
        return LifecycleState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ActiveCollectionError(
            f"Cannot read lifecycle state {path}: {type(exc).__name__}: {exc}"
        ) from exc


def resolve_active_collection(settings: Settings) -> str:
    """Decide which collection production reads; the single entry point for the runtime's
    collection choice.

    When the state file exists, ``state.active.collection_name`` wins. If the user explicitly set
    ``COLLECTION_NAME`` and it disagrees with active, ``ActiveCollectionError`` is raised instead of
    silently picking one: otherwise a stale .env after activate/rollback would quietly make
    production read the old index.
    When the state file does not exist (tests, brand-new environments) it falls back to
    ``Settings.collection_name``.
    """

    if not _lifecycle_governed(settings):
        return settings.collection_name
    active = _load_lifecycle_state(settings).active.collection_name
    # Only an "explicitly set" value counts as a conflict: a default differing from active is
    # normal (the default only serves environments without a state file).
    if "collection_name" in settings.model_fields_set and settings.collection_name != active:
        raise ActiveCollectionError(
            f"COLLECTION_NAME={settings.collection_name!r} conflicts with the lifecycle active "
            f"collection {active!r} ({settings.index_state_path}); unset COLLECTION_NAME or "
            "activate/rollback the index through IndexLifecycle"
        )
    return active


def guard_ingest_target(settings: Settings, collection_name: str) -> None:
    """Refuse to ingest directly into the ACTIVE index.

    The ACTIVE index may only change through staging build + validation + activate; a direct
    ingest would take IngestionService's "delete first, then write" path and leave a half-updated
    production index on failure. Two checks:
    1. when the state file exists, a target equal to the active pointer -> refuse;
    2. defense in depth (independent of the state file): read the target collection's metadata and
       refuse if ``lifecycle == "ACTIVE"``. This covers a missing or accidentally deleted state
       file. A non-existent collection is allowed through, so the first ingest in a brand-new
       environment works. That query loads no model and creates no collection.
    """

    if _lifecycle_governed(settings):
        active = _load_lifecycle_state(settings).active.collection_name
        if collection_name == active:
            raise ActiveIndexProtectedError(
                f"Refusing to ingest into the ACTIVE collection {active!r}; "
                "build a staging index with scripts/build_staging_index.py instead"
            )
    metadata = read_collection_metadata(settings.vector_db_path, collection_name)
    if metadata is not None and metadata.get("lifecycle") == "ACTIVE":
        raise ActiveIndexProtectedError(
            f"Refusing to ingest into collection {collection_name!r}: its metadata is "
            "lifecycle=ACTIVE; build a staging index with scripts/build_staging_index.py instead"
        )


def _open_active_store(settings: Settings, collection_name: str) -> ChromaVectorStore:
    """Open the collection resolved from the active pointer; validation completes before any model
    is loaded.

    The metadata is queried read-only first: a missing collection raises a configuration error
    (instead of leaving an empty collection behind as get_or_create would, then failing later with
    an indirect error such as "size is 0"); and a ``lifecycle`` other than ACTIVE is refused too.
    It is then opened with ``create_if_missing=False``, ensuring the read path never creates a
    collection.
    """

    metadata = read_collection_metadata(settings.vector_db_path, collection_name)
    if metadata is None:
        raise ActiveCollectionError(
            f"Lifecycle active collection {collection_name!r} does not exist in "
            f"{settings.vector_db_path}; check {settings.index_state_path} and VECTOR_DB_PATH"
        )
    lifecycle = metadata.get("lifecycle")
    if lifecycle != "ACTIVE":
        raise ActiveCollectionError(
            f"Collection {collection_name!r} has metadata lifecycle={lifecycle!r}, "
            "expected 'ACTIVE'"
        )
    try:
        return ChromaVectorStore(settings.vector_db_path, collection_name, create_if_missing=False)
    except VectorStoreConfigurationError as exc:
        raise ActiveCollectionError(
            f"Cannot open lifecycle active collection {collection_name!r}: {exc}"
        ) from exc


def build_ingestion_service(settings: Settings) -> IngestionService:
    """Create all runtime dependencies of the official ingestion flow from the unified Settings.

    Protection checks run before any model is loaded: a target that is the active collection is
    refused immediately.
    """

    collection_name = resolve_active_collection(settings)
    guard_ingest_target(settings, collection_name)
    return IngestionService(
        settings.source_docs_path,
        BgeM3Tokenizer(settings.embedding_model),
        BgeM3EmbeddingProvider(settings.embedding_model, device=settings.model_device),
        ChromaVectorStore(settings.vector_db_path, collection_name),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunk_strategy=settings.chunk_strategy,
    )


def build_retriever(
    settings: Settings, *, trace_recorder: TraceRecorder | None = None
) -> DenseRerankerRetriever:
    """Assemble production retrieval: Exact Dense Top20 -> Learned Reranker -> Top5.

    ``trace_recorder`` is optional (default None) and is passed through to
    ``DenseRerankerRetriever`` unchanged; it only observes and does not change behavior.

    Dense candidates use the in-process ``ExactDenseIndex`` (numpy.exact) rather than Chroma HNSW:
    HNSW's approximate results are not guaranteed to be identical across processes, which would
    make A/B experiments non-reproducible question by question; while the full inner-product cost
    of 817 vectors is negligible. Chroma remains the single persistent source.
    """

    collection_name = resolve_active_collection(settings)
    # Collection existence and lifecycle validation come before the embedding model is loaded: a
    # configuration error should surface immediately instead of waiting for model loading to
    # finish. A collection resolved from the active pointer must carry lifecycle=ACTIVE, so the
    # pointer cannot target an unactivated or STAGING-marked collection; a brand-new environment
    # without a state file has no such marker, so it is not validated and keeps the original open
    # order/mode (get_or_create).
    governed = _lifecycle_governed(settings)
    vector_store = _open_active_store(settings, collection_name) if governed else None
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    if vector_store is None:
        vector_store = ChromaVectorStore(settings.vector_db_path, collection_name)
    # At startup read every vector out of Chroma and build the index; no second persistent vector
    # file is ever written.
    pairs = list(vector_store.iter_all_chunks_with_embeddings())
    exact_index = ExactDenseIndex(
        [chunk for chunk, _ in pairs], [embedding for _, embedding in pairs]
    )
    # Startup validation: the index size must equal the collection record count and the dimension
    # must match BGE-M3; otherwise an incomplete or mismatched index was read, so refuse to start
    # instead of silently degrading.
    if exact_index.size != vector_store.count:
        raise ExactDenseIndexError(
            f"Exact index size {exact_index.size} does not match "
            f"collection count {vector_store.count}"
        )
    if exact_index.dimension != EMBEDDING_DIMENSION:
        raise ExactDenseIndexError(
            f"Exact index dimension {exact_index.dimension} != {EMBEDDING_DIMENSION}"
        )
    dense_retriever = DenseRetriever(
        embedding_provider,
        exact_index,
        top_k=settings.reranker_input_k,
    )
    scorer = TransformersCrossEncoderScorer(
        settings.reranker_model,
        device=settings.model_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
    )
    return DenseRerankerRetriever(
        dense_retriever,
        LearnedReranker(
            scorer,
            batch_size=settings.reranker_batch_size,
            max_candidates=settings.reranker_input_k,
        ),
        candidate_pool_k=settings.reranker_input_k,
        reranker_input_k=settings.reranker_input_k,
        final_top_k=settings.reranker_final_k,
        trace_recorder=trace_recorder,
    )


def build_rag_service(
    settings: Settings,
    *,
    retriever: RetrieverProtocol | None = None,
    generator: GeneratorProtocol | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> RAGService:
    """Create the single official Exact Dense Top20 -> Learned Reranker -> Top5 pipeline.

    ``retriever`` and ``generator`` are only for evaluation scripts to inject (to record Top5, read
    Generator.last_diagnostics, or wrap the LLM client for rate limiting and raw-reply retention);
    when not provided, the official assembly is used. Every other component is always created by
    this function from Settings.

    ``trace_recorder`` is passed to the retriever, generator and RAGService that this function
    assembles itself; a retriever / generator injected by the caller is not automatically given
    the recorder (PROJECT_SPEC §12.5).
    """

    retriever = (
        retriever
        if retriever is not None
        else build_retriever(settings, trace_recorder=trace_recorder)
    )
    context_builder = ContextBuilder(
        BgeM3Tokenizer(settings.embedding_model),
        settings.max_context_tokens,
    )
    generator = (
        generator
        if generator is not None
        else Generator(build_llm_client(settings), trace_recorder=trace_recorder)
    )
    return RAGService(retriever, context_builder, generator, trace_recorder=trace_recorder)
