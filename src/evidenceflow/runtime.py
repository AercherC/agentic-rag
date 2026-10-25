"""集中装配 EvidenceFlow 唯一正式 RAG Runtime。"""

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

# BGE-M3 Dense 向量维度；仅当 Embedding Provider 未声明自身维度时作为期望值。
EMBEDDING_DIMENSION = 1024


def expected_embedding_dimension(embedding_provider: object) -> int:
    """Exact 索引必须匹配的维度：优先取 Provider 声明的模型维度。

    维度不符说明 collection 与当前 Embedding 模型不匹配。从模型读取而不是写死，
    更换 ``EMBEDDING_MODEL`` 后校验仍然针对实际加载的模型。
    """

    dimension = getattr(embedding_provider, "dimension", None)
    if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
        return dimension
    return EMBEDDING_DIMENSION


def build_llm_client(settings: Settings) -> OpenAICompatibleLLMClient:
    """在唯一 Composition Root 中把 Settings 映射为显式 LLM Client 参数。"""

    return OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        thinking=settings.llm_thinking,
        trust_env=settings.llm_trust_env,
    )


def _lifecycle_governed(settings: Settings) -> bool:
    """lifecycle 状态文件存在时，线上 collection 由 active 指针决定。

    ``INDEX_STATE_PATH`` 被显式设置（环境变量、.env 或构造参数）而文件不存在时抛
    ``ActiveCollectionError``：显式路径写错或文件丢失时若静默回落到 ``Settings.collection_name``
    （其默认值就是线上 v2），active 指针与 ingest 保护都会被悄悄绕过。
    只有未显式设置、默认路径下也没有文件时，才视为全新环境。
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
    """读取并校验状态文件；损坏时拒绝启动，不回落到 Settings 以免读到错误的索引。"""

    path = Path(settings.index_state_path)
    try:
        return LifecycleState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ActiveCollectionError(
            f"Cannot read lifecycle state {path}: {type(exc).__name__}: {exc}"
        ) from exc


def resolve_active_collection(settings: Settings) -> str:
    """决定线上读取的 collection；是 runtime 选择 collection 的唯一入口。

    状态文件存在时以 ``state.active.collection_name`` 为准。若用户显式设置了
    ``COLLECTION_NAME`` 且与 active 不一致，抛 ``ActiveCollectionError``，而不是静默
    选择其中一个：否则 activate/rollback 之后过期的 .env 会让线上悄悄读到旧索引。
    状态文件不存在（测试、全新环境）时回落到 ``Settings.collection_name``。
    """

    if not _lifecycle_governed(settings):
        return settings.collection_name
    active = _load_lifecycle_state(settings).active.collection_name
    # 只有“显式设置”才算冲突：默认值与 active 不同是正常的（默认值只服务无状态文件的环境）。
    if "collection_name" in settings.model_fields_set and settings.collection_name != active:
        raise ActiveCollectionError(
            f"COLLECTION_NAME={settings.collection_name!r} conflicts with the lifecycle active "
            f"collection {active!r} ({settings.index_state_path}); unset COLLECTION_NAME or "
            "activate/rollback the index through IndexLifecycle"
        )
    return active


def guard_ingest_target(settings: Settings, collection_name: str) -> None:
    """拒绝直接向 ACTIVE 索引摄取。

    ACTIVE 索引只能经 staging 构建 + 校验 + activate 变更；直接 ingest 会走
    IngestionService 的“先删后写”路径，失败时留下半更新的线上索引。两道检查：
    1. 状态文件存在时，目标等于 active 指针 → 拒绝；
    2. 纵深防御（与状态文件无关）：只读查询目标 collection 的 metadata，``lifecycle == "ACTIVE"``
       → 拒绝。用于状态文件缺失或被误删的情况。collection 不存在时放行，保证全新环境的首次
       ingest 可用。该查询不加载模型、不创建 collection。
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
    """打开由 active 指针解析出的 collection，校验在加载任何模型之前完成。

    先只读查询 metadata：collection 不存在时报配置错误（而不是像 get_or_create 那样留下一个空
    collection，再以“大小为 0”之类的间接错误失败）；``lifecycle`` 不是 ACTIVE 时同样拒绝。
    随后以 ``create_if_missing=False`` 打开，确保读取路径绝不创建 collection。
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
    """按统一 Settings 创建正式摄取流程的全部运行时依赖。

    保护检查先于任何模型加载：目标是 active collection 时直接拒绝。
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
    """装配线上检索：Exact Dense Top20 → Learned Reranker → Top5。

    ``trace_recorder`` 可选（默认 None），原样交给 ``DenseRerankerRetriever``，只观测不改行为。

    Dense 候选使用进程内 ``ExactDenseIndex``（numpy.exact），而不是 Chroma HNSW：
    HNSW 的近似结果在不同进程间不保证一致，会让 A/B 实验无法逐题复现；
    而 817 个向量的全量内积成本可以忽略。Chroma 仍是唯一持久化来源。
    """

    collection_name = resolve_active_collection(settings)
    # collection 的存在性与 lifecycle 校验先于 Embedding 模型加载：配置错误应立即暴露，
    # 不必先等模型加载完成。由 active 指针解析出的 collection 必须带 lifecycle=ACTIVE 标记，
    # 防止指针指向未激活或被标记为 STAGING 的 collection；无状态文件的全新环境没有该标记，
    # 不校验，且沿用原有的打开方式与顺序（get_or_create）。
    governed = _lifecycle_governed(settings)
    vector_store = _open_active_store(settings, collection_name) if governed else None
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    if vector_store is None:
        vector_store = ChromaVectorStore(settings.vector_db_path, collection_name)
    # 启动时从 Chroma 全量读出向量并构建索引；不写出任何第二份持久化向量文件。
    pairs = list(vector_store.iter_all_chunks_with_embeddings())
    exact_index = ExactDenseIndex(
        [chunk for chunk, _ in pairs], [embedding for _, embedding in pairs]
    )
    # 启动校验：索引大小必须等于 collection 记录数，维度必须与 Embedding 模型一致，
    # 否则说明读到了不完整或不匹配的索引，拒绝启动而不是静默降级。
    if exact_index.size != vector_store.count:
        raise ExactDenseIndexError(
            f"Exact index size {exact_index.size} does not match "
            f"collection count {vector_store.count}"
        )
    expected_dimension = expected_embedding_dimension(embedding_provider)
    if exact_index.dimension != expected_dimension:
        raise ExactDenseIndexError(
            f"Exact index dimension {exact_index.dimension} != embedding model "
            f"dimension {expected_dimension}"
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
    """创建唯一正式 Exact Dense Top20 → Learned Reranker → Top5 Pipeline。

    ``retriever`` 与 ``generator`` 仅供评测脚本注入（记录 Top5、读取 Generator.last_diagnostics、
    包装 LLM Client 做限流与原始回复留存），未提供时使用正式装配；其余组件始终由本函数按
    Settings 创建。

    ``trace_recorder`` 会传给本函数自己装配的 retriever、generator 与 RAGService；调用方注入的
    retriever / generator 不会被自动挂上 recorder（PROJECT_SPEC §12.5）。
    """

    retriever = (
        retriever
        if retriever is not None
        else build_retriever(settings, trace_recorder=trace_recorder)
    )
    context_builder = build_context_builder(settings)
    generator = (
        generator
        if generator is not None
        else build_generator(settings, trace_recorder=trace_recorder)
    )
    return RAGService(retriever, context_builder, generator, trace_recorder=trace_recorder)


def build_context_builder(settings: Settings) -> ContextBuilder:
    """线上 Context 装配：与 Embedding 同一 Tokenizer，预算取自 Settings。

    评测诊断脚本需要逐阶段观察检索、上下文与生成，复用本函数保证与线上装配一致。
    """

    return ContextBuilder(BgeM3Tokenizer(settings.embedding_model), settings.max_context_tokens)


def build_generator(
    settings: Settings, *, trace_recorder: TraceRecorder | None = None
) -> Generator:
    """线上 Generator 装配；与 ``build_context_builder`` 同样供逐阶段诊断复用。"""

    return Generator(build_llm_client(settings), trace_recorder=trace_recorder)
