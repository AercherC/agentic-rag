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
from evidenceflow.storage import ChromaVectorStore

# BGE-M3 Dense 向量维度；Exact 索引维度不符说明 collection 与当前 Embedding 模型不匹配。
EMBEDDING_DIMENSION = 1024


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
    )


def _lifecycle_governed(settings: Settings) -> bool:
    """lifecycle 状态文件存在时，线上 collection 由 active 指针决定。"""

    return Path(settings.index_state_path).is_file()


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
    """拒绝直接向 lifecycle active 指针指向的 collection 摄取。

    ACTIVE 索引只能经 staging 构建 + 校验 + activate 变更；直接 ingest 会走
    IngestionService 的“先删后写”路径，失败时留下半更新的线上索引。
    """

    if not _lifecycle_governed(settings):
        return
    active = _load_lifecycle_state(settings).active.collection_name
    if collection_name == active:
        raise ActiveIndexProtectedError(
            f"Refusing to ingest into the ACTIVE collection {active!r}; "
            "build a staging index with scripts/build_staging_index.py instead"
        )


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


def build_retriever(settings: Settings) -> DenseRerankerRetriever:
    """装配线上检索：Exact Dense Top20 → Learned Reranker → Top5。

    Dense 候选使用进程内 ``ExactDenseIndex``（numpy.exact），而不是 Chroma HNSW：
    HNSW 的近似结果在不同进程间不保证一致，会让 A/B 实验无法逐题复现；
    而 817 个向量的全量内积成本可以忽略。Chroma 仍是唯一持久化来源。
    """

    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    collection_name = resolve_active_collection(settings)
    vector_store = ChromaVectorStore(settings.vector_db_path, collection_name)
    # 由 active 指针解析出的 collection 必须带 lifecycle=ACTIVE 标记，防止指针指向
    # 未激活或被标记为 STAGING 的 collection；无状态文件的全新环境没有该标记，不校验。
    if _lifecycle_governed(settings):
        lifecycle = vector_store.metadata.get("lifecycle")
        if lifecycle != "ACTIVE":
            raise ActiveCollectionError(
                f"Collection {collection_name!r} has metadata lifecycle={lifecycle!r}, "
                "expected 'ACTIVE'"
            )
    # 启动时从 Chroma 全量读出向量并构建索引；不写出任何第二份持久化向量文件。
    pairs = list(vector_store.iter_all_chunks_with_embeddings())
    exact_index = ExactDenseIndex(
        [chunk for chunk, _ in pairs], [embedding for _, embedding in pairs]
    )
    # 启动校验：索引大小必须等于 collection 记录数，维度必须与 BGE-M3 一致，
    # 否则说明读到了不完整或不匹配的索引，拒绝启动而不是静默降级。
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
    )


def build_rag_service(
    settings: Settings,
    *,
    retriever: RetrieverProtocol | None = None,
    generator: GeneratorProtocol | None = None,
) -> RAGService:
    """创建唯一正式 Exact Dense Top20 → Learned Reranker → Top5 Pipeline。

    ``retriever`` 与 ``generator`` 仅供评测脚本注入（记录 Top5、读取 Generator.last_diagnostics、
    包装 LLM Client 做限流与原始回复留存），未提供时使用正式装配；其余组件始终由本函数按
    Settings 创建。
    """

    retriever = retriever if retriever is not None else build_retriever(settings)
    context_builder = ContextBuilder(
        BgeM3Tokenizer(settings.embedding_model),
        settings.max_context_tokens,
    )
    generator = generator if generator is not None else Generator(build_llm_client(settings))
    return RAGService(retriever, context_builder, generator)
