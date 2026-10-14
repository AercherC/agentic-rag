"""集中装配 EvidenceFlow 唯一正式 RAG Runtime。"""

from evidenceflow.config import Settings
from evidenceflow.context import ContextBuilder
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.generation import Generator
from evidenceflow.ingestion import BgeM3Tokenizer, IngestionService
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


def build_ingestion_service(settings: Settings) -> IngestionService:
    """按统一 Settings 创建正式摄取流程的全部运行时依赖。"""

    return IngestionService(
        settings.source_docs_path,
        BgeM3Tokenizer(settings.embedding_model),
        BgeM3EmbeddingProvider(settings.embedding_model, device=settings.model_device),
        ChromaVectorStore(settings.vector_db_path, settings.collection_name),
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
    vector_store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
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
