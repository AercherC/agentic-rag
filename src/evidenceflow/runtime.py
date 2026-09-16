"""集中装配 EvidenceFlow 唯一正式 RAG Runtime。"""

from evidenceflow.config import Settings
from evidenceflow.context import ContextBuilder
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.generation import Generator
from evidenceflow.ingestion import BgeM3Tokenizer, IngestionService
from evidenceflow.llm import OpenAICompatibleLLMClient
from evidenceflow.rag import RAGService
from evidenceflow.retrieval import (
    DenseRerankerRetriever,
    DenseRetriever,
    LearnedReranker,
    TransformersCrossEncoderScorer,
)
from evidenceflow.storage import ChromaVectorStore


def build_llm_client(settings: Settings) -> OpenAICompatibleLLMClient:
    """在唯一 Composition Root 中把 Settings 映射为显式 LLM Client 参数。"""

    return OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
    )


def build_ingestion_service(settings: Settings) -> IngestionService:
    """按统一 Settings 创建正式摄取流程的全部运行时依赖。"""

    return IngestionService(
        settings.source_docs_path,
        BgeM3Tokenizer(settings.embedding_model),
        BgeM3EmbeddingProvider(settings.embedding_model),
        ChromaVectorStore(settings.vector_db_path, settings.collection_name),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunk_strategy=settings.chunk_strategy,
    )


def build_rag_service(settings: Settings) -> RAGService:
    """创建唯一正式 Dense Top20 → Learned Reranker → Top5 Pipeline。"""

    embedding_provider = BgeM3EmbeddingProvider(settings.embedding_model)
    vector_store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    dense_retriever = DenseRetriever(
        embedding_provider,
        vector_store,
        top_k=settings.reranker_input_k,
    )
    scorer = TransformersCrossEncoderScorer(
        settings.reranker_model,
        device=settings.reranker_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
    )
    retriever = DenseRerankerRetriever(
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
    context_builder = ContextBuilder(
        BgeM3Tokenizer(settings.embedding_model),
        settings.max_context_tokens,
    )
    generator = Generator(build_llm_client(settings))
    return RAGService(retriever, context_builder, generator)
