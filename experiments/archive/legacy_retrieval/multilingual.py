"""Phase 15B.5 归档的历史多语检索链路。"""

from collections.abc import Sequence

from evidenceflow.config import (
    DEFAULT_MULTILINGUAL_CANDIDATE_POOL_K,
    DEFAULT_MULTILINGUAL_CANDIDATE_TOP_K,
    DEFAULT_MULTILINGUAL_FINAL_TOP_K,
    DEFAULT_MULTILINGUAL_ORIGINAL_WEIGHT,
    DEFAULT_MULTILINGUAL_RRF_K,
    DEFAULT_MULTILINGUAL_TRANSLATION_WEIGHT,
)
from evidenceflow.domain.chunk import MultilingualRetrievedChunk, RetrievedChunk
# Equal RRF 与 Rank-only candidate 已迁至 retrieval 包内，避开 Runtime 反向依赖
# evidenceflow.eval.hybrid。eval.hybrid 当前仅作为 re-export 兼容层。
from evidenceflow.retrieval.hybrid_fusion import (
    HybridRRFChunk,
    RankedChunkCandidate,
    equal_ranked_rrf_fusion,
)
# BM25Index 是 Runtime 路径的 production 实现，直接从 retrieval 包内导入以避免
# 通过 eval.lexical 间接构造循环依赖。eval.lexical 本身已经 re-export 该模块。
from evidenceflow.retrieval.bm25 import BM25Index, BM25RetrievedChunk
from evidenceflow.retrieval.fusion import weighted_rrf_fusion
from evidenceflow.retrieval.retriever import DenseRetriever, InvalidQueryError
from evidenceflow.translation.base import TranslationError, TranslationProvider


# Hybrid 末轮 Equal RRF 的固定候选上限；与离线 §17.3.1 实验保持同一预算，避免
# Runtime 与离线出现"看似同样算法但候选预算不同"的隐性漂移。
HYBRID_CANDIDATE_BUDGET = 20

# Hybrid 末轮默认 Top-K：当 caller 未显式指定 top_k 时，沿用 multilingual_final_top_k
# 作为上游 Weighted RRF 的最终深度；BM25 末轮按候选预算截断候选集合，再产出
# multilingual_final_top_k。
HYBRID_DEFAULT_FINAL_TOP_K = DEFAULT_MULTILINGUAL_FINAL_TOP_K

# 新实验必须据此区分修复后的候选合同与历史提前截断行为。
RETRIEVAL_CONTRACT_VERSION = 2


def _validate_top_k(value: int, *, field_name: str) -> None:
    """保证候选或最终 Top-K 为正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_hybrid_rrf_k(value: int) -> None:
    """Hybrid 末轮 RRF k 必须为正整数，否则无法稳定融合 BM25 与多语分支。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("hybrid_rrf_k must be a positive integer")


def _to_multilingual_retrieved_chunks(
    fused: Sequence[HybridRRFChunk],
    first_stage_by_chunk_id: dict[str, MultilingualRetrievedChunk],
    bm25_by_chunk_id: dict[str, BM25RetrievedChunk],
    candidate_budget: int,
) -> list[MultilingualRetrievedChunk]:
    """把 Hybrid 末轮结果包装回 MultilingualRetrievedChunk，保持 RAGService 契约。

    - ``rank`` 与 ``fusion_score`` 来自 Hybrid 末轮 Equal RRF；
    - ``original_rank`` / ``translated_rank`` 反查第一阶段 Weighted RRF 输出，保留
      多语两路分支的可审计证据；
    - ``bm25_rank`` / ``bm25_similarity_score`` 反查 BM25 hits，BM25-only chunk
      多语两路证据留 None，但 BM25 分支必须非空才能满足 §17.3.1 之后的"三路分支任一
      非空"模型校验。
    """

    del candidate_budget  # 当前实现不裁剪候选，等价 hybrid 候选预算等于输出预算。
    results: list[MultilingualRetrievedChunk] = []
    for hybrid in fused:
        first_stage = first_stage_by_chunk_id.get(hybrid.chunk.chunk_id)
        bm25_hit = bm25_by_chunk_id.get(hybrid.chunk.chunk_id)
        results.append(
            MultilingualRetrievedChunk(
                chunk=hybrid.chunk,
                rank=hybrid.rank,
                fusion_score=hybrid.fusion_score,
                original_rank=first_stage.original_rank if first_stage else None,
                translated_rank=first_stage.translated_rank if first_stage else None,
                original_similarity_score=(
                    first_stage.original_similarity_score if first_stage else None
                ),
                translated_similarity_score=(
                    first_stage.translated_similarity_score if first_stage else None
                ),
                bm25_rank=bm25_hit.rank if bm25_hit else None,
                bm25_similarity_score=(
                    bm25_hit.bm25_score if bm25_hit else None
                ),
            )
        )
    return results


class MultilingualRetriever:
    """使用 Translation 增强 + 可选 BM25 末轮 Equal RRF 返回统一多语检索结果。"""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        translation_provider: TranslationProvider,
        *,
        candidate_top_k: int = DEFAULT_MULTILINGUAL_CANDIDATE_TOP_K,
        candidate_pool_k: int = DEFAULT_MULTILINGUAL_CANDIDATE_POOL_K,
        final_top_k: int = DEFAULT_MULTILINGUAL_FINAL_TOP_K,
        rrf_k: int = DEFAULT_MULTILINGUAL_RRF_K,
        original_weight: float = DEFAULT_MULTILINGUAL_ORIGINAL_WEIGHT,
        translation_weight: float = DEFAULT_MULTILINGUAL_TRANSLATION_WEIGHT,
        bm25_index: BM25Index | None = None,
        hybrid_rrf_k: int = DEFAULT_MULTILINGUAL_RRF_K,
        hybrid_candidate_budget: int = HYBRID_CANDIDATE_BUDGET,
        hybrid_final_top_k: int = HYBRID_DEFAULT_FINAL_TOP_K,
    ) -> None:
        """注入可替换依赖，并在 Hybrid 启用时校验额外参数。"""

        _validate_top_k(candidate_top_k, field_name="candidate_top_k")
        _validate_top_k(candidate_pool_k, field_name="candidate_pool_k")
        _validate_top_k(final_top_k, field_name="final_top_k")
        if candidate_pool_k > candidate_top_k * 2:
            raise ValueError("candidate_pool_k must not exceed two-branch union potential")
        if final_top_k > candidate_pool_k:
            raise ValueError("final_top_k must not exceed candidate_pool_k")
        # 复用纯函数验证 RRF 常数和权重，不重复维护一套数值规则。
        weighted_rrf_fusion(
            [],
            [],
            original_weight=original_weight,
            translation_weight=translation_weight,
            rrf_k=rrf_k,
            final_top_k=final_top_k,
        )
        self.dense_retriever = dense_retriever
        self.translation_provider = translation_provider
        self.candidate_top_k = candidate_top_k
        self.candidate_pool_k = candidate_pool_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.original_weight = original_weight
        self.translation_weight = translation_weight

        # Hybrid 集成开关：仅当 bm25_index 非空时启用末轮 Equal RRF，避免给现有
        # §16.2 冻结路径引入任何运行时分支。
        self.bm25_index = bm25_index
        if bm25_index is not None:
            _validate_hybrid_rrf_k(hybrid_rrf_k)
            _validate_top_k(hybrid_candidate_budget, field_name="hybrid_candidate_budget")
            _validate_top_k(hybrid_final_top_k, field_name="hybrid_final_top_k")
        self.hybrid_rrf_k = hybrid_rrf_k
        self.hybrid_candidate_budget = hybrid_candidate_budget
        self.hybrid_final_top_k = hybrid_final_top_k

    def _fuse(
        self,
        original: list[RetrievedChunk],
        translated: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[MultilingualRetrievedChunk]:
        """以统一输出模型融合正常双路结果或 Original-only fallback。"""

        return weighted_rrf_fusion(
            original,
            translated,
            original_weight=self.original_weight,
            translation_weight=self.translation_weight,
            rrf_k=self.rrf_k,
            final_top_k=top_k,
        )

    def _fuse_with_bm25(
        self,
        first_stage: Sequence[MultilingualRetrievedChunk],
        query: str,
        *,
        top_k: int,
    ) -> list[MultilingualRetrievedChunk]:
        """§17.3.1 末轮 Equal RRF，把多语融合结果与 BM25 hits 统一重排。

        第一阶段输出的 rank 直接作为 Equal RRF 的 multilingual_fused rank；BM25
        hits 来自 ``self.bm25_index.search(query, top_k=hybrid_candidate_budget)``。
        Equal RRF 候选预算与离线 §17.3.1 实验保持一致，避免 Runtime 路径在
        不变更 SPEC 的前提下悄悄改变行为。
        """

        candidates: list[RankedChunkCandidate] = []
        for result in first_stage:
            candidates.append(
                RankedChunkCandidate(chunk=result.chunk, rank=result.rank)
            )
        bm25_hits = self.bm25_index.search(  # type: ignore[union-attr]
            query, top_k=self.hybrid_candidate_budget
        )
        bm25_candidates = [
            RankedChunkCandidate(chunk=hit.chunk, rank=hit.rank) for hit in bm25_hits
        ]
        fused = equal_ranked_rrf_fusion(
            candidates,
            bm25_candidates,
            rrf_k=self.hybrid_rrf_k,
            candidate_budget=self.hybrid_candidate_budget,
        )
        first_stage_by_chunk_id = {
            result.chunk.chunk_id: result for result in first_stage
        }
        bm25_by_chunk_id = {hit.chunk.chunk_id: hit for hit in bm25_hits}
        wrapped = _to_multilingual_retrieved_chunks(
            fused,
            first_stage_by_chunk_id,
            bm25_by_chunk_id,
            candidate_budget=self.hybrid_candidate_budget,
        )
        return wrapped[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[MultilingualRetrievedChunk]:
        """先运行 Original，再尝试 Translation；Hybrid 启用时再叠加 BM25 末轮融合。"""

        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("query must not be empty or whitespace")
        effective_top_k = self.final_top_k if top_k is None else top_k
        _validate_top_k(effective_top_k, field_name="top_k")
        if effective_top_k > self.candidate_pool_k:
            raise ValueError("top_k must not exceed candidate_pool_k")

        # Original 是可独立使用的基础能力，其错误必须原样传播而不能被 fallback 隐藏。
        original = self.dense_retriever.retrieve(query, top_k=self.candidate_top_k)
        try:
            translation = self.translation_provider.translate_to_english(query)
        except TranslationError:
            fused = self._fuse(original, [], top_k=self.candidate_pool_k)
        else:
            translated_text = translation.translated_text
            if (
                not translation.was_translated
                or not translated_text.strip()
                or translated_text.strip() == query.strip()
            ):
                fused = self._fuse(original, [], top_k=self.candidate_pool_k)
            else:
                try:
                    translated = self.dense_retriever.retrieve(
                        translated_text,
                        top_k=self.candidate_top_k,
                    )
                except Exception:
                    # 第二分支是增强能力；失败时保留已经成功获得的 Original 候选。
                    fused = self._fuse(original, [], top_k=self.candidate_pool_k)
                else:
                    # v2 保留双路并集到 candidate_pool_k；旧实现按单分支深度提前
                    # 截断，导致 caller 请求 20 时最多只能得到 10 个候选。
                    fused = self._fuse(original, translated, top_k=self.candidate_pool_k)

        # Hybrid 未启用时保持 §16.2 冻结行为：只截到 effective_top_k，不触发 BM25。
        if self.bm25_index is None:
            return fused[:effective_top_k]
        return self._fuse_with_bm25(fused, query, top_k=effective_top_k)
