"""Phase 15B.5 归档的历史 BM25 Lexical Retrieval。

§17.3.1 Hybrid 集成把该模块引入 Runtime 路径；位于 ``retrieval/`` 包内，确保
正式生产代码不会反向依赖 ``evidenceflow.eval`` 实验脚手架。``evidenceflow.eval.lexical``
改为 re-export，本模块是真实实现位置。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.ingestion.tokenizer import TokenizerProtocol


@dataclass(frozen=True)
class BM25RetrievedChunk:
    """保存 BM25 排名、分数、原始 Chunk 身份及匹配 Token。"""

    chunk: DocumentChunk
    rank: int
    bm25_score: float
    matched_token_ids: tuple[int, ...]


def _validate_token_ids(token_ids: object, *, field_name: str) -> list[int]:
    """校验 Tokenizer 输出，避免第三方对象泄漏到正式检索排序逻辑。"""

    if not isinstance(token_ids, list) or not all(type(item) is int for item in token_ids):
        raise ValueError(f"{field_name} must be a list of integer token IDs")
    return token_ids


class BM25Index:
    """在固定 DocumentChunk 集合上建立确定性的内存 BM25 索引。"""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        tokenizer: TokenizerProtocol,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """使用注入的 BGE Tokenizer 编码同一批正式 Chunk，避免 Runtime 与离线实验漂移。"""

        if (
            isinstance(k1, bool)
            or not isinstance(k1, (int, float))
            or not math.isfinite(k1)
            or k1 <= 0
        ):
            raise ValueError("k1 must be a finite positive number")
        if (
            isinstance(b, bool)
            or not isinstance(b, (int, float))
            or not math.isfinite(b)
            or not 0 <= b <= 1
        ):
            raise ValueError("b must be a finite number between 0 and 1")

        self._tokenizer = tokenizer
        self._k1 = float(k1)
        self._b = float(b)
        self._chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in self._chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            # 重复 ID 会让 Gold 对齐和确定性 tie-break 失去唯一语义。
            raise ValueError("chunks must not contain duplicate chunk_id values")

        self._term_frequencies: list[Counter[int]] = []
        self._document_lengths: list[int] = []
        document_frequency: Counter[int] = Counter()
        for index, chunk in enumerate(self._chunks):
            token_ids = _validate_token_ids(
                tokenizer.encode(chunk.content),
                field_name=f"chunk[{index}] tokens",
            )
            frequencies = Counter(token_ids)
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(len(token_ids))
            # IDF 按包含某词项的文档数计算，同一 Chunk 内重复只计一次。
            document_frequency.update(frequencies.keys())
        self._document_frequency = document_frequency
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

    @property
    def document_count(self) -> int:
        """返回索引中的唯一正式 Chunk 数量。"""

        return len(self._chunks)

    @property
    def average_document_length(self) -> float:
        """返回用于长度归一化的平均 BGE Token 数。"""

        return self._average_document_length

    def _idf(self, token_id: int) -> float:
        """使用 Robertson BM25 的非负 IDF 变体计算词项稀有度。"""

        document_count = self.document_count
        document_frequency = self._document_frequency[token_id]
        return math.log(
            1.0
            + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def search(self, query: str, *, top_k: int = 20) -> list[BM25RetrievedChunk]:
        """按 BM25 分数检索，并以 chunk_id 作为稳定的同分排序键。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty or whitespace")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_ids = _validate_token_ids(
            self._tokenizer.encode(query), field_name="query tokens"
        )
        if not query_ids or not self._chunks:
            return []

        # Query 重复词不重复累加，避免自然语言中的词频偶然改变检索意图。
        query_terms = tuple(dict.fromkeys(query_ids))
        scored: list[tuple[float, str, int, tuple[int, ...]]] = []
        for index, (frequencies, document_length) in enumerate(
            zip(self._term_frequencies, self._document_lengths)
        ):
            matched = tuple(
                token_id for token_id in query_terms if token_id in frequencies
            )
            if not matched:
                continue
            length_ratio = (
                document_length / self._average_document_length
                if self._average_document_length > 0
                else 0.0
            )
            score = 0.0
            for token_id in matched:
                term_frequency = frequencies[token_id]
                denominator = term_frequency + self._k1 * (
                    1.0 - self._b + self._b * length_ratio
                )
                score += self._idf(token_id) * (
                    term_frequency * (self._k1 + 1.0) / denominator
                )
            chunk_id = self._chunks[index].chunk_id
            scored.append((score, chunk_id, index, matched))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            BM25RetrievedChunk(
                chunk=self._chunks[index].model_copy(deep=True),
                rank=rank,
                bm25_score=score,
                matched_token_ids=matched,
            )
            for rank, (score, _, index, matched) in enumerate(
                scored[:top_k], start=1
            )
        ]
