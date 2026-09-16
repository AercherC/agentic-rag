"""验证 Phase 9 完整 Runtime 链与 Promotion Gate 产物。"""

import json
from pathlib import Path

from evidenceflow.context import ContextBuilder
from evidenceflow.domain import DocumentChunk, RetrievedChunk
from evidenceflow.generation import Generator
from evidenceflow.rag import RAGService
from evidenceflow.retrieval import DenseRerankerRetriever, LearnedReranker


PHASE9 = Path(__file__).parents[1] / "evals/evaluation_v2/phase9_runtime_promotion_v1"


class CharacterTokenizer:
    """提供不依赖真实模型的确定性 Context Token 计数。"""

    def encode(self, text: str) -> list[int]:
        """把每个字符映射为一个测试 Token。"""

        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """满足 TokenizerProtocol 的完整测试接口。"""

        return "".join(chr(token_id) for token_id in token_ids)


class FixedDenseRetriever:
    """模拟正式 ANN，固定返回 20 个带 cosine 的候选。"""

    def __init__(self, count: int = 20) -> None:
        """构造指定数量且身份唯一的 Dense 候选。"""

        self.calls: list[tuple[str, int]] = []
        self.rows = []
        for index in range(count):
            chunk = DocumentChunk(
                chunk_id=f"chunk-{index}",
                document_id=f"document-{index}",
                content=f"evidence {index}",
                token_count=2,
                chunk_index=index,
                file_name="source.pdf",
                content_hash="hash",
            )
            self.rows.append(
                RetrievedChunk(
                    chunk=chunk,
                    rank=index + 1,
                    similarity_score=1 - index / 100,
                )
            )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """记录 Candidate Generation 深度并返回固定 ANN 顺序。"""

        self.calls.append((query, top_k))
        return self.rows[:top_k]


class ReverseScorer:
    """让后置候选获得更高分，证明最终排序来自 Reranker。"""

    def score_pairs(self, pairs):
        """从 passage 序号生成与 batch 无关的有限分数。"""

        return [float(passage.rsplit(" ", 1)[1]) for _, passage in pairs]


class CitingLLM:
    """返回符合 Generator 协议且引用首个来源的确定性回答。"""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages) -> str:
        """记录调用并返回可通过 Citation 校验的回答。"""

        self.calls += 1
        return "STATUS: ANSWERED\nANSWER:\nHighest reranked evidence [S1]."


def _pipeline(dense: FixedDenseRetriever) -> DenseRerankerRetriever:
    """构造与正式 20→20→5 合同一致的轻量 Pipeline。"""

    return DenseRerankerRetriever(
        dense,
        LearnedReranker(ReverseScorer(), batch_size=4, max_candidates=20),
        candidate_pool_k=20,
        reranker_input_k=20,
        final_top_k=5,
    )


def test_full_runtime_chain_uses_dense20_reranker5_context_generation_and_citation() -> None:
    """Normal Query 必须贯通 Retrieval、Context、Generator、Citation 与 RAGAnswer。"""

    dense = FixedDenseRetriever()
    llm = CitingLLM()
    service = RAGService(
        _pipeline(dense),
        ContextBuilder(CharacterTokenizer(), max_context_tokens=10_000),
        Generator(llm),
    )

    answer = service.answer("Which evidence ranks highest?")

    assert dense.calls == [("Which evidence ranks highest?", 20)]
    assert len(answer.retrieved_chunks) == 5
    assert answer.retrieved_chunks[0].chunk.chunk_id == "chunk-19"
    assert answer.retrieved_chunks[0].candidate_rank == 20
    assert answer.sources[0].chunk_id == "chunk-19"
    assert llm.calls == 1


def test_empty_runtime_retrieval_keeps_insufficient_context_behavior() -> None:
    """空 ANN 候选仍走既有确定性不足回答，且不调用 LLM。"""

    dense = FixedDenseRetriever(count=0)
    llm = CitingLLM()
    answer = RAGService(
        _pipeline(dense),
        ContextBuilder(CharacterTokenizer(), max_context_tokens=100),
        Generator(llm),
    ).answer("unknown")

    assert answer.retrieved_chunks == []
    assert answer.sources == []
    assert "没有足够" in answer.answer
    assert llm.calls == 0


def test_phase9_artifacts_prove_runtime_promotion_and_architecture_roles() -> None:
    """产物必须覆盖质量、回滚、性能、架构简化及最终 Gate。"""

    required = {
        "run_manifest.json",
        "runtime_before.json",
        "runtime_after.json",
        "runtime_regression.json",
        "paired_comparison.json",
        "latency.json",
        "rollback_validation.json",
        "architecture_simplification_v1.json",
        "cleanup_manifest_v3.json",
        "execution_result.json",
        "report.md",
    }
    assert required <= {path.name for path in PHASE9.iterdir()}

    execution = json.loads((PHASE9 / "execution_result.json").read_text(encoding="utf-8"))
    regression = json.loads((PHASE9 / "runtime_regression.json").read_text(encoding="utf-8"))
    rollback = json.loads((PHASE9 / "rollback_validation.json").read_text(encoding="utf-8"))
    architecture = json.loads(
        (PHASE9 / "architecture_simplification_v1.json").read_text(encoding="utf-8")
    )

    assert execution["runtime_promotion"] == "PASS"
    assert all(execution["gates"].values())
    assert regression["phase8_reproduced"] is True
    assert set(regression["new_minus_phase8"].values()) == {0.0}
    assert rollback["status"] == "PASS" and rollback["requires_code_change"] is False
    by_module = {row["module"]: row for row in architecture["modules"]}
    assert by_module["Translation"]["fallback_role"] == "FALLBACK_ONLY"
    assert by_module["BM25"]["runtime_default"] is False
    assert by_module["Hybrid"]["fallback_role"] == "KEEP_CONDITIONAL"
    assert by_module["RRF Stage 1"]["new_role"] == "multi-branch candidate selector"
