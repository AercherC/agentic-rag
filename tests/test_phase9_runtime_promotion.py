"""Verify the complete Phase 9 runtime chain and the promotion-gate artifacts."""

import json
from pathlib import Path

from evidenceflow.context import ContextBuilder
from evidenceflow.domain import DocumentChunk, RetrievedChunk
from evidenceflow.generation import Generator
from evidenceflow.rag import RAGService
from evidenceflow.retrieval import DenseRerankerRetriever, LearnedReranker


PHASE9 = Path(__file__).parents[1] / "evals/evaluation_v2/phase9_runtime_promotion_v1"


class CharacterTokenizer:
    """Provide deterministic context token counting that does not depend on a real model."""

    def encode(self, text: str) -> list[int]:
        """Map each character to one test token."""

        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """Complete the test interface required by TokenizerProtocol."""

        return "".join(chr(token_id) for token_id in token_ids)


class FixedDenseRetriever:
    """Simulate the production ANN, always returning 20 candidates with a cosine score."""

    def __init__(self, count: int = 20) -> None:
        """Build the requested number of Dense candidates with unique identities."""

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
        """Record the candidate-generation depth and return the fixed ANN order."""

        self.calls.append((query, top_k))
        return self.rows[:top_k]


class ReverseScorer:
    """Give later candidates a higher score, proving the final order comes from the reranker."""

    def score_pairs(self, pairs):
        """Produce finite scores independent of the batch from the passage index."""

        return [float(passage.rsplit(" ", 1)[1]) for _, passage in pairs]


class CitingLLM:
    """Return a deterministic answer that follows the Generator protocol and cites the first source."""

    def __init__(self) -> None:
        self.calls = 0

    last_usage = None

    def generate_json(self, messages, *, schema_name, json_schema) -> str:
        """Record the call and return a v3 JSON answer that passes citation validation."""

        self.calls += 1
        return '{"status": "answered", "answer": "Highest reranked evidence [S1]."}'


def _pipeline(dense: FixedDenseRetriever) -> DenseRerankerRetriever:
    """Build a lightweight pipeline consistent with the production 20->20->5 contract."""

    return DenseRerankerRetriever(
        dense,
        LearnedReranker(ReverseScorer(), batch_size=4, max_candidates=20),
        candidate_pool_k=20,
        reranker_input_k=20,
        final_top_k=5,
    )


def test_full_runtime_chain_uses_dense20_reranker5_context_generation_and_citation() -> None:
    """A normal query must flow through Retrieval, Context, Generator, Citation, and RAGAnswer."""

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
    """Empty ANN candidates still go through the existing deterministic insufficient-context answer, without calling the LLM."""

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
    """The artifacts must cover quality, rollback, performance, architecture simplification, and the final Gate."""

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
