"""Tests for ``scripts/e2e_diagnose.py``.

``observe`` must reproduce the production ``RAGService`` answer path item by item, ``attribute``
writes the report, and output protection refuses to overwrite existing artifacts. Nothing here
loads a real model.
"""

from __future__ import annotations

import json
import math

import pytest

import scripts.e2e_diagnose as module
from evidenceflow.context import ContextBuilder
from evidenceflow.domain import DocumentChunk
from evidenceflow.eval.e2e.dataset import E2EDataset, compute_fingerprint
from evidenceflow.eval.e2e.mapping import E2EGroupMapping, E2EMapping
from evidenceflow.generation import Generator
from evidenceflow.rag import RAGService
from evidenceflow.retrieval import DenseRerankerRetriever, DenseRetriever, ExactDenseIndex, LearnedReranker


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class FixedEmbedding:
    """Fixed query vector.

    The inner product with each chunk is set by the angle of that chunk vector, so the dense
    ranking is predictable.
    """

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts):  # pragma: no cover - not ingested in this test
        raise AssertionError


class LengthScorer:
    """Score by passage length.

    The order deliberately disagrees with the dense order, so the reranker really has to change
    the ranking.
    """

    def score_pairs(self, pairs):
        return [float(len(passage)) for _, passage in pairs]


class FakeLLM:
    last_usage = None

    def generate(self, messages):  # pragma: no cover
        raise AssertionError

    def generate_json(self, messages, *, schema_name, json_schema) -> str:
        return json.dumps({"status": "answered", "answer": "PoinTr uses dynamic queries [S1] [S2]."})


def _chunk(index: int, document_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=f"k{index}", document_id=document_id, content=content, token_count=5,
                         chunk_index=index, page_start=1, page_end=1, file_name=f"{document_id}.pdf",
                         content_hash="h", metadata={"heading_path": "References" if index == 3 else "3 Method"})


def _components():
    contents = ["short", "a much longer passage about queries", "mid length text", "refs refs", "x" * 50, "tail"]
    chunks = [_chunk(i, "docA" if i % 2 == 0 else "docB", text) for i, text in enumerate(contents)]
    # Increasing angle means a decreasing inner product with the query [1, 0], so dense order is
    # k0, k1, ..., k5.
    vectors = [[math.cos(0.1 * i), math.sin(0.1 * i)] for i in range(len(chunks))]
    retriever = DenseRerankerRetriever(
        DenseRetriever(FixedEmbedding(), ExactDenseIndex(chunks, vectors), top_k=4),
        LearnedReranker(LengthScorer(), batch_size=2, max_candidates=4),
        candidate_pool_k=4, reranker_input_k=4, final_top_k=2,
    )
    return retriever, ContextBuilder(CharacterTokenizer(), 4000), Generator(FakeLLM())


def _case() -> dict:
    alt = {"document_id": "docB", "content_version": "v", "evidence_anchor": "queries",
           "locator": {"page_start": 1, "page_end": 1}}
    return {"case_id": "M1", "family_id": "f", "query": "How does PoinTr build queries?", "query_language": "en",
            "question_type": "multi_document", "tags": ["multi_doc"], "critical": False,
            "expected_status": "answered", "acceptable_abstention": False,
            "required_evidence_groups": [{"group_id": "G1", "alternatives": [alt]}],
            "required_claims": [{"claim_id": "C1", "text": "PoinTr uses dynamic queries.", "required": True,
                                 "evidence_anchors": [alt], "answer_keys": [["dynamic queries"]]}],
            "forbidden_claims": [], "unanswerable_kind": None,
            "provenance": {"drafted_by": "agent", "approved_by": None, "source": "test"}}


def test_observe_matches_the_production_answer_path() -> None:
    dataset = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c", "cases": [_case()]})
    retriever, builder, generator = _components()
    case = dataset.cases[0]

    observation = module.observe_case(case, retriever=retriever, context_builder=builder, generator=generator,
                                      depth=6, retrieval_only=False, collection_name="col", chunk_count=6)
    production = RAGService(retriever, builder, generator).answer(case.query)

    assert [item.chunk_id for item in observation.dense_ranking] == [f"k{i}" for i in range(6)]
    # The reranker reorders the candidate pool by length, so the final Top-K must match the
    # production RAGService exactly.
    assert observation.reranked_pool[: observation.final_top_k] == [
        item.chunk.chunk_id for item in production.retrieved_chunks]
    assert observation.reranked_pool == ["k1", "k2", "k3", "k0"]
    assert observation.source_chunk_ids == [source.chunk_id for source in production.sources]
    assert (observation.status, observation.answer) == (production.status, production.answer)
    assert observation.dense_ranking[3].heading_path == "References"


def test_observe_retrieval_only_never_calls_the_generator() -> None:
    dataset = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c", "cases": [_case()]})
    retriever, builder, _ = _components()

    observation = module.observe_case(dataset.cases[0], retriever=retriever, context_builder=builder,
                                      generator=None, depth=6, retrieval_only=True, collection_name="col",
                                      chunk_count=6)

    assert observation.generation_ran is False and observation.status is None
    assert observation.context_chunk_ids == ["k1", "k2"]


def test_observe_records_generation_contract_errors() -> None:
    class BrokenLLM(FakeLLM):
        def generate_json(self, messages, *, schema_name, json_schema) -> str:
            return "not json"

    dataset = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c", "cases": [_case()]})
    retriever, builder, _ = _components()
    observation = module.observe_case(dataset.cases[0], retriever=retriever, context_builder=builder,
                                      generator=Generator(BrokenLLM()), depth=6, retrieval_only=False,
                                      collection_name="col", chunk_count=6)

    assert observation.error_type == "GenerationContractError" and observation.status is None


def test_observe_cli_uses_production_wiring(monkeypatch, tmp_path) -> None:
    dataset = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c", "cases": [_case()]})
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(dataset.model_dump_json(), encoding="utf-8")
    retriever, builder, generator = _components()
    monkeypatch.setattr(module, "Settings", lambda: object())
    monkeypatch.setattr(module, "resolve_active_collection", lambda settings: "col")
    monkeypatch.setattr(module, "build_retriever", lambda settings: retriever)
    monkeypatch.setattr(module, "build_context_builder", lambda settings: builder)
    monkeypatch.setattr(module, "build_generator", lambda settings: generator)
    out = tmp_path / "obs.jsonl"

    assert module.main(["observe", "--dataset", str(dataset_path), "--out", str(out), "--depth", "6"]) == 0

    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert (record["case_id"], record["chunk_count"], record["collection_name"]) == ("M1", 6, "col")
    assert record["generation_ran"] is True


def test_attribute_cli_writes_report_and_markdown(tmp_path) -> None:
    dataset = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c", "cases": [_case()]})
    retriever, builder, generator = _components()
    observation = module.observe_case(dataset.cases[0], retriever=retriever, context_builder=builder,
                                      generator=generator, depth=6, retrieval_only=False,
                                      collection_name="col", chunk_count=6)
    mapping = E2EMapping(dataset_fingerprint=compute_fingerprint(dataset), collection_name="col",
                         index_signature="sig", chunk_count=6, claim_anchors=[],
                         groups=[E2EGroupMapping(case_id="M1", group_id="G1", status="FULLY_MAPPED",
                                                 chunk_sets=[["k1"]])])
    paths = {name: tmp_path / name for name in ("dataset.json", "mapping.json", "obs.jsonl")}
    paths["dataset.json"].write_text(dataset.model_dump_json(), encoding="utf-8")
    paths["mapping.json"].write_text(mapping.model_dump_json(), encoding="utf-8")
    paths["obs.jsonl"].write_text(observation.model_dump_json() + "\n", encoding="utf-8")
    out, markdown = tmp_path / "report.json", tmp_path / "report.md"

    code = module.main(["attribute", "--dataset", str(paths["dataset.json"]), "--mapping",
                        str(paths["mapping.json"]), "--observations", str(paths["obs.jsonl"]),
                        "--out", str(out), "--markdown", str(markdown)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["slices"]["multi_doc"]["question_layers"] == {"PASS": 1}
    assert report["slices"]["multi_doc"]["final_top_k_composition"]["mean_distinct_documents"] == 2.0
    assert "failure diagnosis" in markdown.read_text(encoding="utf-8")


def test_output_protection_refuses_before_reading_inputs(tmp_path, capsys) -> None:
    existing = tmp_path / "report.json"
    existing.write_text("{}", encoding="utf-8")
    args = ["attribute", "--dataset", "missing.json", "--mapping", "m.json", "--observations", "o.jsonl"]

    assert module.main([*args, "--out", str(existing)]) == 2
    assert module.main([*args, "--out", str(module.EVALS_ROOT / "x.json")]) == 2
    assert "REFUSED" in capsys.readouterr().err
    # A missing input is a runtime error (exit 1), not an output-protection refusal.
    assert module.main([*args, "--out", str(tmp_path / "new.json")]) == 1
