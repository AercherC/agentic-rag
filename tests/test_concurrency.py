"""Verify that shared Runtime components keep diagnostics and traces separate under concurrent
requests (without network access or model loading).

Each test uses a Barrier to hold multiple threads inside a component at the same time, forcing
genuine interleaving. The old implementation kept "last call" state in plain instance
attributes, so that interleaving caused calls to overwrite each other.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from evidenceflow.concurrency import PerThread
from evidenceflow.context import ContextBuilder
from evidenceflow.domain import DocumentChunk, LLMMessage, RetrievedChunk
from evidenceflow.generation import Generator
from evidenceflow.llm import OpenAICompatibleLLMClient
from evidenceflow.rag import RAGService
from evidenceflow.retrieval import DenseRerankerRetriever, LearnedReranker
from evidenceflow.tracing import RunTraceRecorder, resolve_recorder, trace_scope

THREADS = 4
BARRIER_TIMEOUT_SECONDS = 10


def _run_in_parallel(worker, count: int = THREADS) -> list:
    """Run worker(index) concurrently and return results in index order.

    Any exception raised by a worker is propagated unchanged.
    """

    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(worker, range(count)))


# ---------- PerThread ----------


class _Holder:
    value = PerThread(lambda: "default")


def test_per_thread_values_are_isolated_and_default_per_thread() -> None:
    holder = _Holder()
    holder.value = "main"
    barrier = threading.Barrier(THREADS, timeout=BARRIER_TIMEOUT_SECONDS)

    def worker(index: int) -> tuple[str, str]:
        first = holder.value
        holder.value = f"thread-{index}"
        barrier.wait()  # Read after all writes; a plain attribute keeps only the last value.
        return first, holder.value

    results = _run_in_parallel(worker)

    assert results == [("default", f"thread-{index}") for index in range(THREADS)]
    assert holder.value == "main"
    assert isinstance(_Holder.value, PerThread)


def test_per_thread_state_is_per_instance() -> None:
    first, second = _Holder(), _Holder()
    first.value = "first"

    assert second.value == "default"


# ---------- Real LLM Client (Fake SDK) ----------


class _BarrierSDK:
    """Fake OpenAI SDK: returns only after all threads arrive.

    Usage is taken from the thread marker in the prompt.
    """

    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties, timeout=BARRIER_TIMEOUT_SECONDS)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, temperature, **extra):
        marker = int(messages[-1]["content"].split("MARKER-")[1].split()[0])
        self.barrier.wait()
        reply = json.dumps({"status": "answered", "answer": f"answer {marker} [S1]"})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=reply, model_extra={}), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1000 + marker, completion_tokens=marker),
        )


def _client(sdk: _BarrierSDK) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        api_key="test-only-key", base_url="https://example.test/v1", model="m", _client=sdk
    )


def test_llm_client_last_usage_is_per_thread() -> None:
    client = _client(_BarrierSDK(THREADS))

    def worker(index: int) -> dict | None:
        client.generate([LLMMessage(role="user", content=f"JSON MARKER-{index} please")])
        return client.last_usage

    usages = _run_in_parallel(worker)

    assert usages == [
        {"prompt_tokens": 1000 + index, "completion_tokens": index} for index in range(THREADS)
    ]


# ---------- Full RAGService: shared components + per-request trace_scope ----------


class _CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _rows() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=f"chunk-{index}",
                document_id=f"doc-{index}",
                content=f"passage {index}",
                token_count=3,
                chunk_index=index,
                file_name="paper.pdf",
                content_hash="hash",
            ),
            rank=index + 1,
            similarity_score=0.9 - index / 100,
        )
        for index in range(4)
    ]


class _FakeDense:
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return _rows()[:top_k]


class _BarrierScorer:
    """All threads stop inside scoring; counts report each thread's marker.

    This mimics the per-thread state a real scorer would keep.
    """

    last_truncated_pair_count = PerThread(int)

    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties, timeout=BARRIER_TIMEOUT_SECONDS)

    def score_pairs(self, pairs):
        marker = int(pairs[0][0].split("MARKER-")[1].split()[0])
        self.barrier.wait()
        self.last_truncated_pair_count = marker
        return [float(len(passage)) for _, passage in pairs]


def _shared_service(parties: int) -> tuple[RAGService, Generator]:
    """Build every component once without a recorder, like a process-wide server singleton."""

    retriever = DenseRerankerRetriever(
        _FakeDense(),
        # batch_size exceeds the candidate count: one scoring pass per rerank, so the
        # number of Barrier parties is predictable.
        LearnedReranker(_BarrierScorer(parties), batch_size=8, max_candidates=4),
        candidate_pool_k=4,
        reranker_input_k=4,
        final_top_k=2,
    )
    generator = Generator(_client(_BarrierSDK(parties)))
    service = RAGService(retriever, ContextBuilder(_CharacterTokenizer(), 4000), generator)
    return service, generator


def test_concurrent_runs_keep_their_own_trace_and_diagnostics() -> None:
    service, generator = _shared_service(THREADS)

    def worker(index: int):
        query = f"question MARKER-{index} about point clouds"
        recorder = RunTraceRecorder()
        recorder.start_run(query, run_id=f"run-{index}")
        with trace_scope(recorder):
            answer = service.answer(query)
        return answer, generator.last_diagnostics, recorder.finish()

    results = _run_in_parallel(worker)

    for index, (answer, diagnostics, trace) in enumerate(results):
        assert answer.answer == f"answer {index} [S1]"
        assert diagnostics["usages"] == [{"prompt_tokens": 1000 + index, "completion_tokens": index}]
        assert trace.run_id == f"run-{index}"
        names = [event.name for event in trace.events]
        # Each trace is exactly one complete run, with no events leaked from other threads.
        assert names.count("run.started") == 1 and names[-1] == "run.completed"
        by_name = {event.name: event.payload for event in trace.events}
        assert by_name["rerank.completed"]["truncated_pair_count"] == index
        assert by_name["generation.completed"]["usages"] == [
            {"prompt_tokens": 1000 + index, "completion_tokens": index}
        ]


def test_explicit_recorder_takes_precedence_over_scope() -> None:
    explicit, scoped = RunTraceRecorder(), RunTraceRecorder()

    with trace_scope(scoped):
        assert resolve_recorder(explicit) is explicit
        assert resolve_recorder(None) is scoped
    assert resolve_recorder(None) is None


def test_trace_scope_restores_previous_recorder_after_error() -> None:
    outer, inner = RunTraceRecorder(), RunTraceRecorder()

    with trace_scope(outer):
        with pytest.raises(RuntimeError):
            with trace_scope(inner):
                raise RuntimeError("boom")
        assert resolve_recorder(None) is outer
