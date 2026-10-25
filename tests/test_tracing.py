"""Test RunTrace (PROJECT_SPEC §12.5): recorder, event ordering, payload allowlist, failure paths, and flag equivalence.

Fakes/Stubs only: a Fake Dense retriever, a Fake Reranker scorer, and a Fake LLM Client are wired
into the real DenseRerankerRetriever, LearnedReranker, ContextBuilder, Generator, and RAGService.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import pytest

from evidenceflow.__main__ import main
from evidenceflow.citation import CitationValidationError
from evidenceflow.config import Settings
from evidenceflow.context import ContextBuilder
from evidenceflow.domain import DocumentChunk, RetrievedChunk, RunTrace, TraceEvent
from evidenceflow.generation import GENERATOR_PROMPT_VERSION, GenerationContractError, Generator
from evidenceflow.llm import LLMRequestError
from evidenceflow.rag import RAGService
from evidenceflow.retrieval import DenseRerankerRetriever, LearnedReranker
from evidenceflow.tracing import NullTraceRecorder, RunTraceRecorder, safe_record

FAKE_API_KEY = "sk-fake-trace-test-key-0000"
SYSTEM_PROMPT_FRAGMENT = "You answer questions using only the supplied knowledge-base context."
QUERY = "QUERYTEXT 为什么 Agent 需要外部工具？"
ANSWER_BODY = "ANSWERBODY 工具为 Agent 提供外部能力 [S1]，并能执行动作 [S2]。"

# # §12.5 payload allowlist: each event's key set must equal it exactly.
PAYLOAD_KEYS = {
    "run.started": set(),
    "retrieval.dense.started": set(),
    "retrieval.dense.completed": {"candidate_count", "candidates"},
    "rerank.started": set(),
    "rerank.completed": {"input_count", "output_count", "truncated_pair_count", "results"},
    "context.started": set(),
    "context.completed": {
        "included_count",
        "skipped_count",
        "total_tokens",
        "entries",
        "context_text_sha256",
    },
    "generation.started": set(),
    "generation.completed": {
        "status",
        "llm_called",
        "repair_attempted",
        "repair_succeeded",
        "usages",
        "raw_response_lengths",
        "prompt_version",
    },
    "citation_validation.completed": {"valid_labels", "invalid_labels", "source_count"},
    "run.completed": set(),
    "run.failed": {"error_type", "stage"},
}
# # The keys allowed for each item inside a list-valued payload.
ITEM_KEYS = {
    ("retrieval.dense.completed", "candidates"): {"rank", "chunk_id", "similarity_score"},
    ("rerank.completed", "results"): {"rank", "chunk_id", "candidate_rank", "reranker_score"},
    ("context.completed", "entries"): {"source_label", "chunk_id", "token_count"},
}

SUCCESS_ORDER = [
    "run.started",
    "retrieval.dense.started",
    "retrieval.dense.completed",
    "rerank.started",
    "rerank.completed",
    "context.started",
    "context.completed",
    "generation.started",
    "generation.completed",
    "citation_validation.completed",
    "run.completed",
]


class CharacterTokenizer:
    """Treat characters as tokens, so the context budget is fully predictable."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _chunk(index: int) -> DocumentChunk:
    """Build a Chunk whose body text carries a detectable marker, to assert that body text never enters the trace."""

    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}",
        content=f"CHUNKBODY{index} 工具让 Agent 获得外部能力，第 {index} 段。",
        token_count=10,
        chunk_index=index,
        page_start=index + 1,
        page_end=index + 1,
        file_name="agents.pdf",
        content_hash="hash",
    )


def _dense_rows(count: int = 6) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=_chunk(i), rank=i + 1, similarity_score=0.9 - i / 100)
        for i in range(count)
    ]


class FakeDense:
    """Return fixed Dense candidates, or raise the given exception."""

    def __init__(self, rows: list[RetrievedChunk], error: Exception | None = None) -> None:
        self.rows = rows
        self.error = error

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.error is not None:
            raise self.error
        return self.rows[:top_k]


class FakeScorer:
    """Return a deterministic score from the paragraph number in the body text (higher paragraph = more relevant) and report the truncation count."""

    last_truncated_pair_count = 1

    def score_pairs(self, pairs):
        return [float(len(passage) % 5) + passage.count("5") for _, passage in pairs]


class FakeLLM:
    """Return preset replies in order, with usage and a fake API key that must never enter the trace."""

    def __init__(self, *replies: str) -> None:
        self.api_key = FAKE_API_KEY
        self.replies = list(replies)
        self.calls = 0
        self.last_usage: dict | None = None

    def generate(self, messages):  # # pragma: no cover - v3 only goes through generate_json
        raise AssertionError("must use generate_json")

    def generate_json(self, messages, *, schema_name, json_schema) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        # # total_tokens is not in the allowlist, so it must be filtered out of the trace.
        self.last_usage = {"prompt_tokens": 100 + self.calls, "completion_tokens": 7, "total_tokens": 999}
        return reply


def _reply(answer: str = ANSWER_BODY, status: str = "answered") -> str:
    return json.dumps({"status": status, "answer": answer}, ensure_ascii=False)


def build_service(
    recorder,
    *,
    replies: tuple[str, ...] = (),
    rows: list[RetrievedChunk] | None = None,
    dense_error: Exception | None = None,
) -> RAGService:
    """Assemble RAGService from real components plus Fake dependencies; the recorder is injected into all three components."""

    retriever = DenseRerankerRetriever(
        FakeDense(_dense_rows() if rows is None else rows, dense_error),
        LearnedReranker(FakeScorer(), batch_size=2, max_candidates=6),
        candidate_pool_k=6,
        reranker_input_k=6,
        final_top_k=3,
        trace_recorder=recorder,
    )
    generator = Generator(FakeLLM(*(replies or (_reply(),))), trace_recorder=recorder)
    return RAGService(
        retriever, ContextBuilder(CharacterTokenizer(), 4000), generator, trace_recorder=recorder
    )


def _run(service: RAGService, recorder: RunTraceRecorder, query: str = QUERY):
    recorder.start_run(query, run_id="run-1")
    answer = service.answer(query)
    return answer, recorder.finish()


# # ---------- success path ----------


def test_success_event_order_seq_and_payloads() -> None:
    recorder = RunTraceRecorder()
    answer, trace = _run(build_service(recorder), recorder)

    assert [event.name for event in trace.events] == SUCCESS_ORDER
    assert [event.seq for event in trace.events] == list(range(len(SUCCESS_ORDER)))
    assert trace.status == "completed"
    assert trace.run_id == "run-1"
    assert trace.query_sha256 == hashlib.sha256(QUERY.encode("utf-8")).hexdigest()
    by_name = {event.name: event.payload for event in trace.events}

    dense = by_name["retrieval.dense.completed"]
    assert dense["candidate_count"] == 6
    assert [item["chunk_id"] for item in dense["candidates"]] == [f"chunk-{i}" for i in range(6)]
    rerank = by_name["rerank.completed"]
    assert rerank["input_count"] == 6 and rerank["output_count"] == 3
    # # One truncation count per batch: each of the 3 batches reports 1.
    assert rerank["truncated_pair_count"] == 3
    assert [
        (item["rank"], item["chunk_id"], item["candidate_rank"], item["reranker_score"])
        for item in rerank["results"]
    ] == [
        (item.rank, item.chunk.chunk_id, item.candidate_rank, item.reranker_score)
        for item in answer.retrieved_chunks
    ]
    context = by_name["context.completed"]
    assert context["included_count"] == 3
    assert [entry["source_label"] for entry in context["entries"]] == ["[S1]", "[S2]", "[S3]"]
    generation = by_name["generation.completed"]
    assert generation["status"] == "answered"
    assert generation["llm_called"] is True
    assert generation["usages"] == [{"prompt_tokens": 101, "completion_tokens": 7}]
    assert generation["raw_response_lengths"] == [len(_reply())]
    assert generation["prompt_version"] == GENERATOR_PROMPT_VERSION
    assert by_name["citation_validation.completed"] == {
        "valid_labels": ["[S1]", "[S2]"],
        "invalid_labels": [],
        "source_count": 2,
    }


def test_payload_keys_match_whitelist_and_no_text_leaks() -> None:
    recorder = RunTraceRecorder()
    service = build_service(recorder)
    answer, trace = _run(service, recorder)

    for event in trace.events:
        assert set(event.payload) == PAYLOAD_KEYS[event.name], event.name
        for (name, key), allowed in ITEM_KEYS.items():
            if event.name == name:
                for item in event.payload[key]:
                    assert set(item) == allowed
    text = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
    assert "CHUNKBODY" not in text
    assert "QUERYTEXT" not in text and QUERY not in text
    assert "ANSWERBODY" not in text and answer.answer not in text
    assert FAKE_API_KEY not in text
    assert SYSTEM_PROMPT_FRAGMENT not in text
    assert "SOURCE_CONTEXT" not in text and "Content:" not in text
    assert "total_tokens\": 999" not in text


def test_context_sha256_uses_utf8_context_text() -> None:
    recorder = RunTraceRecorder()
    service = build_service(recorder)
    _, trace = _run(service, recorder)
    retrieved = service._retriever.retrieve(QUERY)
    expected = ContextBuilder(CharacterTokenizer(), 4000).build(retrieved).context_text
    payload = next(e.payload for e in trace.events if e.name == "context.completed")
    assert payload["context_text_sha256"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()


# # ---------- failure path ----------


def test_retrieval_failure_reraises_same_object_and_records_run_failed() -> None:
    error = RuntimeError("dense boom")
    recorder = RunTraceRecorder()
    service = build_service(recorder, dense_error=error)
    recorder.start_run(QUERY)

    with pytest.raises(RuntimeError) as excinfo:
        service.answer(QUERY)

    assert excinfo.value is error
    trace = recorder.finish()
    assert trace.status == "failed"
    assert [e.name for e in trace.events] == ["run.started", "retrieval.dense.started", "run.failed"]
    assert trace.events[-1].payload == {"error_type": "RuntimeError", "stage": "retrieval"}
    assert "dense boom" not in trace.model_dump_json()


def _capture(service: RAGService, exc_type: type[Exception]) -> Exception:
    with pytest.raises(exc_type) as excinfo:
        service.answer(QUERY)
    return excinfo.value


@pytest.mark.parametrize(
    ("replies", "exc_type", "stage"),
    [
        (("not json", "still not json"), GenerationContractError, "generation"),
        ((_reply("回答引用了不存在的来源 [S9]。"),), CitationValidationError, "citation_validation"),
    ],
)
def test_generation_and_citation_failures_propagate_unchanged(replies, exc_type, stage) -> None:
    baseline = _capture(build_service(None, replies=replies), exc_type)
    recorder = RunTraceRecorder()
    service = build_service(recorder, replies=replies)
    recorder.start_run(QUERY)
    traced = _capture(service, exc_type)

    # # Exception type, message, and exception chain are exactly the same as without tracing.
    assert type(traced) is type(baseline) and str(traced) == str(baseline)
    assert type(traced.__cause__) is type(baseline.__cause__)
    trace = recorder.finish()
    assert trace.status == "failed"
    assert trace.events[-1].name == "run.failed"
    assert trace.events[-1].payload == {"error_type": exc_type.__name__, "stage": stage}


def test_llm_error_object_is_reraised_identically() -> None:
    error = LLMRequestError("upstream down")

    class FailingLLM(FakeLLM):
        def generate_json(self, messages, *, schema_name, json_schema):
            raise error

    recorder = RunTraceRecorder()
    service = build_service(recorder)
    service._generator = Generator(FailingLLM(), trace_recorder=recorder)
    recorder.start_run(QUERY)
    with pytest.raises(LLMRequestError) as excinfo:
        service.answer(QUERY)
    assert excinfo.value is error
    names = [e.name for e in recorder.finish().events]
    assert names[-2:] == ["generation.started", "run.failed"]


def test_blank_query_records_nothing() -> None:
    recorder = RunTraceRecorder()
    recorder.start_run(" ")
    with pytest.raises(ValueError):
        build_service(recorder).answer("   ")
    assert recorder.events == []


# # ---------- deterministic abstention ----------


def test_empty_context_records_generation_completed_without_llm_call() -> None:
    recorder = RunTraceRecorder()
    answer, trace = _run(build_service(recorder, rows=[]), recorder)

    assert answer.status.value == "insufficient_context"
    assert [e.name for e in trace.events] == SUCCESS_ORDER
    generation = next(e.payload for e in trace.events if e.name == "generation.completed")
    assert generation["llm_called"] is False
    assert generation["usages"] == [] and generation["status"] == "insufficient_context"
    citation = next(e.payload for e in trace.events if e.name == "citation_validation.completed")
    assert citation == {"valid_labels": [], "invalid_labels": [], "source_count": 0}


# # ---------- flag equivalence and the recorder's own exception ----------


class ExplodingRecorder:
    """`record` always raises, used to verify a recorder defect does not affect the answer."""

    def __init__(self) -> None:
        self.calls = 0

    def record(self, name, **payload):
        self.calls += 1
        raise RuntimeError("recorder broken")


def test_exploding_recorder_does_not_change_answer(caplog) -> None:
    baseline = build_service(None).answer(QUERY)
    recorder = ExplodingRecorder()
    with caplog.at_level(logging.WARNING, logger="evidenceflow.tracing"):
        traced = build_service(recorder).answer(QUERY)

    assert traced.model_dump() == baseline.model_dump()
    assert recorder.calls == len(SUCCESS_ORDER)
    assert "recorder broken" not in caplog.text


QUESTIONS = [
    ("为什么 Agent 需要工具？", _reply()),
    ("What is ReAct?", _reply("ReAct interleaves reasoning and acting [S3].")),
    ("知识库里有天气预报吗？", _reply("知识库没有足够信息。", status="insufficient_context")),
]


@pytest.mark.parametrize(("query", "reply"), QUESTIONS)
def test_trace_on_off_answers_are_field_equal(query, reply) -> None:
    off = build_service(None, replies=(reply,)).answer(query)
    recorder = RunTraceRecorder()
    service = build_service(recorder, replies=(reply,))
    recorder.start_run(query)
    on = service.answer(query)

    assert on.model_dump() == off.model_dump()
    assert recorder.finish().status == "completed"


def test_null_recorder_is_accepted_and_changes_nothing() -> None:
    assert build_service(NullTraceRecorder()).answer(QUERY).model_dump() == (
        build_service(None).answer(QUERY).model_dump()
    )


def test_safe_record_none_and_exception() -> None:
    safe_record(None, "run.started", anything=1)
    safe_record(ExplodingRecorder(), "run.started")


# # ---------- RunTraceRecorder itself ----------


class FakeClock:
    """Return monotonic clock seconds from a preset sequence."""

    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        return self.values.pop(0)


FIXED_UTC = datetime(2026, 10, 15, 1, 2, 3, tzinfo=timezone.utc)


def test_record_before_start_run_is_dropped(caplog) -> None:
    recorder = RunTraceRecorder()
    with caplog.at_level(logging.WARNING, logger="evidenceflow.tracing"):
        recorder.record("run.started")
    assert recorder.events == []
    assert "before start_run" in caplog.text


def test_duration_ms_with_injected_clock() -> None:
    clock = FakeClock(10.0, 10.0, 10.5, 10.75, 11.0, 12.0)
    recorder = RunTraceRecorder(clock=clock, utc_now=lambda: FIXED_UTC)
    recorder.start_run("q", run_id="r")
    recorder.record("run.started")
    recorder.record("rerank.started")
    recorder.record("rerank.completed", input_count=1)
    recorder.record("citation_validation.completed")
    recorder.record("run.completed")
    trace = recorder.finish()

    assert [e.elapsed_ms for e in trace.events] == [0.0, 500.0, 750.0, 1000.0, 2000.0]
    assert [e.duration_ms for e in trace.events] == [None, None, 250.0, None, 2000.0]
    assert trace.started_at_utc == trace.completed_at_utc == FIXED_UTC.isoformat()
    # # Round-trip check: the persisted JSON can be reloaded as a RunTrace.
    assert RunTrace.model_validate_json(trace.model_dump_json()) == trace


def test_recorder_reuse_resets_events() -> None:
    recorder = RunTraceRecorder()
    recorder.start_run("first")
    recorder.record("run.started")
    recorder.record("run.failed", error_type="X", stage="retrieval")
    assert recorder.finish().status == "failed"

    recorder.start_run("second", run_id="r2")
    recorder.record("run.started")
    recorder.record("run.completed")
    trace = recorder.finish()
    assert trace.status == "completed"
    assert [e.seq for e in trace.events] == [0, 1]
    assert trace.query_sha256 == hashlib.sha256(b"second").hexdigest()


def test_finish_requires_terminal_run_event() -> None:
    recorder = RunTraceRecorder()
    recorder.start_run("q")
    with pytest.raises(ValueError):
        recorder.finish()
    recorder.record("run.started")
    recorder.record("rerank.started")
    with pytest.raises(ValueError):
        recorder.finish()


def test_trace_models_forbid_extra_fields() -> None:
    with pytest.raises(ValueError):
        TraceEvent(seq=0, name="run.started", elapsed_ms=0.0, payload={}, extra=1)


# ---------- CLI ask --trace ----------


def _patch_build_rag_service(monkeypatch, *, replies: tuple[str, ...] = ()) -> list:
    """Replace runtime.build_rag_service: assemble from Fake dependencies and capture the trace_recorder received."""

    received: list = []

    def fake_build(settings, *, retriever=None, generator=None, trace_recorder=None):
        received.append(trace_recorder)
        return build_service(trace_recorder, replies=replies)

    monkeypatch.setattr("evidenceflow.runtime.build_rag_service", fake_build)
    return received


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_cli_trace_written_on_success_and_stdout_unchanged(monkeypatch, tmp_path, capsys) -> None:
    received = _patch_build_rag_service(monkeypatch)
    assert main(["ask", QUERY], settings=_settings()) == 0
    plain = capsys.readouterr()

    trace_path = tmp_path / "trace.json"
    assert main(["ask", QUERY, "--trace", str(trace_path)], settings=_settings()) == 0
    traced = capsys.readouterr()

    assert received[0] is None and isinstance(received[1], RunTraceRecorder)
    assert traced.out == plain.out and traced.err == plain.err == ""
    raw = trace_path.read_text(encoding="utf-8")
    trace = RunTrace.model_validate_json(raw)
    assert trace.status == "completed"
    assert [e.name for e in trace.events] == SUCCESS_ORDER
    # # Sorted keys, 2-space indent, non-ASCII not escaped.
    assert raw == json.dumps(trace.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    assert "run.started" not in traced.out


def test_cli_trace_written_on_failure_with_same_exit_and_stderr(monkeypatch, tmp_path, capsys) -> None:
    replies = ("not json", "still not json")
    _patch_build_rag_service(monkeypatch, replies=replies)
    assert main(["ask", QUERY], settings=_settings()) == 1
    plain = capsys.readouterr()

    trace_path = tmp_path / "trace.json"
    assert main(["ask", QUERY, "--trace", str(trace_path)], settings=_settings()) == 1
    traced = capsys.readouterr()

    assert traced.out == plain.out == ""
    assert traced.err == plain.err
    assert traced.err.startswith("Generation format error: ")
    trace = RunTrace.model_validate_json(trace_path.read_text(encoding="utf-8"))
    assert trace.status == "failed"
    assert trace.events[-1].payload == {"error_type": "GenerationContractError", "stage": "generation"}


def test_cli_trace_write_failure_only_warns(monkeypatch, tmp_path, capsys) -> None:
    _patch_build_rag_service(monkeypatch)
    bad_path = tmp_path / "missing-dir" / "trace.json"
    exit_code = main(["ask", QUERY, "--trace", str(bad_path)], settings=_settings())
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: answered" in captured.out
    assert captured.err.count("\n") == 1 and captured.err.startswith("Warning: failed to write trace")
    assert not bad_path.exists()


def test_cli_without_trace_writes_no_files(monkeypatch, tmp_path, capsys) -> None:
    received = _patch_build_rag_service(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert main(["ask", QUERY], settings=_settings()) == 0
    assert received == [None]
    assert list(tmp_path.iterdir()) == []


def test_cli_trace_keyboard_interrupt_not_converted(monkeypatch, tmp_path) -> None:
    def fake_build(settings, *, retriever=None, generator=None, trace_recorder=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr("evidenceflow.runtime.build_rag_service", fake_build)
    with pytest.raises(KeyboardInterrupt):
        main(["ask", QUERY, "--trace", str(tmp_path / "t.json")], settings=_settings())
