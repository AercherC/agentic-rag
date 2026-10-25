"""Compose the formal Retrieval, Context, Generation, and Citation layers into a single-turn RAG."""

from __future__ import annotations

import hashlib
from typing import Protocol

from evidenceflow.citation import assemble_rag_answer, parse_citations
from evidenceflow.domain.answer import RAGAnswer
from evidenceflow.domain.chunk import (
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.context import BuiltContext
from evidenceflow.domain.generation import GeneratedAnswer
from evidenceflow.tracing import TraceRecorder, safe_record


class RetrieverProtocol(Protocol):
    """The minimal formal retrieval interface RAGService needs."""

    def retrieve(
        self, query: str
    ) -> list[RetrievedChunk | RerankedRetrievedChunk]: ...


class ContextBuilderProtocol(Protocol):
    """The minimal Context-building interface RAGService needs."""

    def build(
        self,
        retrieved_chunks: list[RetrievedChunk | RerankedRetrievedChunk],
    ) -> BuiltContext: ...


class GeneratorProtocol(Protocol):
    """The minimal Grounded Generation interface RAGService needs."""

    def generate(self, query: str, context: BuiltContext) -> GeneratedAnswer: ...


class RAGService:
    """Orchestrate the existing components in the frozen order only, and return a Citation-validated RAGAnswer."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        context_builder: ContextBuilderProtocol,
        generator: GeneratorProtocol,
        *,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """Inject already-created components so the Service never implicitly loads expensive Runtime dependencies.

                ``trace_recorder`` is optional (default None): it records only run.*, context.*, and
                citation_validation.completed, and never changes answers or exceptions (PROJECT_SPEC §12.5).
                Retrieval and generation internal events are recorded by each component's own recorder.
        """

        self._retriever = retriever
        self._context_builder = context_builder
        self._generator = generator
        self.trace_recorder = trace_recorder

    def answer(self, query: str) -> RAGAnswer:
        """Run Retrieve->Context->Generate->Citation, letting errors propagate unchanged."""

        # The blank-query check stays first: the run has not started yet, so no event is recorded.
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        recorder = self.trace_recorder
        safe_record(recorder, "run.started")
        # Track the current stage name so a failure can write run.failed.stage.
        stage = "retrieval"
        try:
            # Pass no override parameters to the Retriever, so the formal Final Top-K frozen by the
            # Composition Root is used.
            retrieved_chunks = self._retriever.retrieve(query)
            stage = "context"
            safe_record(recorder, "context.started")
            built_context = self._context_builder.build(retrieved_chunks)
            if recorder is not None:
                # Record only labels, chunk_id, token counts, and the UTF-8 sha256 of context_text;
                # never the body.
                safe_record(
                    recorder,
                    "context.completed",
                    included_count=built_context.included_count,
                    skipped_count=built_context.skipped_count,
                    total_tokens=built_context.total_tokens,
                    entries=[
                        {
                            "source_label": entry.source_label,
                            "chunk_id": entry.chunk_id,
                            "token_count": entry.token_count,
                        }
                        for entry in built_context.entries
                    ],
                    context_text_sha256=hashlib.sha256(
                        built_context.context_text.encode("utf-8")
                    ).hexdigest(),
                )
            stage = "generation"
            generated_answer = self._generator.generate(query, built_context)
            stage = "citation_validation"
            answer = assemble_rag_answer(
                query=query,
                generated_answer=generated_answer,
                built_context=built_context,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as exc:
            # Record run.failed first (only the exception class name and stage, never the message),
            # then re-raise as is:
            # no wrapping and no chain rewriting, so callers (CLI error-prefix mapping, etc.) see
            # exactly the same exception as without tracing.
            safe_record(recorder, "run.failed", error_type=type(exc).__name__, stage=stage)
            raise
        if recorder is not None:
            # Recompute labels with the same parser after assemble_rag_answer succeeds; this cost
            # is paid only when tracing is enabled.
            parsed = parse_citations(answer.answer, built_context)
            safe_record(
                recorder,
                "citation_validation.completed",
                valid_labels=list(parsed.valid_labels),
                invalid_labels=list(parsed.invalid_labels),
                source_count=len(answer.sources),
            )
        safe_record(recorder, "run.completed")
        return answer
