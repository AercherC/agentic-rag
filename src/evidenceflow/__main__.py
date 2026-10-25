"""EvidenceFlow command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from evidenceflow.config import Settings
from evidenceflow.citation import CitationValidationError
from evidenceflow.domain import RAGAnswer
from evidenceflow.generation import GenerationContractError
from evidenceflow.ingestion.service import IngestionService, IngestionSummary
from evidenceflow.knowledge import ActiveCollectionError, ActiveIndexProtectedError
from evidenceflow.llm import LLMConfigurationError, LLMRequestError
from evidenceflow.rag import RAGService
from evidenceflow.tracing import RunTraceRecorder


def render_ingestion_summary(summary: IngestionSummary, root: Path) -> str:
    """Render per-file status and batch statistics without leaking Embedding content."""

    lines = ["EvidenceFlow Ingestion", "", f"Root: {root.resolve()}", "", f"Discovered: {summary.files_discovered} files", ""]
    for item in summary.results:
        lines += [f"[{item.status.upper()}]", item.source_path]
        if item.status in {"inserted", "updated"}:
            lines.append(f"chunks={item.chunk_count} embeddings={item.embedding_count}")
        elif item.status == "failed":
            lines += [f"stage={item.failed_stage}", f"reason={item.error_message}"]
        lines.append("")
    lines += [
        "Summary",
        "",
        f"Inserted: {summary.inserted}",
        f"Updated: {summary.updated}",
        f"Skipped: {summary.skipped}",
        f"Failed: {summary.failed}",
        f"Chunks written: {summary.chunks_written}",
        f"Embeddings created: {summary.embeddings_created}",
    ]
    return "\n".join(lines)


def render_rag_answer(answer: RAGAnswer) -> str:
    """Render the Answer and its actually cited sources for the user, exposing no internal IDs or retrieval scores."""

    # The Status line lets users and downstream code tell "answered" from "insufficient evidence,
    # abstained".
    lines = [
        "EvidenceFlow",
        "",
        "Question:",
        answer.query,
        "",
        f"Status: {answer.status.value}",
        "Answer:",
        answer.answer,
        "",
        "Sources:",
    ]
    if not answer.sources:
        lines.append("No cited sources.")
        return "\n".join(lines)

    for index, source in enumerate(answer.sources):
        if index:
            lines.append("")
        lines.extend([source.source_label, f"File: {source.file_name}"])
        if source.page_start is not None:
            # A single page avoids a redundant range; when both ends are None (Markdown/TXT) the
            # page line is omitted entirely.
            pages = (
                str(source.page_start)
                if source.page_start == source.page_end
                else f"{source.page_start}-{source.page_end}"
            )
            lines.append(f"Pages: {pages}")
    return "\n".join(lines)


def _render_ask_error(exc: Exception) -> str:
    """Turn known Public Boundary failures into stable user hints that contain no sensitive values."""

    if isinstance(exc, ActiveCollectionError):
        prefix = "Index configuration error"
    elif isinstance(exc, LLMConfigurationError):
        prefix = "LLM configuration error"
    elif isinstance(exc, LLMRequestError):
        prefix = "LLM request failed"
    elif isinstance(exc, GenerationContractError):
        prefix = "Generation format error"
    elif isinstance(exc, CitationValidationError):
        prefix = "Citation validation failed"
    else:
        prefix = "Retrieval or RAG processing failed"
    return f"{prefix}: {exc}"


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    ingestion_service: IngestionService | None = None,
    rag_service: RAGService | None = None,
) -> int:
    """Parse the command and return a shell-friendly exit code."""

    parser = argparse.ArgumentParser(prog="python -m evidenceflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", help="增量更新本地知识库")
    ingest_parser.add_argument("path", nargs="?", type=Path, help="root 内的单文件或子目录")
    ask_parser = subparsers.add_parser("ask", help="基于本地知识库回答单个问题")
    ask_parser.add_argument("query", help="使用引号包裹的完整问题")
    ask_parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        metavar="PATH",
        help="把本次提问的 RunTrace 写成 JSON（只含 ID、计数、分数、耗时与 token 用量）",
    )
    args = parser.parse_args(argv)

    active_settings = settings or Settings()
    if args.command == "ingest":
        try:
            if ingestion_service is None:
                # The CLI only triggers assembly; choosing production dependencies and mapping
                # configuration is runtime.py's job.
                from evidenceflow.runtime import build_ingestion_service

                ingestion_service = build_ingestion_service(active_settings)
            service = ingestion_service
            summary = service.ingest(args.path)
        except ActiveIndexProtectedError as exc:
            # Distinguish a protective refusal from an execution failure: exit code 2 means
            # "rejected by policy, no index was changed".
            print(f"Ingestion refused: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(render_ingestion_summary(summary, service.ingestion_root))
        return 1 if summary.failed else 0

    # Create a recorder only when --trace is passed; otherwise the call path is identical to
    # running without tracing.
    recorder: RunTraceRecorder | None = None
    if args.trace is not None:
        recorder = RunTraceRecorder()
        recorder.start_run(args.query)
    try:
        try:
            # Import the Composition Root lazily so Fake Service CLI tests load neither models nor
            # Chroma.
            if rag_service is None:
                from evidenceflow.runtime import build_rag_service

                if recorder is None:
                    rag_service = build_rag_service(active_settings)
                else:
                    rag_service = build_rag_service(active_settings, trace_recorder=recorder)
            answer = rag_service.answer(args.query)
        except Exception as exc:
            print(_render_ask_error(exc), file=sys.stderr)
            return 1
        print(render_rag_answer(answer))
        return 0
    finally:
        # Write the trace on both success and failure (KeyboardInterrupt also passes through here,
        # but is not converted);
        # a write failure only warns on stderr and changes neither the exit code nor the existing
        # output.
        if recorder is not None:
            _write_trace(recorder, args.trace)


def _write_trace(recorder: RunTraceRecorder, path: Path) -> None:
    """Finish the run and write the RunTrace as UTF-8, key-sorted JSON; any failure only prints one warning line.

        When assembly fails, RAGService has not recorded a run.* terminal event yet, so ``finish()``
        raises ValueError; that is handled the same way, as a trace write failure.
    """

    try:
        trace = recorder.finish()
        text = json.dumps(
            trace.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, indent=2
        )
        path.write_text(text + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - the trace is a side channel and must not affect the result of ask
        print(f"Warning: failed to write trace: {type(exc).__name__}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
