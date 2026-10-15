"""EvidenceFlow 命令行入口。"""

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
    """渲染逐文件状态和批量统计，不泄露 Embedding 内容。"""

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
    """面向用户渲染 Answer 与实际引用来源，不暴露内部 ID 或检索分数。"""

    # Status 行让用户和下游区分“已回答”与“证据不足拒答”。
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
            # 单页避免冗余范围；Markdown/TXT 两端为 None 时完全省略页码行。
            pages = (
                str(source.page_start)
                if source.page_start == source.page_end
                else f"{source.page_start}-{source.page_end}"
            )
            lines.append(f"Pages: {pages}")
    return "\n".join(lines)


def _render_ask_error(exc: Exception) -> str:
    """将已知 Public Boundary 失败转成稳定、无敏感值的用户提示。"""

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
    """解析命令并返回适合 shell 的退出码。"""

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
                # CLI 只触发装配；生产依赖的选择和配置映射统一由 runtime.py 负责。
                from evidenceflow.runtime import build_ingestion_service

                ingestion_service = build_ingestion_service(active_settings)
            service = ingestion_service
            summary = service.ingest(args.path)
        except ActiveIndexProtectedError as exc:
            # 保护拒绝与执行失败区分：退出码 2 表示“被策略拒绝，没有改动任何索引”。
            print(f"Ingestion refused: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(render_ingestion_summary(summary, service.ingestion_root))
        return 1 if summary.failed else 0

    # 只有传 --trace 时才创建 recorder；否则与未接入 trace 时的调用路径完全一致。
    recorder: RunTraceRecorder | None = None
    if args.trace is not None:
        recorder = RunTraceRecorder()
        recorder.start_run(args.query)
    try:
        try:
            # 延迟导入 Composition Root，Fake Service CLI 测试不会加载模型或 Chroma。
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
        # 成功与失败都写 trace（KeyboardInterrupt 也会经过这里，但不会被转换）；
        # 写入失败只在 stderr 警告一行，不改变退出码与原有输出。
        if recorder is not None:
            _write_trace(recorder, args.trace)


def _write_trace(recorder: RunTraceRecorder, path: Path) -> None:
    """结束 run 并把 RunTrace 写成 UTF-8、键排序的 JSON；任何失败只输出一行警告。

    装配阶段失败时 RAGService 尚未记录 run.* 终止事件，``finish()`` 会抛 ValueError，
    同样按“trace 写入失败”处理。
    """

    try:
        trace = recorder.finish()
        text = json.dumps(
            trace.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, indent=2
        )
        path.write_text(text + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - trace 是旁路产物，不得影响 ask 的结果
        print(f"Warning: failed to write trace: {type(exc).__name__}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
