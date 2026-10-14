"""EvidenceFlow 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from evidenceflow.config import Settings
from evidenceflow.citation import CitationValidationError
from evidenceflow.domain import RAGAnswer
from evidenceflow.generation import GenerationContractError
from evidenceflow.ingestion.service import IngestionService, IngestionSummary
from evidenceflow.llm import LLMConfigurationError, LLMRequestError
from evidenceflow.rag import RAGService


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

    if isinstance(exc, LLMConfigurationError):
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
        except Exception as exc:
            print(f"Ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(render_ingestion_summary(summary, service.ingestion_root))
        return 1 if summary.failed else 0

    try:
        # 延迟导入 Composition Root，Fake Service CLI 测试不会加载模型或 Chroma。
        if rag_service is None:
            from evidenceflow.runtime import build_rag_service

            rag_service = build_rag_service(active_settings)
        answer = rag_service.answer(args.query)
    except Exception as exc:
        print(_render_ask_error(exc), file=sys.stderr)
        return 1
    print(render_rag_answer(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
