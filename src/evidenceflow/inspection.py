"""开发者巡检工具共用的纯格式化函数。

本模块只放无副作用的展示/筛选函数：不加载模型、不做 IO、不持有全局状态。
它被 `experiments/debug/` 下的 Inspector 脚本与对应的单元测试共同引用，
因此放在 src 侧，避免 tests → experiments 的反向依赖。

模块不参与 Runtime 检索、Context 构建与生成链路，删除它不会影响问答结果。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def content_preview(content: str, preview_chars: int = 500, *, full: bool = False) -> str:
    """压平外围空白并按字符截断，避免默认把完整 C512 Chunk 刷屏。"""

    if preview_chars <= 0:
        raise ValueError("preview_chars must be positive")
    normalized = content.strip()
    if full or len(normalized) <= preview_chars:
        return normalized
    return normalized[:preview_chars] + "…"


def format_page(page_start: int | None, page_end: int | None) -> str:
    """将无页码文本和单页/跨页 PDF 转为紧凑的人类可读表示。"""

    if page_start is None:
        return "None"
    return str(page_start) if page_start == page_end else f"{page_start}-{page_end}"


def format_retrieval_result(result: Any, *, preview_chars: int = 500, full: bool = False) -> str:
    """格式化单路 RetrievedChunk，保留 cosine 与来源定位字段。"""

    chunk = result.chunk
    return "\n".join(
        [
            f"Rank: {result.rank}",
            f"Cosine: {result.similarity_score:.6f}",
            f"File: {chunk.file_name}",
            f"Page: {format_page(chunk.page_start, chunk.page_end)}",
            f"Chunk index: {chunk.chunk_index}",
            f"Chunk ID: {chunk.chunk_id}",
            f"Document ID: {chunk.document_id}",
            f"Content: {content_preview(chunk.content, preview_chars, full=full)}",
        ]
    )


def select_records(
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    file_name: str | None = None,
) -> list[dict[str, Any]]:
    """对 Chroma 原始字段做稳定配对、文件过滤和上限裁剪。"""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("stored record fields are misaligned")
    records = [
        {"chunk_id": chunk_id, "content": content, "metadata": dict(metadata)}
        for chunk_id, content, metadata in zip(ids, documents, metadatas)
        if file_name is None or metadata.get("file_name") == file_name
    ]
    # Chroma get() 不承诺业务排序；按稳定 chunk_id 排序方便重复检查和测试。
    return sorted(records, key=lambda item: item["chunk_id"])[:limit]


def format_storage_record(record: Mapping[str, Any], *, preview_chars: int = 500, full: bool = False) -> str:
    """展示内容与来源 Metadata；默认不输出完整 Embedding。"""

    metadata = record["metadata"]
    domain_metadata = metadata.get("domain_metadata_json", "{}")
    try:
        domain_metadata = json.loads(domain_metadata)
    except (TypeError, json.JSONDecodeError):
        domain_metadata = {"raw": domain_metadata}
    return "\n".join(
        [
            f"Chunk ID: {record['chunk_id']}",
            f"Document ID: {metadata.get('document_id')}",
            f"File: {metadata.get('file_name')}",
            f"Chunk index: {metadata.get('chunk_index')}",
            f"Page: {format_page(metadata.get('page_start'), metadata.get('page_end'))}",
            f"Token count: {metadata.get('token_count')}",
            f"Metadata: {json.dumps(domain_metadata, ensure_ascii=False, sort_keys=True)}",
            f"Content: {content_preview(record['content'], preview_chars, full=full)}",
        ]
    )
