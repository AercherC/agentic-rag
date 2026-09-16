"""使用可配置 Token 滑动窗口将 Clean Document 切分为 DocumentChunk。

Phase 13F 新增混合切分（§9.3）：``chunk_document_structured`` 消费结构解析元素流，
按 PH1 冻结语义（standalone 独立出块 + body 按 section 聚合、窗口零重叠）切分。
固定窗口路径保持不变，作为 ``chunk_strategy="fixed"`` 的回滚通道。
"""

from copy import deepcopy
from typing import Any, Sequence

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.document import Document
from evidenceflow.domain.ids import build_chunk_id
from evidenceflow.ingestion.tokenizer import TokenizerProtocol

SEGMENT_BOUNDARY = "\n\n"

# hybrid 策略的 standalone 元素类型：各成独立 Chunk 并从正文流移除（13C-C PC2/13E PH1 语义）。
HYBRID_STANDALONE_TYPES = {"table", "table_caption", "figure_caption", "equation"}
# hybrid 策略冻结的窗口重叠为 0（13E PH1 = hybrid_section_o0）。
HYBRID_OVERLAP_TOKENS = 0


class ChunkingError(ValueError):
    """表示 Chunk 配置、输入数据或 Tokenizer 输出无法生成有效 Chunk。"""


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """校验滑动窗口能够以正步长向后推进。"""

    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be smaller than chunk_size")


def _encode_checked(
    tokenizer: TokenizerProtocol,
    text: str,
    *,
    source_name: str,
    allow_empty: bool = False,
) -> list[int]:
    """调用 Tokenizer 并验证返回值符合内部 Token Stream 合同。"""

    try:
        token_ids = tokenizer.encode(text)
    except Exception as exc:
        raise ChunkingError(f"Tokenizer failed to encode {source_name}") from exc
    if not isinstance(token_ids, list) or not all(
        type(token_id) is int for token_id in token_ids
    ):
        raise ChunkingError(f"Tokenizer returned invalid token IDs for {source_name}")
    if not token_ids and not allow_empty:
        raise ChunkingError(f"Tokenizer returned no token IDs for {source_name}")
    return token_ids


def _build_token_stream(
    document: Document,
    tokenizer: TokenizerProtocol,
) -> tuple[list[int], list[int | None]]:
    """构建连续 Token Stream，并为每个 Token 保存真实来源页码。"""

    if not document.segments:
        raise ChunkingError("Document contains no segments to chunk")

    token_ids: list[int] = []
    page_numbers: list[int | None] = []
    boundary_ids: list[int] | None = None

    for segment_index, segment in enumerate(document.segments):
        if segment_index > 0:
            if boundary_ids is None:
                # 某些 SentencePiece Tokenizer 会忽略纯空白分隔符并返回空列表；
                # 这不属于异常，后一个 Segment 自身的词首标记仍能避免单词粘连。
                boundary_ids = _encode_checked(
                    tokenizer,
                    SEGMENT_BOUNDARY,
                    source_name="segment boundary",
                    allow_empty=True,
                )
            token_ids.extend(boundary_ids)
            # 人工分隔符不属于任何真实页面，计算 Chunk 页码时会忽略 None。
            page_numbers.extend([None] * len(boundary_ids))

        segment_ids = _encode_checked(
            tokenizer,
            segment.content,
            source_name=f"segment {segment_index}",
        )
        token_ids.extend(segment_ids)
        page_numbers.extend([segment.page_number] * len(segment_ids))

    if len(token_ids) != len(page_numbers):
        raise ChunkingError("Token IDs and page sources are not aligned")
    return token_ids, page_numbers


def _decode_checked(tokenizer: TokenizerProtocol, token_ids: list[int]) -> str:
    """解码一个窗口，并拒绝无法形成可用正文的结果。"""

    try:
        decoded_text = tokenizer.decode(token_ids)
    except Exception as exc:
        raise ChunkingError("Tokenizer failed to decode chunk token IDs") from exc
    if not isinstance(decoded_text, str):
        raise ChunkingError("Tokenizer decode returned non-text output")

    # 只移除窗口外围空白，不重新执行 Cleaner 或修改内部 Markdown 结构。
    decoded_text = decoded_text.strip()
    if not decoded_text:
        raise ChunkingError("Decoded chunk contains no usable text")
    return decoded_text


def chunk_document(
    document: Document,
    tokenizer: TokenizerProtocol,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """按正文 Token 数生成带稳定 ID 和页码范围的滑动窗口 Chunk。"""

    _validate_chunk_parameters(chunk_size, chunk_overlap)
    token_ids, page_numbers = _build_token_stream(document, tokenizer)
    step = chunk_size - chunk_overlap
    chunks: list[DocumentChunk] = []
    start = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_token_ids = token_ids[start:end]
        chunk_page_numbers = page_numbers[start:end]
        content = _decode_checked(tokenizer, chunk_token_ids)

        # Boundary Token 的 None 页码不参与范围计算，保留 PDF 的真实页码缺口。
        real_pages = [page for page in chunk_page_numbers if page is not None]
        page_start = min(real_pages) if real_pages else None
        page_end = max(real_pages) if real_pages else None
        chunk_index = len(chunks)

        # Chunk ID 必须基于最终 decode 文本创建，确保重复运行稳定可追踪。
        chunk_id = build_chunk_id(
            document_id=document.document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_index=chunk_index,
            content=content,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                content=content,
                token_count=len(chunk_token_ids),
                chunk_index=chunk_index,
                page_start=page_start,
                page_end=page_end,
                file_name=document.file_name,
                content_hash=document.content_hash,
                metadata=deepcopy(document.metadata),
            )
        )

        # 当前窗口已经覆盖末尾时立即停止，避免产生几乎全由 overlap 构成的尾块。
        if end == len(token_ids):
            break
        start += step

    if not chunks:
        raise ChunkingError("Document produced no chunks")
    return chunks


# --------------------------------------------------------------------------------------
# 混合切分（Phase 13F，§9.3）：元素级打包，语义移植自 13E PH1（hybrid_section_o0）。
# --------------------------------------------------------------------------------------


def _count_tokens(tokenizer: TokenizerProtocol, text: str) -> int:
    """对最终拼接文本精确计数；空文本计 0（与实验口径一致）。"""

    if not text:
        return 0
    token_ids = _encode_checked(tokenizer, text, source_name="hybrid chunk text", allow_empty=True)
    return len(token_ids)


def _split_oversized_element(
    text: str, tokenizer: TokenizerProtocol, target: int, overlap: int
) -> list[str]:
    """单个超长元素按词切分为 ≤target token 的窗口，相邻窗口按词近似重叠。

    移植自 14A-C ``_split_oversized_unit``：重叠按词数近似，从上一窗尾部回退词
    直至 token 和 ≤ overlap（至少回退 0 词）；overlap=0 时尾部仅保留 1 词。
    """

    words = text.split()
    if not words:
        return [text]
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = _count_tokens(tokenizer, word + " ")
        if current and current_tokens + word_tokens > target:
            windows.append(" ".join(current))
            # 计算重叠尾部：从尾部回退词，token 和 ≤ overlap。
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                prev_tokens = _count_tokens(tokenizer, prev + " ")
                if tail and tail_tokens + prev_tokens > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += prev_tokens
            current = tail + [word]
            current_tokens = _count_tokens(tokenizer, " ".join(current))
        else:
            current.append(word)
            current_tokens += word_tokens
    if current:
        windows.append(" ".join(current))
    return windows


def _pack_element_windows(
    units: Sequence[tuple[str, list[str], dict[str, Any]]],
    tokenizer: TokenizerProtocol,
    target: int,
    overlap: int,
    separator: str = SEGMENT_BOUNDARY,
) -> list[dict[str, Any]]:
    """元素 unit 流 → 打包窗口（移植 13C-C ``pack_units_sep``，语义保持一致）。

    - unit token > target → 词级拆分（owners 复制，split=True）；
    - 顺序贪心打包；关闭窗口时仅当 overlap>0 携带尾部 unit；
    - 硬预算优先于重叠：尾部与下一 unit 合并若超 target 则放弃尾部。
    """

    flat: list[tuple[str, list[str], dict[str, Any], bool]] = []
    for text, owners, element in units:
        if not text.strip():
            continue
        token_count = _count_tokens(tokenizer, text)
        if token_count > target:
            # 超长 unit 拆分为伪 unit（owners 复制；词级重叠为近似口径，记录 split=True）。
            for piece in _split_oversized_element(text, tokenizer, target, overlap):
                flat.append((piece, owners, element, True))
        else:
            flat.append((text, owners, element, False))

    def _joined(items: list[tuple[str, list[str], dict[str, Any], bool]]) -> str:
        return separator.join(item[0] for item in items)

    windows: list[dict[str, Any]] = []
    current: list[tuple[str, list[str], dict[str, Any], bool]] = []
    for unit in flat:
        text, owners, element, was_split = unit
        if current and _count_tokens(tokenizer, _joined(current) + separator + text) > target:
            # 关闭当前窗；重叠尾部仅当 overlap>0 时计算。
            tail: list[tuple[str, list[str], dict[str, Any], bool]] = []
            if overlap > 0:
                for prev in reversed(current):
                    candidate = [prev] + tail
                    if _count_tokens(tokenizer, _joined(candidate)) > overlap:
                        break
                    tail.insert(0, prev)
                # 硬预算优先于重叠：尾部与下一 unit 合并若超 target 则放弃尾部。
                if tail and _count_tokens(tokenizer, _joined(tail) + separator + text) > target:
                    tail = []
            windows.append(_make_hybrid_window(current, separator))
            current = list(tail)
        current.append(unit)
    if current:
        windows.append(_make_hybrid_window(current, separator))
    return windows


def _make_hybrid_window(
    items: list[tuple[str, list[str], dict[str, Any], bool]], separator: str
) -> dict[str, Any]:
    """由 unit 序列构造窗口：text=sep 拼接、owners 去重保序、split 任一为 True。"""

    text = separator.join(item[0] for item in items)
    owners: list[str] = []
    for _, element_ids, _, _ in items:
        for element_id in element_ids:
            if element_id not in owners:
                owners.append(element_id)
    owner_elements = [item[2] for item in items]
    return {
        "text": text,
        "element_ids": owners,
        "elements": owner_elements,
        "split": any(item[3] for item in items),
    }


def _hybrid_page_range(elements: Sequence[dict[str, Any]]) -> tuple[int | None, int | None]:
    """owner 元素真实页码的 min/max（无页码时为 None，不推测）。"""

    pages = [element["page"] for element in elements if isinstance(element.get("page"), int)]
    return (min(pages), max(pages)) if pages else (None, None)


def _hybrid_section_key(element: dict[str, Any]) -> tuple[str, ...]:
    """section 分组键（section_path 列表 → 元组；缺失用空元组）。"""

    path = element.get("section_path")
    return tuple(path) if isinstance(path, list) else ()


def chunk_document_structured(
    document: Document,
    elements: Sequence[dict[str, Any]],
    tokenizer: TokenizerProtocol,
    chunk_size: int,
) -> list[DocumentChunk]:
    """按 PH1 冻结语义切分结构化元素流（hybrid 策略，§9.3）。

    - standalone 类型（table / table_caption / figure_caption / equation）独立成 Chunk
      并从正文流移除，位于文档 Chunk 序列前部（元素阅读顺序）；
    - 正文按连续相同 section_path 分组，组边界强制断开，组内顺序贪心打包；
    - 窗口重叠冻结为 0（``HYBRID_OVERLAP_TOKENS``），组间由 section 边界自然分隔；
    - 硬预算：任何 Chunk token 数 ≤ chunk_size；
    - ``chunk_id`` 走 §10 ``build_chunk_id``，overlap 载荷记录 0（hybrid 冻结语义）；
    - Chunk metadata 记录 ``chunk_strategy="hybrid"``、``element_ids`` 溯源与
      ``heading_path``（正文组可空）。
    """

    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be greater than 0")
    if not elements:
        raise ChunkingError("Document contains no structured elements to chunk")

    overlap = HYBRID_OVERLAP_TOKENS
    # 先按元素类型拆分 standalone 与正文 unit 流（保持元素阅读顺序）。
    standalone_units: list[tuple[str, list[str], dict[str, Any]]] = []
    body_units: list[tuple[str, list[str], dict[str, Any]]] = []
    for element in elements:
        text = str(element.get("text", ""))
        element_id = element.get("element_id")
        if not text.strip() or not element_id:
            # 无文本元素（如图形载体）不入索引；缺 ID 的元素无法溯源，直接拒绝。
            if element_id is None:
                raise ChunkingError("Structured element is missing element_id")
            continue
        unit = (text, [str(element_id)], element)
        if element.get("element_type") in HYBRID_STANDALONE_TYPES:
            standalone_units.append(unit)
        else:
            body_units.append(unit)

    raw_windows: list[dict[str, Any]] = []
    # standalone 元素各自独立打包（超长时词级拆分为多个窗口）。
    for unit in standalone_units:
        raw_windows.extend(_pack_element_windows([unit], tokenizer, chunk_size, overlap))
    # 正文按连续 section_path 分组，组边界强制断开；组内贪心打包。
    body_groups: list[list[tuple[str, list[str], dict[str, Any]]]] = []
    previous_key: tuple[str, ...] | None = None
    for unit in body_units:
        key = _hybrid_section_key(unit[2])
        if previous_key is None or key != previous_key:
            body_groups.append([])
            previous_key = key
        body_groups[-1].append(unit)
    for group in body_groups:
        section_path = _hybrid_section_key(group[0][2])
        heading_path = " > ".join(section_path) if section_path else None
        for window in _pack_element_windows(group, tokenizer, chunk_size, overlap):
            window["heading_path"] = heading_path
            raw_windows.append(window)

    if not raw_windows:
        raise ChunkingError("Document produced no hybrid chunks")

    # 统一构造 DocumentChunk：chunk_index 从 0 连续；页码取 owner 元素真实 min/max。
    chunks: list[DocumentChunk] = []
    for chunk_index, window in enumerate(raw_windows):
        page_start, page_end = _hybrid_page_range(window["elements"])
        metadata = deepcopy(document.metadata)
        metadata["chunk_strategy"] = "hybrid"
        metadata["element_ids"] = list(window["element_ids"])
        if window.get("heading_path") is not None:
            metadata["heading_path"] = window["heading_path"]
        chunks.append(
            DocumentChunk(
                chunk_id=build_chunk_id(
                    document_id=document.document_id,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    chunk_index=chunk_index,
                    content=window["text"],
                ),
                document_id=document.document_id,
                content=window["text"],
                token_count=_count_tokens(tokenizer, window["text"]),
                chunk_index=chunk_index,
                page_start=page_start,
                page_end=page_end,
                file_name=document.file_name,
                content_hash=document.content_hash,
                metadata=metadata,
            )
        )
    return chunks
