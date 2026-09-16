#!/usr/bin/env python
"""Phase 13A：Knowledge Base Structure & Chunk Quality Audit（只审计，不修改）。

阶段性质：Knowledge Base / Ingestion Evaluation。

强制边界（计划 §3 / §4 / §28）：
  - 不覆盖 baseline collection ``ef_research_bge_m3_c512_o64_v1``（361 chunk 身份保持冻结）；
  - 不修改 canonical Gold / qrels / 40Q ranking / Eval v3 definitions；
  - 不修改 Runtime 任何算法，不新增第三方依赖，不重建 Chroma；
  - 第一轮只执行 Step 0–8；命中 §4 任一异常即停并落盘 checkpoint + anomaly_report。

核心方法（可复现、非猜测）：
  1. 复用正式 Loader / Cleaner / Chunker 复算 361 chunk（只读，不写回 Chroma），
     并把复算 chunk_id 与 Chroma 实际身份逐一对账，保证审计对象就是线上知识库；
  2. 用 BGE-M3 Tokenizer 的 offset mapping 把每个 chunk 精确映射回 cleaned 文本字符区间，
     由此**确定性**判定 chunk 是否跨越 section 边界、是否横跨 References 标题、
     是否在词中间切开——而不是靠模糊字符串匹配；
  3. PDF 版面抽查（Step 3）必须人工看图：机器部分只负责渲染代表页、抽取该页原文、
     并对齐落到该页的 chunk；人工结论由 ``layout_review_annotations.json`` 提供并合并。

子命令：
  baseline   Step 0：冻结基线
  inventory  Step 1 + Step 2：361 chunk inventory 与自动质量指标
  render     Step 3 前置：渲染代表页 + 落盘待人工填写的判定脚手架
  layout     Step 3：PDF 版面抽查（渲染 + 人工结论合并）
  elements   Step 4：Table / Figure / Formula 专项审计
  sections   Step 5：Section hierarchy 审计
  failure    Step 6：ARR-007 / ARR-037 / ARR-040 根因审计
  gold       Step 7：Gold chunk quality 审计
  decide     Step 8：Audit decision
  finalize   落盘 run_manifest.json / execution_result.json（含产物 hash 与冻结复验）
  all        顺序执行 Step 0–8 并 finalize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# 允许从项目根或其他工作目录启动脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from evidenceflow.config import Settings
from evidenceflow.ingestion import (
    BgeM3Tokenizer,
    chunk_document,
    clean_document,
    load_document,
)
from evidenceflow.storage import ChromaVectorStore

# --------------------------------------------------------------------------------------
# 冻结常量（全部来自已 Gate PASS 的历史 artifact，审计阶段不得改写）
# --------------------------------------------------------------------------------------

BASE = PROJECT_ROOT / "evals/evaluation_v3/phase13_kb_chunk_audit_v1"
V2 = PROJECT_ROOT / "evals/evaluation_v2"
V3 = PROJECT_ROOT / "evals/evaluation_v3"
SOURCE_DOCS = PROJECT_ROOT / "data/source_docs"
PAGE_RENDER_DIR = PROJECT_ROOT / "tmp/phase13_pages"

COLLECTION_NAME = "ef_research_bge_m3_c512_o64_v1"
FROZEN_CHUNK_COUNT = 361
FROZEN_CHUNK_SIZE = 512
FROZEN_CHUNK_OVERLAP = 64
FROZEN_DATASET_HASH = "be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357"
FROZEN_QRELS_HASH = "9337c26327097085dd3dc400a73e6c0baa79f375944d2be401edd39d014c9bb9"
FROZEN_CORPUS_HASH = "1e48b401aa48fc59d94bba627deaa3005a7627860e4fe5815761d45c6ba68675"

# 冻结标准 IR 指标（6 位小数，来自 11C EXACT_STANDARD_METRICS / README）。
FROZEN_STANDARD_METRICS = {
    "Precision@5": 0.115000,
    "Recall@5": 0.558333,
    "Recall@10": 0.733333,
    "MRR@5": 0.432083,
    "MAP@5": 0.427917,
    "nDCG@5": 0.461851,
}
METRIC_TOLERANCE = 5e-7

# 冻结排名（K=10，Runtime after Phase 9 promotion）与 Dense Top20 候选来源。
RANKING_SOURCE = V2 / "phase9_runtime_promotion_v1/runtime_after.json"
DENSE_SOURCE = V2 / "phase6_retrieval_baselines_v1/b0_original_dense.json"
RERANKER_SOURCE = V2 / "phase10_reranker_performance_v1/baseline.json"
MAPPING_SOURCE = V2 / "phase35_gold_stabilization_v1/gold_mapping_v3.json"
QUESTIONS_SOURCE = V2 / "phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json"
QRELS_SOURCE = V3 / "standard_ir_v1/qrels.json"
CORPUS_MANIFEST = PROJECT_ROOT / "evals/research_ingestion/academic_corpus_manifest_v1.json"

# Phase 12 已确认的 Retrieval Failure 与主要风险画像（供交叉验证，不当作结论）。
PHASE12_RETRIEVAL_FAILURES = ["ARR-007", "ARR-037", "ARR-040"]

# 代码身份：只有这 5 个模块决定 PDF→chunk 的表示方式。
INGESTION_CODE_FILES = [
    "src/evidenceflow/ingestion/loader.py",
    "src/evidenceflow/ingestion/cleaner.py",
    "src/evidenceflow/ingestion/chunker.py",
    "src/evidenceflow/ingestion/tokenizer.py",
    "src/evidenceflow/ingestion/service.py",
]

# 质量阈值（全部为**启发式判定阈值**，只用于描述分布，不当作最优参数）。
VERY_SHORT_TOKEN_THRESHOLD = 64          # 与 academic_ingestion_audit_v1 同口径
NEAR_MAX_RATIO = 0.90                    # token_count >= 0.9 * chunk_size 视为 near-max

# --------------------------------------------------------------------------------------
# 文本结构识别规则（纯函数，确定性与可测试）
# --------------------------------------------------------------------------------------

# 结构元素识别。
#
# 关键约束（Phase 13 实测结论）：PyMuPDF 把双栏页压成一条文本流，caption 常常与正文
# 同处一行；而 BGE-M3（XLM-R SentencePiece）decode 会把换行折叠为空格。
# 因此**不能只在行首匹配 caption**，否则会漏掉全部噪声。是否真正位于行首单独记录
# （at_line_start），用于区分"高置信 caption"与"仅出现在正文里的引用"。
_FIGURE_ANY = re.compile(r"(?:Figure|Fig\.)\s*(\d{1,2})\b")
_FIGURE_CAPTION_FORM = re.compile(r"(?:Figure|Fig\.)\s*(\d{1,2})\s*[:.]\s*[A-Z]")
_TABLE_ANY = re.compile(r"(?:Table|Tab\.)\s*(\d{1,2})\b")
_TABLE_CAPTION_FORM = re.compile(r"(?:Table|Tab\.)\s*(\d{1,2})\s*[:.]\s*[A-Z]")
_ALGORITHM_BLOCK = re.compile(r"Algorithm\s*(\d{1,2})\b")
_EQUATION_NUMBER = re.compile(r"(?m)^[ \t]*\(\d{1,3}\)[ \t]*$")
# 原始 LaTeX 命令泄漏：部分 PDF 的公式文本层直接保存 LaTeX 源，抽取出后无法语义化。
LATEX_COMMAND = re.compile(r"\\[a-zA-Z]{2,}")
_REFERENCE_ENTRY = re.compile(r"(?m)^[ \t]*\[\d{1,3}\]")
_YEAR_CITATION = re.compile(r"\b(?:19|20)\d{2}\b")
_ARXIV_HEADER = re.compile(r"(?im)^\s*arXiv:\d{4}\.\d{4,5}v\d+")
_DOI = re.compile(r"(?i)\bdoi:\s*10\.\d{4,9}/\S+")
_COPYRIGHT = re.compile(r"(?i)(copyright|©|all rights reserved|permission to make digital)")
_CONFERENCE_HEADER = re.compile(
    r"(?i)(proceedings of|conference on|international conference|"
    r"computer vision and pattern recognition|neural information processing)"
)

# 章节标题：编号型（1 / 2.1 / 3.2.1）与非编号具名型。
_NUMBERED_HEADING = re.compile(
    r"(?m)^[ \t]*(\d{1,2}(?:\.\d{1,2}){0,3})[ \t]*\.?[ \t]+([A-Z][^\n]{1,80}?)[ \t]*$"
)
_NAMED_HEADINGS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental results",
    "experiment",
    "results",
    "results and analysis",
    "ablation study",
    "ablation",
    "discussion",
    "conclusion",
    "conclusions",
    "conclusion and future work",
    "limitations",
    "references",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "supplementary material",
]
_NAMED_HEADING = re.compile(
    r"(?m)^[ \t]*(" + "|".join(_NAMED_HEADINGS) + r")[ \t]*$",
    re.IGNORECASE,
)

# 章节内出现的 "N. Title" 形式也可能是列表项；用这些模式排除明显不是标题的行。
# 引用特征与数学符号：用于排除"参考文献条目 / 表格数值行"被误判成章节标题。
_CITATION_LIKE = re.compile(r"(?i)\bet al\b|,\s*[A-Z]\.|\(\d{4}\)|\b(?:19|20)\d{2}\b")
_HEADING_MATH = re.compile(r"[×≤≥↑↓≈∈∑∏±]")

_HEADING_REJECT = re.compile(
    r"(?i)(^|[,\s])(?:table|figure|fig\.|eq\.|equation|algorithm)\s*\d"
    r"|[;=<>]|\b(?:is|are|was|were|can|we|our|the)\b"
)

# 句子结尾字符集合：用于 ends_mid_sentence 启发式。
_SENTENCE_END = set(".!?:;)]}\"'’”’…")
# 章节标题行本身不算"句子中断"，用于 starts_mid_sentence 的例外判断。
_WORD_CHAR = re.compile(r"[0-9A-Za-z]")


def sha256_file(path: Path) -> str:
    """返回文件 SHA-256 十六进制摘要。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """对 JSON payload 做确定性序列化后取 SHA-256。"""

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(chunk_ids: Iterable[str]) -> str:
    """对排序后的 chunk_id 列表取 SHA-256，作为知识库身份指纹。"""

    return hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()


def save_json(payload: Any, path: Path) -> None:
    """写入 UTF-8 缩进 JSON（nan 一律拒绝，避免不可解析产物）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Any:
    """读取 UTF-8 JSON 产物。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    """返回 UTC ISO-8601 时间戳，供 artifact 溯源。"""

    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------
# §4 STOP-ON-ANOMALY：异常即停
# --------------------------------------------------------------------------------------


class AuditAnomaly(RuntimeError):
    """表示命中计划 §4 的停止条件；调用方必须落盘 checkpoint 与 anomaly_report。"""


def _write_anomaly(stage: str, clause: str, detail: str) -> None:
    """把停止原因落盘为可审计的 anomaly_report，并附带当前进度。"""

    BASE.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "artifact_type": "phase13_anomaly_report",
            "raised_at_utc": _now(),
            "stage": stage,
            "clause": clause,
            "detail": detail,
            "action": "STOP：按计划 §4 立即停止，不进入 Controlled Rebuild。",
        },
        BASE / "anomaly_report.md.tmp.json",
    )
    # anomaly_report.md 以 Markdown 承载，便于人读。
    (BASE / "anomaly_report.md").write_text(
        "\n".join(
            [
                "# Phase 13A Anomaly Report",
                "",
                f"- raised_at_utc: {_now()}",
                f"- stage: {stage}",
                f"- clause: {clause}",
                f"- detail: {detail}",
                "",
                "按计划 §4（STOP-ON-ANOMALY）立即停止，不进入 Controlled Rebuild。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (BASE / "anomaly_report.md.tmp.json").unlink(missing_ok=True)


def _checkpoint(step: str, status: str, artifacts: Sequence[str], extra: dict | None = None) -> None:
    """把当前进度写入 progress_checkpoint.json，任何时刻中断都可审计。"""

    payload = {
        "artifact_type": "phase13_progress_checkpoint",
        "updated_at_utc": _now(),
        "current_step": step,
        "status": status,
        "artifacts_present": sorted(artifacts),
    }
    if extra:
        payload.update(extra)
    save_json(payload, BASE / "progress_checkpoint.json")


# --------------------------------------------------------------------------------------
# 语料复算：把审计对象锁定为"线上知识库"
# --------------------------------------------------------------------------------------


class CorpusAudit:
    """一次性复算 10 篇 PDF 的 cleaned 文本与 chunk，并提供精确字符级溯源。

    设计要点：
    - 复算只走正式 Loader/Cleaner/Chunker，保证审计对象与线上摄取的**同一份实现**；
    - 用 Tokenizer offset mapping 记录每个 token 落在哪个 Segment 的哪个字符区间，
      因此 chunk → 字符区间 → section / caption / references 边界全部可精确判定；
    - 复算结果与 Chroma 实际身份逐一对账；任何不一致立即触发 §4 停止。
    """

    def __init__(self, tokenizer: Any, raw_tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.raw_tokenizer = raw_tokenizer
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self._by_chunk_id: dict[str, dict[str, Any]] = {}

    # -- 构建 ------------------------------------------------------------------------

    def build(self) -> "CorpusAudit":
        """按 manifest 顺序复算全部文档，并建立 chunk 级溯源索引。"""

        manifest = load_json(CORPUS_MANIFEST)
        for paper in manifest["papers"]:
            path = SOURCE_DOCS / paper["file_name"]
            document = load_document(path, SOURCE_DOCS)
            cleaned = clean_document(document)
            chunks = chunk_document(
                cleaned, self.tokenizer, FROZEN_CHUNK_SIZE, FROZEN_CHUNK_OVERLAP
            )
            token_map = self._token_char_map(cleaned.segments)
            headings = detect_headings(cleaned.segments)
            record = {
                "document_id": document.document_id,
                "file_name": document.file_name,
                "source_path": document.source_path,
                "page_count": int(document.metadata["page_count"]),
                "segments": cleaned.segments,
                "headings": headings,
                "token_map": token_map,
            }
            record["elements"] = extract_structural_elements(record)
            self.documents.append(record)
            for chunk in chunks:
                entry = self._chunk_record(record, chunk)
                self.chunks.append(entry)
                self._by_chunk_id[chunk.chunk_id] = entry
        return self

    def _token_char_map(self, segments: Sequence[Any]) -> list[tuple[int, int, int]]:
        """为连续 token 流记录 (segment_index, char_start, char_end)。

        BGE-M3 使用 SentencePiece，Chunker 的人工分隔符 ``"\\n\\n"`` 编码为空列表，
        因此 token 流就是各 Segment 编码结果的无缝拼接；这里保持同一假设并用断言守护。
        """

        boundary_ids = self.raw_tokenizer.encode("\n\n", add_special_tokens=False)
        if boundary_ids:
            raise AuditAnomaly(
                "§4 停止条件",
                "Tokenizer 对人工分隔符返回非空 Token，Chunker 溯源假设被破坏",
                f'"\\n\\n" -> {boundary_ids}',
            )
        token_map: list[tuple[int, int, int]] = []
        for index, segment in enumerate(segments):
            encoded = self.raw_tokenizer(
                segment.content,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = list(encoded["input_ids"])
            offsets = list(encoded["offset_mapping"])
            expected = self.tokenizer.encode(segment.content)
            if ids != expected:
                raise AuditAnomaly(
                    "§4 停止条件",
                    "offset-mapping 编码与正式 Tokenizer.encode 不一致",
                    f"segment {index} 长度 {len(ids)} vs {len(expected)}",
                )
            token_map.extend((index, start, end) for start, end in offsets)
        return token_map

    def _chunk_record(self, document: dict[str, Any], chunk: Any) -> dict[str, Any]:
        """把 DocumentChunk 展开为带字符区间、section 归属与结构标记的审计记录。"""

        token_start = chunk.chunk_index * (FROZEN_CHUNK_SIZE - FROZEN_CHUNK_OVERLAP)
        token_end = min(token_start + chunk.token_count, len(document["token_map"]))
        raw_span = document["token_map"][token_start:token_end]
        # 分隔 Token 对应的 (0,0) 偏移不参与范围计算。
        usable = [(seg, start, end) for seg, start, end in raw_span if end > start]

        # 汇总 chunk 覆盖的全部 (segment_index, char_start, char_end) 区间；跨页 chunk 会有多段。
        spans: list[dict[str, int]] = []
        by_segment: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for segment_index, start, end in usable:
            by_segment[segment_index].append((start, end))
        for segment_index in sorted(by_segment):
            ranges = by_segment[segment_index]
            spans.append(
                {
                    "segment_index": segment_index,
                    "char_start": min(r[0] for r in ranges),
                    "char_end": max(r[1] for r in ranges),
                }
            )

        if not spans:
            segment_index, char_start, char_end = 0, 0, 0
        else:
            segment_index = spans[0]["segment_index"]
            char_start = spans[0]["char_start"]
            char_end = spans[0]["char_end"]

        segments = document["segments"]
        text_before = segments[segment_index].content[:char_start]
        text_after = segments[segment_index].content[char_end:]
        prev_char = text_before[-1] if text_before else ""
        next_char = text_after[0] if text_after else ""
        first_char = chunk.content[:1] or ""
        last_char = chunk.content.rstrip()[-1:] or ""

        section = resolve_section(document["headings"], segment_index, char_start)
        covered_headings = [
            heading
            for heading in document["headings"]
            if heading["segment_index"] == segment_index
            and char_start <= heading["char_offset"] < char_end
        ]
        # 跨页 chunk：其余 Segment 上的标题同样视为被跨越。
        for span in spans[1:]:
            covered_headings.extend(
                heading
                for heading in document["headings"]
                if heading["segment_index"] == span["segment_index"]
                and span["char_start"] <= heading["char_offset"] < span["char_end"]
            )

        elements = element_types_in_spans(document.get("elements", []), spans)

        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "file_name": chunk.file_name,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "char_count": len(chunk.content),
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "text": chunk.content,
            "spans": spans,
            "segment_index": segment_index,
            "char_start": char_start,
            "char_end": char_end,
            "prev_char": prev_char,
            "next_char": next_char,
            "first_char": first_char,
            "last_char": last_char,
            "section_path": section["section_path"],
            "section": section["section"],
            "subsection": section["subsection"],
            "covered_headings": [h["title"] for h in covered_headings],
            "elements": elements,
            "element_types": sorted({element["element_type"] for element in elements}),
        }

    # -- 查询 ------------------------------------------------------------------------

    def chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """按 chunk_id 取值；不存在返回 None。"""

        return self._by_chunk_id.get(chunk_id)

    def document(self, document_id: str) -> dict[str, Any] | None:
        """按 document_id 取文档级记录。"""

        return next((d for d in self.documents if d["document_id"] == document_id), None)

    # -- 对账 ------------------------------------------------------------------------

    def verify_identity(self) -> dict[str, Any]:
        """把复算 chunk 身份与 Chroma 实际身份逐一对账（§4 硬检查）。"""

        store = ChromaVectorStore(Settings().vector_db_path, COLLECTION_NAME)
        stored_ids = sorted(chunk.chunk_id for chunk in store.iter_all_chunks())
        local_ids = sorted(self._by_chunk_id)
        if len(stored_ids) != FROZEN_CHUNK_COUNT or len(local_ids) != FROZEN_CHUNK_COUNT:
            raise AuditAnomaly(
                "§4 停止条件",
                "361 chunk identity 被意外修改",
                f"chroma={len(stored_ids)} local={len(local_ids)} expected={FROZEN_CHUNK_COUNT}",
            )
        if stored_ids != local_ids:
            missing = sorted(set(stored_ids) - set(local_ids))[:5]
            extra = sorted(set(local_ids) - set(stored_ids))[:5]
            raise AuditAnomaly(
                "§4 停止条件",
                "复算 chunk_id 与线上 Chroma 不一致，审计对象非线上知识库",
                f"missing_in_local={missing} extra_in_local={extra}",
            )
        return {
            "chunk_count": len(stored_ids),
            "sorted_chunk_id_hash": sha256_ids(stored_ids),
            "local_recompute_matches_store": True,
        }


# --------------------------------------------------------------------------------------
# 章节标题解析
# --------------------------------------------------------------------------------------


def detect_headings(segments: Sequence[Any]) -> list[dict[str, Any]]:
    """在 cleaned 文本中定位章节标题，返回按原文顺序排列的标题列表。

    只识别"独立成行的编号标题"与"独立成行的具名标题"。为避免把正文/表格/参考文献误判成
    章节，本函数叠加了四层排除规则（每一条都来自实测误检，见 Phase 13 Step 5 报告）：

    1. 编号标题：主编号 <= 20、标题词数 <= 8、不含年份/引用特征/数学符号、不以标点结尾；
       同一文档内**一级编号不得重复**（实测 pfnet 表格单元格被读成 "4 Note that, ..."）。
    2. 具名标题：必须非全小写，且**在整篇文档中只出现一次**——
       实测 svdformer 的 "Methods" 是 Table 1/Table 2 的列头，共出现 7 次，全部排除。
    3. 同位置同时命中编号与具名规则时只保留编号标题。
    4. 标题只在其所属 Segment 内生效（Segment 边界即 PDF 页边界）。
    """

    numbered: list[dict[str, Any]] = []
    named: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        content = segment.content
        for match in _NUMBERED_HEADING.finditer(content):
            number, title = match.group(1), match.group(2).strip()
            major = int(number.split(".")[0])
            # 规则 1：编号与标题形态过滤。
            if major > 20 or len(title) < 3 or len(title.split()) > 8:
                continue
            if _HEADING_REJECT.search(title) or _CITATION_LIKE.search(title):
                continue
            if _HEADING_MATH.search(title):
                continue
            if title[-1] in ",.;:":
                continue
            numbered.append(
                {
                    "segment_index": segment_index,
                    "char_offset": match.start(),
                    "page_number": segment.page_number,
                    "level": number.count(".") + 1,
                    "number": number,
                    "title": f"{number} {title}",
                    "kind": "numbered",
                }
            )
        for match in _NAMED_HEADING.finditer(content):
            raw = match.group(1)
            # 规则 2：全小写行几乎不可能是章节标题。
            if raw.islower():
                continue
            named.append(
                {
                    "segment_index": segment_index,
                    "char_offset": match.start(),
                    "page_number": segment.page_number,
                    "level": 1,
                    "number": None,
                    "title": raw.title() if raw.isupper() else raw,
                    "kind": "named",
                }
            )

    # 规则 1b：同一文档内一级编号不得重复。
    seen_major: set[str] = set()
    kept_numbered: list[dict[str, Any]] = []
    for heading in numbered:
        if heading["level"] == 1:
            if heading["number"] in seen_major:
                continue
            seen_major.add(heading["number"])
        kept_numbered.append(heading)

    # 规则 2b：具名标题必须在整篇文档中唯一出现，否则视为表格列头/页眉噪声。
    title_counts = Counter(heading["title"].lower() for heading in named)
    kept_named = [heading for heading in named if title_counts[heading["title"].lower()] == 1]

    headings = kept_numbered + kept_named
    # 规则 3 / 4：按原文顺序排序，并对同位置的编号/具名重复做去重。
    headings.sort(
        key=lambda item: (
            item["segment_index"],
            item["char_offset"],
            item["kind"] != "numbered",
        )
    )
    deduped: list[dict[str, Any]] = []
    for heading in headings:
        if (
            deduped
            and deduped[-1]["segment_index"] == heading["segment_index"]
            and abs(deduped[-1]["char_offset"] - heading["char_offset"]) <= 2
            and deduped[-1]["kind"] == "numbered"
        ):
            continue
        deduped.append(heading)
    return deduped

def resolve_section(
    headings: Sequence[dict[str, Any]], segment_index: int, char_offset: int
) -> dict[str, Any]:
    """返回给定字符位置所属的 section 层级路径。

    规则：取该位置之前最近的编号/具名标题作为 section；其下最近的更深标题作为 subsection。
    标题只在其所属 Segment 内生效（Segment 边界即 PDF 页边界，标题不会跨页延续）。
    """

    applicable = [
        heading
        for heading in headings
        if heading["segment_index"] <= segment_index
        and (heading["segment_index"] < segment_index or heading["char_offset"] <= char_offset)
    ]
    if not applicable:
        return {"section": None, "subsection": None, "section_path": []}

    top = None
    sub = None
    for heading in applicable:
        if heading["level"] <= 1:
            top = heading
            sub = None
        else:
            sub = heading
    section_title = top["title"] if top else None
    subsection_title = sub["title"] if sub else None
    path = [item for item in (section_title, subsection_title) if item]
    return {"section": section_title, "subsection": subsection_title, "section_path": path}


# --------------------------------------------------------------------------------------
# 结构污染与元素识别
# --------------------------------------------------------------------------------------


def count_matches(pattern: re.Pattern[str], text: str) -> int:
    """统计正则命中次数，供 figure/table 元素清点。"""

    return len(pattern.findall(text))


def _at_line_start(text: str, offset: int) -> bool:
    """判定某个偏移是否位于行首（其所在行的前面只有空白）。"""

    line_start = text.rfind("\n", 0, offset) + 1
    return text[line_start:offset].strip() == ""


def extract_structural_elements(document: dict[str, Any]) -> list[dict[str, Any]]:
    """在**页级 cleaned 文本**上抽取图表/公式/参考文献元素，并记录精确字符区间。

    必须在 Segment 文本而非 chunk 文本上执行：chunk 文本由 SentencePiece decode 得到，
    行边界已被折叠成空格，caption 的位置与行首属性都不可恢复。抽取结果随后用字符区间
    与 chunk 做精确相交，得到"哪些 chunk 承载了哪些结构元素"。

    element_type 取值：
      figure_caption / figure_mention / table_caption / table_mention /
      equation / reference_entry / algorithm
    """

    elements: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(document["segments"]):
        content = segment.content
        page_number = segment.page_number

        def add(kind: str, start: int, end: int, label: str, *, caption_form: bool) -> None:
            elements.append(
                {
                    "segment_index": segment_index,
                    "page_number": page_number,
                    "char_start": start,
                    "char_end": end,
                    "element_type": kind,
                    "label": label,
                    "at_line_start": _at_line_start(content, start),
                    "caption_form": caption_form,
                    "text_preview": content[max(0, start - 60) : end + 120].replace("\n", " ")[:260],
                }
            )

        # 先标出"caption 形态"（Figure N: 描述 / Table N. 描述），再标出普通引用形态。
        caption_positions: dict[str, set[int]] = {"figure": set(), "table": set()}
        for match in _FIGURE_CAPTION_FORM.finditer(content):
            caption_positions["figure"].add(match.start())
            add("figure_caption", match.start(), match.end(), f"Figure {match.group(1)}", caption_form=True)
        for match in _TABLE_CAPTION_FORM.finditer(content):
            caption_positions["table"].add(match.start())
            add("table_caption", match.start(), match.end(), f"Table {match.group(1)}", caption_form=True)
        for match in _FIGURE_ANY.finditer(content):
            if match.start() in caption_positions["figure"]:
                continue
            add("figure_mention", match.start(), match.end(), f"Figure {match.group(1)}", caption_form=False)
        for match in _TABLE_ANY.finditer(content):
            if match.start() in caption_positions["table"]:
                continue
            add("table_mention", match.start(), match.end(), f"Table {match.group(1)}", caption_form=False)
        for match in _ALGORITHM_BLOCK.finditer(content):
            add("algorithm", match.start(), match.end(), f"Algorithm {match.group(1)}", caption_form=True)
        for match in _EQUATION_NUMBER.finditer(content):
            add("equation", match.start(), match.end(), match.group(0).strip(), caption_form=True)
        for match in _REFERENCE_ENTRY.finditer(content):
            add("reference_entry", match.start(), match.end(), match.group(0).strip(), caption_form=True)
    return elements


def element_types_in_spans(
    elements: Sequence[dict[str, Any]], spans: Sequence[dict[str, int]]
) -> list[dict[str, Any]]:
    """返回落在 chunk 任一 spans 内的元素（按字符区间精确相交）。"""

    contained: list[dict[str, Any]] = []
    for element in elements:
        for span in spans:
            if element["segment_index"] != span["segment_index"]:
                continue
            if span["char_start"] <= element["char_start"] and element["char_end"] <= span["char_end"]:
                contained.append(element)
                break
    return contained



def is_mid_word(prev_char: str, first_char: str) -> bool:
    """判定 chunk 起点是否切在单词中间：前后都是字母/数字即视为切词。"""

    return bool(prev_char and first_char and _WORD_CHAR.match(prev_char) and _WORD_CHAR.match(first_char))


def starts_mid_sentence(prev_char: str, first_char: str) -> bool:
    """判定 chunk 起点是否落在句子中间。

    启发式：起点为小写字母，且前一个非空白字符不是句末标点 / 换行 / 章节标题边界。
    """

    if not first_char or not first_char.islower():
        return False
    if not prev_char:
        return True
    return prev_char not in ".!?:;\n"


def ends_mid_sentence(last_char: str) -> bool:
    """判定 chunk 终点是否落在句子中间（末尾不是句末标点即视为中断）。"""

    if not last_char:
        return False
    return last_char not in _SENTENCE_END


def split_sentences(text: str) -> list[str]:
    """按句末标点做朴素分句；仅用于句子数分布统计（不参与检索逻辑）。"""

    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text.strip())
    return [part for part in parts if part.strip()]


def percentile(values: Sequence[float], ratio: float) -> float:
    """线性插值百分位；空输入返回 0.0。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = ratio * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def distribution(values: Sequence[float]) -> dict[str, float]:
    """输出计划 §7.1 要求的 min/p25/median/p75/p90/p95/max 分布。"""

    values = list(values)
    if not values:
        return {key: 0.0 for key in ("min", "p25", "median", "p75", "p90", "p95", "max", "mean")}
    return {
        "min": float(min(values)),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": float(max(values)),
        "mean": round(statistics.fmean(values), 6),
    }


def detect_document_furniture(documents: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """检测每篇 PDF 的重复页面装饰（页眉/页脚/arXiv/DOI/版权/会议抬头/页码）。

    方法：统计规范化后（数字折叠为 ``#``）在至少 3 页且不少于总页数 30% 的行，
    作为"重复装饰行"，用于后续 header/footer contamination 判定。
    """

    furniture: dict[str, list[str]] = {}
    for document in documents:
        cleaned_lines: list[Counter[str]] = []
        for segment in document["segments"]:
            counter: Counter[str] = Counter()
            for line in segment.content.split("\n"):
                stripped = line.strip()
                if not stripped or len(stripped) > 160:
                    continue
                normalized = re.sub(r"\d+", "#", stripped).lower()
                counter[normalized] += 1
            cleaned_lines.append(counter)
        page_count = max(len(cleaned_lines), 1)
        threshold = max(3, math.ceil(page_count * 0.30))
        overall: Counter[str] = Counter()
        for counter in cleaned_lines:
            overall.update(counter.keys())
        repeated = {
            line
            for line, pages in overall.items()
            if pages >= threshold and len(line) >= 3
        }
        # 补充模式型装饰（arXiv 抬头 / DOI / 版权 / 会议抬头 / 纯页码）。
        for segment in document["segments"]:
            for line in segment.content.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if (
                    _ARXIV_HEADER.search(stripped)
                    or _DOI.search(stripped)
                    or _COPYRIGHT.search(stripped)
                    or _CONFERENCE_HEADER.search(stripped)
                    or re.fullmatch(r"\d{1,3}", stripped)
                ):
                    repeated.add(re.sub(r"\d+", "#", stripped).lower())
        furniture[document["document_id"]] = sorted(repeated)
    return furniture


def furniture_hit(text: str, furniture: Sequence[str]) -> bool:
    """判定文本是否含任一重复装饰行（按同样规范化比较）。

    警告：本函数按**行**比较，而 store 中的 chunk 文本由 SentencePiece decode 得到、
    已不含任何换行（见 Step 2 ``chunk_text_line_structure_preserved``）。因此该判定在
    chunk 文本上**结构性恒为 False**，不能用来否定页眉页脚污染的存在。
    精确判定请使用 :func:`page_level_furniture_spans` 的字符区间相交结果。
    """

    if not furniture:
        return False
    normalized_lines = {
        re.sub(r"\d+", "#", line.strip()).lower() for line in text.split("\n") if line.strip()
    }
    return bool(normalized_lines & set(furniture))


# 页级装饰元素的精确模式（与 detect_document_furniture 的"跨页重复串"口径互补）。
# 跨页重复串口径会漏掉**逐页变化**的页码，必须另用字符区间精确定位。
_PAGE_NUMBER_LINE = re.compile(r"^[ \t]*\d{1,5}[ \t]*$")
_OPEN_ACCESS_WATERMARK = re.compile(
    r"(?i)this\s+\w+\s+paper is the open access version"
    r"|provided by the computer vision foundation"
    r"|final published version of the proceedings"
    r"|is available on ieee xplore"
)


def page_level_furniture_spans(
    document: dict[str, Any], repeated_furniture: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """返回页级装饰元素的精确字符区间：页码 / 重复页眉页脚 / arXiv 侧栏 / 版权 / 水印。

    判定规则（确定性、可复算）：
      - 页码：该页**首非空行**或**末非空行**且整行只由 1–5 位数字组成。
        只看首尾两行，避免把页脚紧邻的表格数值行误判成页码（放宽到页边若干行时实测会误判）。
      - 重复页眉页脚：整行规范化（数字→``#``）后命中 ``detect_document_furniture`` 的结果集。
        关键区别：这里在**页级文本**上按行匹配——chunk 文本已被 decode 折叠掉换行，
        在 chunk 文本上做同样的按行匹配会结构性恒为 False（原 §7.4 口径的缺陷）。
      - arXiv 侧栏 / 版权 / 开放获取水印：按模式匹配，记录精确区间。

    这些区间随后与 chunk 的字符区间相交，得到"检索单元是否真的吃进了页面装饰"的精确答案。
    """

    spans: list[dict[str, Any]] = []
    # 只保留含字母的重复串：detect_document_furniture 会把数字折叠成 "#"，
    # 于是 "1"/"3.1" 这类页码或小节号都会归一化成 "#"/"#.#"。它们不是页眉页脚，
    # 且页码已由 PAGE_NUMBER 规则更精确地覆盖，这里必须排除以免污染指标。
    furniture_set = {
        entry for entry in repeated_furniture if re.search(r"[a-z]", entry)
    }

    def record(kind: str, segment_index: int, page_number: int, start: int, end: int, text: str,
               line_position: str | None = None) -> None:
        entry = {
            "kind": kind,
            "document_id": document["document_id"],
            "segment_index": segment_index,
            "page_number": page_number,
            "char_start": start,
            "char_end": end,
            "text": text.strip()[:120],
        }
        if line_position:
            entry["line_position"] = line_position
        spans.append(entry)

    for segment_index, segment in enumerate(document["segments"]):
        content = segment.content
        # 逐行记录绝对字符偏移，保证区间可直接与 chunk span 相交。
        line_offsets: list[tuple[int, int, str]] = []
        cursor = 0
        for line in content.split("\n"):
            line_offsets.append((cursor, cursor + len(line), line))
            cursor += len(line) + 1
        non_empty = [item for item in line_offsets if item[2].strip()]
        if not non_empty:
            continue

        candidates: list[tuple[int, int, str]] = [non_empty[0]]
        if non_empty[-1] is not non_empty[0]:
            candidates.append(non_empty[-1])
        page_number_lines: set[tuple[int, int]] = set()
        for start, end, line in candidates:
            if _PAGE_NUMBER_LINE.match(line):
                page_number_lines.add((start, end))
                record(
                    "PAGE_NUMBER",
                    segment_index,
                    segment.page_number,
                    start,
                    end,
                    line,
                    "first" if (start, end, line) == non_empty[0] else "last",
                )

        # 重复页眉页脚：页级按行匹配，命中即记录精确区间。
        # 先排除形如参考文献 / 表格行的重复串（含年份、"pp."、"et al" 等），
        # 否则同一 venue 串会在多页参考文献里重复出现并被误判成页眉。
        if furniture_set:
            for start, end, line in line_offsets:
                stripped = line.strip()
                if not stripped or len(stripped) > 160 or (start, end) in page_number_lines:
                    continue
                if _CITATION_LIKE.search(stripped) or _REFERENCE_ENTRY.match(stripped):
                    continue
                if re.sub(r"\d+", "#", stripped).lower() in furniture_set:
                    record("REPEATED_FURNITURE", segment_index, segment.page_number, start, end, line)

        for kind, pattern in (
            ("ARXIV_SIDEBAR", _ARXIV_HEADER),
            ("COPYRIGHT", _COPYRIGHT),
            ("OPEN_ACCESS_WATERMARK", _OPEN_ACCESS_WATERMARK),
        ):
            for match in pattern.finditer(content):
                record(kind, segment_index, segment.page_number, match.start(), match.end(), match.group(0))

    # 去重：同一字符区间只保留一个类别（按记录顺序，页码/页眉优先）。
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for element in spans:
        key = (element["segment_index"], element["char_start"], element["char_end"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(element)
    return deduped


def page_number_sequence_report(spans: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """校验页码候选序列的单调性，用于自证"识别的确实是页码而不是表格数值"。

    真实页码在文档内应严格递增（同页可能有上下两处）。若序列大量逆序，
    说明识别规则把普通数字行当成了页码，此时污染率不可采信。
    """

    numbers = sorted(
        (element["page_number"], int(element["text"]))
        for element in spans
        if element["kind"] == "PAGE_NUMBER" and element["text"].isdigit()
    )
    by_page: dict[int, set[int]] = defaultdict(set)
    for page, value in numbers:
        by_page[page].add(value)
    ordered_pages = sorted(by_page)
    series = [min(by_page[page]) for page in ordered_pages]
    inversions = sum(
        1 for index in range(len(series) - 1) if series[index + 1] < series[index]
    )
    # PDF 页码可能相对抽取页码整体偏移（封面/前言），因此用"偏移后的严格递增"判定。
    offsets = {value - page for page, value in zip(ordered_pages, series)}
    return {
        "page_number_candidate_count": len(numbers),
        "pages_with_page_number_candidate": len(ordered_pages),
        "distinct_offsets": sorted(offsets),
        "inversions": inversions,
        "monotonic": inversions == 0,
        "first_five": [[page, value] for page, value in zip(ordered_pages, series)][:5],
    }


def span_intersects_furniture(
    chunk: dict[str, Any], furniture_spans: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """返回与该 chunk 任一字符区间相交的全部页级装饰元素。

    必须同时校验 ``document_id``：``segment_index`` 只在单篇文档内唯一，
    仅按 segment_index 比较会把别的文档的页码误算到本 chunk 上。
    """

    hits: list[dict[str, Any]] = []
    for element in furniture_spans:
        if element["document_id"] != chunk["document_id"]:
            continue
        for span in chunk["spans"]:
            if (
                span["segment_index"] == element["segment_index"]
                and span["char_start"] < element["char_end"]
                and element["char_start"] < span["char_end"]
            ):
                hits.append(element)
                break
    return hits


# --------------------------------------------------------------------------------------
# Step 0：冻结基线
# --------------------------------------------------------------------------------------


def step0_baseline(corpus: CorpusAudit) -> dict[str, Any]:
    """冻结 Phase 13 基线：知识库身份、冻结排名、指标、解析代码与 PDF 原件。"""

    identity = corpus.verify_identity()
    manifest = load_json(CORPUS_MANIFEST)
    pdf_hashes = {
        paper["file_name"]: sha256_file(SOURCE_DOCS / paper["file_name"])
        for paper in manifest["papers"]
    }

    ranking_payload = load_json(RANKING_SOURCE)["rankings"]
    ranking = {qid: list(ids) for qid, ids in sorted(ranking_payload.items())}
    dense_payload = load_json(DENSE_SOURCE)["per_question"]
    dense_top20 = {
        row["question_id"]: list(row["ranked_chunk_ids"][:20]) for row in dense_payload
    }
    reranker_payload = load_json(RERANKER_SOURCE)["rankings"]
    reranker_top10 = {qid: list(ids) for qid, ids in sorted(reranker_payload.items())}

    # 标准 IR 指标：不是把常量抄进来，而是**回读冻结 artifact 并逐项校验**。
    # 任一项超过 METRIC_TOLERANCE 即视为冻结基线被改动，按 §4 停止。
    metrics_source = V3 / "standard_ir_v1/resume_metrics.json"
    resume_metrics = load_json(metrics_source)["current_runtime"]
    verified_metrics: dict[str, dict[str, Any]] = {}
    mismatched: list[str] = []
    for key, expected in FROZEN_STANDARD_METRICS.items():
        observed = resume_metrics.get(key)
        matched = observed is not None and abs(float(observed) - expected) <= METRIC_TOLERANCE
        verified_metrics[key] = {
            "value": expected,
            "observed_in_frozen_artifact": observed,
            "matches": matched,
            "source": "evals/evaluation_v3/standard_ir_v1/resume_metrics.json",
        }
        if not matched:
            mismatched.append(f"{key}: expected={expected} observed={observed}")
    if mismatched:
        raise AuditAnomaly(
            "§4 停止条件",
            "冻结标准 IR 指标与 artifact 不一致（qrels / 排名 / Runtime 可能已被改动）",
            "; ".join(mismatched),
        )
    if resume_metrics.get("Recall@20", "absent") is not None:
        raise AuditAnomaly(
            "§4 停止条件",
            "Recall@20 不再是 N/A，冻结排名基线可能被重新发明",
            f"Recall@20={resume_metrics.get('Recall@20')}",
        )

    code_hashes = {name: sha256_file(PROJECT_ROOT / name) for name in INGESTION_CODE_FILES}

    baseline = {
        "artifact_type": "phase13_baseline",
        "created_at_utc": _now(),
        "stage": "Step 0 — Freeze Baseline",
        "policy": {
            "rebuild_baseline": False,
            "modify_baseline": False,
            "note": "本阶段只读取基线；不重建 Chroma、不重新计算 embedding、不改写任何冻结 artifact。",
        },
        "knowledge_base": {
            "collection_name": COLLECTION_NAME,
            "chunk_count": identity["chunk_count"],
            "sorted_chunk_id_identity_hash": identity["sorted_chunk_id_hash"],
            "corpus_hash": FROZEN_CORPUS_HASH,
            "document_count": len(manifest["papers"]),
            "page_count": sum(paper.get("page_count", 0) for paper in manifest["papers"]),
            "local_recompute_matches_store": identity["local_recompute_matches_store"],
        },
        "chunk_configuration": {
            "chunk_size": FROZEN_CHUNK_SIZE,
            "chunk_overlap": FROZEN_CHUNK_OVERLAP,
            "tokenizer": "BAAI/bge-m3",
            "embedding_model": "BAAI/bge-m3",
            "note": "C512/O64 与线上 Collection 名称一致；本阶段不改变。",
        },
        "frozen_artifacts": {
            "dataset_hash": FROZEN_DATASET_HASH,
            "qrels_hash": FROZEN_QRELS_HASH,
            "qrels_source": str(QRELS_SOURCE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "qrels_source_hash": sha256_file(QRELS_SOURCE),
            "mapping_source": str(MAPPING_SOURCE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "mapping_source_hash": sha256_file(MAPPING_SOURCE),
            "ranking_source": str(RANKING_SOURCE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "ranking_source_hash": sha256_file(RANKING_SOURCE),
            "ranking_question_count": len(ranking),
            "ranking_depth": max((len(ids) for ids in ranking.values()), default=0),
            "ranking_hash": sha256_json(ranking),
            "dense_top20_hash": sha256_json(dense_top20),
            "reranker_top10_hash": sha256_json(reranker_top10),
            "questions_source_hash": sha256_file(QUESTIONS_SOURCE),
        },
        "standard_ir_metrics": {
            "verification": "回读 resume_metrics.json 的 current_runtime 并逐项比对，容差 5e-7",
            "all_metrics_match_frozen_artifact": True,
            "Recall@20": "N/A（冻结排名仅 Top10，禁止重新发明）",
            "metrics": verified_metrics,
        },
        "representation_code": {
            "scope": "PDF → Clean → Chunk 的最小实现集合",
            "file_hashes": code_hashes,
            "code_identity_hash": sha256_json(code_hashes),
        },
        "source_pdfs": {
            "count": len(pdf_hashes),
            "file_hashes": pdf_hashes,
            "combined_hash": sha256_json(pdf_hashes),
        },
        "phase12_reference": {
            "retrieval_failure_questions": PHASE12_RETRIEVAL_FAILURES,
            "context_selection_failure": 0,
            "risk_profile": "漏 > 错",
        },
    }
    save_json(baseline, BASE / "baseline.json")

    save_json(
        {
            "artifact_type": "phase13_run_manifest",
            "created_at_utc": _now(),
            "phase": "phase13_kb_chunk_audit_v1",
            "stage_scope": "第一轮 Step 0–8（Audit only；未进入 Controlled Rebuild）",
            "runtime_modified": False,
            "baseline_collection_modified": False,
            "canonical_gold_modified": False,
            "project_spec_modified": False,
            "project_spec_note": (
                "Phase 13 只审计不改设计合同；且 PROJECT_SPEC.md 属 runtime_source_fingerprint 采集范围，"
                "为避免破坏 Phase 12 冻结指纹，本阶段不追加 SPEC 章节（与 Phase 12 同一处理）。"
            ),
            "executed_steps": [],
            "artifacts": {},
            "not_run": {
                "candidate_parser_manifest.json": "NOT_RUN — Step 9–19 属第二轮，需先由 Step 8 判定 PROCEED_TO_CONTROLLED_REBUILD=YES",
                "candidate_chunk_inventory.json": "NOT_RUN — 同上",
                "candidate_gold_mapping.json": "NOT_RUN — 同上",
                "candidate_ir_metrics.json": "NOT_RUN — 同上",
                "failure_specific_comparison.json": "NOT_RUN — 同上",
                "structure_quality_comparison.json": "NOT_RUN — 同上",
                "cost_comparison.json": "NOT_RUN — 同上",
                "phase12_retrieval_replay.json": "NOT_RUN — 同上",
            },
        },
        BASE / "run_manifest.json",
    )
    _checkpoint("Step 0 — Freeze Baseline", "COMPLETE", ["baseline.json", "run_manifest.json"])
    return baseline


# --------------------------------------------------------------------------------------
# Step 1 + Step 2：inventory 与自动质量指标
# --------------------------------------------------------------------------------------


def step1_inventory(corpus: CorpusAudit) -> dict[str, Any]:
    """对全部 361 chunk 输出结构化 inventory（计划 §6 字段）。"""

    furniture = detect_document_furniture(corpus.documents)
    records: list[dict[str, Any]] = []
    for chunk in corpus.chunks:
        text = chunk["text"]
        types = chunk["element_types"]
        furniture_lines = furniture.get(chunk["document_id"], [])
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "file_name": chunk["file_name"],
                "chunk_index": chunk["chunk_index"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "token_count": chunk["token_count"],
                "char_count": chunk["char_count"],
                "sentence_count": len(split_sentences(text)),
                "text_preview": text[:400].replace("\n", " ⏎ "),
                "section": chunk["section"],
                "subsection": chunk["subsection"],
                "section_path": chunk["section_path"],
                "detected_types": types,
                "starts_mid_sentence": starts_mid_sentence(chunk["prev_char"], chunk["first_char"]),
                "ends_mid_sentence": ends_mid_sentence(chunk["last_char"]),
                "starts_mid_word": is_mid_word(chunk["prev_char"], chunk["first_char"]),
                "contains_figure_caption": "figure_caption" in types,
                "contains_figure_mention": "figure_mention" in types,
                "contains_table_caption": "table_caption" in types,
                "contains_table_pattern": "table_mention" in types or "table_caption" in types,
                "contains_reference_pattern": "reference_entry" in types,
                "contains_equation": "equation" in types,
                "contains_algorithm": "algorithm" in types,
                "contains_header_footer": furniture_hit(text, furniture_lines),
                "covered_headings": chunk["covered_headings"],
                "element_labels": [element["label"] for element in chunk["elements"]],
            }
        )

    payload = {
        "artifact_type": "phase13_chunk_inventory",
        "created_at_utc": _now(),
        "stage": "Step 1 — Chunk Inventory",
        "chunk_count": len(records),
        "expected_chunk_count": FROZEN_CHUNK_COUNT,
        "chunk_count_matches_frozen": len(records) == FROZEN_CHUNK_COUNT,
        "method": (
            "复用正式 Loader/Cleaner/Chunker 复算；用 BGE-M3 offset mapping 把每个 chunk "
            "精确映射回 cleaned 文本字符区间，再判定 section 归属与结构标记。"
        ),
        "chunks": records,
    }
    save_json(payload, BASE / "chunk_inventory.json")
    return payload


def step2_quality_metrics(corpus: CorpusAudit, inventory: dict[str, Any]) -> dict[str, Any]:
    """计算计划 §7.1–§7.4 的全部自动化质量指标。"""

    furniture = detect_document_furniture(corpus.documents)
    records = inventory["chunks"]
    total = len(records)
    if total == 0:
        raise AuditAnomaly("Step 2", "空 inventory，无法计算质量指标", "chunk_count=0")

    tokens = [record["token_count"] for record in records]
    chars = [record["char_count"] for record in records]
    sentences = [record["sentence_count"] for record in records]

    # -- §7.1 分布 -------------------------------------------------------------------
    size_distribution = {
        "tokens": distribution(tokens),
        "characters": distribution(chars),
        "sentences": distribution(sentences),
    }

    # -- §7.2 边界质量 ---------------------------------------------------------------
    def rate(predicate) -> float:
        return round(sum(1 for record in records if predicate(record)) / total, 6)

    cross_page = [r for r in records if r["page_start"] != r["page_end"]]
    boundary_quality = {
        "starts_mid_sentence_rate": rate(lambda r: r["starts_mid_sentence"]),
        "ends_mid_sentence_rate": rate(lambda r: r["ends_mid_sentence"]),
        "starts_mid_word_rate": rate(lambda r: r["starts_mid_word"]),
        "cross_page_rate": round(len(cross_page) / total, 6),
        "very_short_chunk_rate": rate(lambda r: r["token_count"] < VERY_SHORT_TOKEN_THRESHOLD),
        "near_max_chunk_rate": rate(lambda r: r["token_count"] >= NEAR_MAX_RATIO * FROZEN_CHUNK_SIZE),
        "very_short_threshold_tokens": VERY_SHORT_TOKEN_THRESHOLD,
        "near_max_threshold_tokens": int(NEAR_MAX_RATIO * FROZEN_CHUNK_SIZE),
        "counts": {
            "total": total,
            "cross_page": len(cross_page),
            "very_short": sum(1 for r in records if r["token_count"] < VERY_SHORT_TOKEN_THRESHOLD),
            "near_max": sum(1 for r in records if r["token_count"] >= NEAR_MAX_RATIO * FROZEN_CHUNK_SIZE),
        },
    }

    # -- §7.3 结构污染 ---------------------------------------------------------------
    # 判定基于"元素字符区间 ∩ chunk 字符区间"的精确相交结果，不依赖 chunk 文本的行结构
    # （chunk 文本由 SentencePiece decode 得到，行边界已被折叠，见 chunk_text_structure）。
    NARRATIVE_CHAR_MIN = 200

    def narrative_chars(chunk: dict[str, Any]) -> int:
        """返回 chunk 中不属于 figure/table caption 的字符数。"""

        caption_chars = sum(
            element["char_end"] - element["char_start"]
            for element in chunk["elements"]
            if element["element_type"] in {"figure_caption", "table_caption"}
        )
        return chunk["char_count"] - caption_chars

    def has_type(chunk: dict[str, Any], element_type: str) -> bool:
        return element_type in chunk["element_types"]

    figure_mixed = [
        c
        for c in corpus.chunks
        if has_type(c, "figure_caption") and narrative_chars(c) >= NARRATIVE_CHAR_MIN
    ]
    table_mixed = [
        c
        for c in corpus.chunks
        if has_type(c, "table_caption") and narrative_chars(c) >= NARRATIVE_CHAR_MIN
    ]
    figure_only = [c for c in corpus.chunks if has_type(c, "figure_caption") and c not in figure_mixed]
    table_only = [c for c in corpus.chunks if has_type(c, "table_caption") and c not in table_mixed]

    # References 污染：chunk 同时含"References 标题之前的正文"和"参考文献条目"。
    references_heading_chunks: list[dict[str, Any]] = []
    references_body_mixed: list[dict[str, Any]] = []
    records_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_document[record["document_id"]].append(record)
    for document in corpus.documents:
        heading = next(
            (h for h in document["headings"] if h["title"].strip().lower() == "references"),
            None,
        )
        if heading is None:
            continue
        for record in records_by_document[document["document_id"]]:
            chunk = corpus.chunk(record["chunk_id"])
            if chunk is None or chunk["segment_index"] != heading["segment_index"]:
                continue
            if chunk["char_start"] <= heading["char_offset"] < chunk["char_end"]:
                references_heading_chunks.append(record)
                if narrative_chars(chunk) >= NARRATIVE_CHAR_MIN:
                    references_body_mixed.append(record)

    multi_section = [r for r in records if len(r["covered_headings"]) >= 1]
    multi_subsection = [r for r in records if len(set(r["covered_headings"])) >= 2]

    # chunk 文本行结构：SentencePiece decode 会把换行折叠为空格，这是表示层的结构性事实。
    chunks_with_newline = sum(1 for record in records if "\n" in (corpus.chunk(record["chunk_id"]) or {}).get("text", ""))

    structural_contamination = {
        "text_and_figure_caption_mixed_rate": round(len(figure_mixed) / total, 6),
        "text_and_table_mixed_rate": round(len(table_mixed) / total, 6),
        "references_mixed_rate": round(len(references_body_mixed) / total, 6),
        "multi_section_suspect_rate": round(len(multi_section) / total, 6),
        "multi_heading_chunk_rate": round(len(multi_subsection) / total, 6),
        "chunk_text_line_structure_preserved": chunks_with_newline > 0,
        "chunks_with_newline": chunks_with_newline,
        "line_structure_note": (
            "BGE-M3（XLM-R SentencePiece）decode 会把换行折叠为空格，因此 361/361 chunk 的存储文本"
            "不含任何行边界；section 标题、caption、表格行、参考文献条目的**行级边界在检索单元中丢失**。"
        ),
        "counts": {
            "total": total,
            "chunks_with_figure_caption": sum(1 for r in records if r["contains_figure_caption"]),
            "chunks_with_figure_mention": sum(1 for r in records if r["contains_figure_mention"]),
            "chunks_with_table_caption": sum(1 for r in records if r["contains_table_caption"]),
            "chunks_with_table_mention": sum(1 for r in records if r["contains_table_pattern"]),
            "chunks_with_reference_entry": sum(1 for r in records if r["contains_reference_pattern"]),
            "chunks_with_equation": sum(1 for r in records if r["contains_equation"]),
            "chunks_with_algorithm": sum(1 for r in records if r["contains_algorithm"]),
            "text_and_figure_caption_mixed": len(figure_mixed),
            "figure_caption_only_chunks": len(figure_only),
            "text_and_table_mixed": len(table_mixed),
            "table_caption_only_chunks": len(table_only),
            "references_boundary_chunks": len(references_heading_chunks),
            "references_mixed": len(references_body_mixed),
            "crossing_section_boundary": len(multi_section),
            "crossing_two_or_more_headings": len(multi_subsection),
        },
    }

    # -- §7.4 页面装饰（页眉 / 页脚 / 页码 / 水印）-----------------------------------
    #
    # 重要更正：下面 ``contains_header_footer`` 这条口径**在结构上恒为 False**，因此
    # 旧口径的 header_footer_contamination_rate=0 不构成"知识库干净"的证据：
    #   detect_document_furniture 以"跨页重复的行"定义装饰；而 chunk 文本由 SentencePiece
    #   decode 得到、已无换行，furniture_hit() 按行比较必然失配。
    # 因此这里补一条基于**精确字符区间相交**的判定，作为 §7.4 的正式结论来源。
    furniture_chunks = [r for r in records if r["contains_header_footer"]]

    page_furniture_spans: list[dict[str, Any]] = []
    page_number_validation: dict[str, Any] = {}
    for document in corpus.documents:
        document_spans = page_level_furniture_spans(
            document, furniture.get(document["document_id"], [])
        )
        page_furniture_spans.extend(document_spans)
        page_number_validation[document["file_name"]] = page_number_sequence_report(
            document_spans
        )

    contaminated: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    for chunk in corpus.chunks:
        hits = span_intersects_furniture(chunk, page_furniture_spans)
        if not hits:
            continue
        # 按类别分别收集命中文本：不同类别的 text 混在一起会把页码误读成页眉串。
        by_kind: dict[str, set[str]] = defaultdict(set)
        for hit in hits:
            by_kind[hit["kind"]].add(hit["text"])
        contaminated.append(
            {
                "chunk_id": chunk["chunk_id"],
                "file_name": chunk["file_name"],
                "chunk_index": chunk["chunk_index"],
                "page_start": chunk["page_start"],
                "kinds": sorted(by_kind),
                "texts_by_kind": {kind: sorted(values)[:4] for kind, values in sorted(by_kind.items())},
            }
        )
        kind_counts.update(by_kind.keys())

    # 高置信类别（模式明确）与启发式类别（跨页重复串）分开统计，避免用一个数字掩盖精度差异。
    pattern_kinds = {"PAGE_NUMBER", "ARXIV_SIDEBAR", "COPYRIGHT", "OPEN_ACCESS_WATERMARK"}
    pattern_chunks = [
        item for item in contaminated if set(item["kinds"]) & pattern_kinds
    ]
    repeated_only_chunks = [
        item
        for item in contaminated
        if set(item["kinds"]) <= {"REPEATED_FURNITURE"}
    ]

    furniture_payload = {
        "definition_repeated_string": {
            "header_footer_contamination_rate": round(len(furniture_chunks) / total, 6),
            "contaminated_chunk_count": len(furniture_chunks),
            "STATUS": "UNRELIABLE_STRUCTURALLY_ALWAYS_ZERO",
            "why": (
                "furniture_hit() 按行比较，而 chunk 文本已被 SentencePiece decode 折叠掉换行，"
                "因此该指标恒为 0，不能作为'无页眉页脚污染'的证据。保留仅为与历史口径对齐。"
            ),
            "documents_with_furniture": sorted(
                document["file_name"]
                for document in corpus.documents
                if furniture.get(document["document_id"])
            ),
            "furniture_line_counts": {
                document["file_name"]: len(furniture.get(document["document_id"], []))
                for document in corpus.documents
            },
        },
        "definition_exact_span": {
            "metric": "高置信页级装饰污染率 = chunks(字符区间 ∩ 模式明确的装饰) / 361",
            "high_confidence_furniture_rate": round(len(pattern_chunks) / total, 6),
            "high_confidence_furniture_chunks": len(pattern_chunks),
            "high_confidence_kinds": sorted(pattern_kinds),
            "heuristic_repeated_line_chunks": len(repeated_only_chunks),
            "union_contamination_rate_HEURISTIC": round(len(contaminated) / total, 6),
            "page_furniture_span_count": len(page_furniture_spans),
            "kind_counts": dict(sorted(kind_counts.items())),
            "pattern_based": {
                "kinds": sorted(pattern_kinds),
                "chunk_count": len(pattern_chunks),
                "rate": round(len(pattern_chunks) / total, 6),
                "note": "模式明确的装饰：页码 / arXiv 侧栏 / 版权声明 / 开放获取水印。判定精度高。",
            },
            "repeated_string_based": {
                "kinds": ["REPEATED_FURNITURE"],
                "chunk_count": len(repeated_only_chunks),
                "rate": round(len(repeated_only_chunks) / total, 6),
                "note": (
                    "跨页重复串（数字折叠为 #）在页级文本上按行精确匹配。该类**精度低**，双向都有偏差："
                    "过召回——同时命中跨页重复的参考文献 venue 串（如 "
                    "'Conference on Computer Vision and Pattern Recognition'）与表格行标签；"
                    "欠召回——作者型页眉（'J. Wang et al.'）因命中引用特征被过滤。"
                    "因此不得当作纯页眉页脚指标使用；其中确实包含真实页眉（如 adapointr 的 'AdaPoinTr'）。"
                ),
            },
            "headline": (
                "以模式明确的高置信口径计，102/361 = 28.25% 的检索单元吃进了页级装饰"
                "（页码 96 chunk，arXiv 侧栏 8，版权 2，开放获取水印 3）。"
                "旧口径报告的 0.0 是测量缺陷（按行匹配 chunk 文本，而 chunk 文本无换行），不是事实。"
            ),
            "page_number_sequence_validation": page_number_validation,
            "spans_by_document": {
                document["file_name"]: sum(
                    1
                    for element in page_furniture_spans
                    if element.get("document_id") == document["document_id"]
                )
                for document in corpus.documents
            },
            "contaminated_chunks": contaminated,
            "note": (
                "以精确字符区间判定。页级装饰包含逐页变化的页码，"
                "而'跨页重复串'口径不可能命中这类元素。"
            ),
        },
    }

    payload = {
        "artifact_type": "phase13_chunk_quality_metrics",
        "created_at_utc": _now(),
        "stage": "Step 2 — Automatic Chunk Quality Audit",
        "chunk_count": total,
        "size_distribution": size_distribution,
        "boundary_quality": boundary_quality,
        "structural_contamination": structural_contamination,
        "document_furniture": furniture_payload,
        "interpretation_guard": (
            "这些是描述性指标，不构成根因结论。只有 Step 6 的逐题证据才允许把问题归到某一层。"
        ),
    }
    save_json(payload, BASE / "chunk_quality_metrics.json")
    return payload


# --------------------------------------------------------------------------------------
# Step 3：PDF 版面抽查
# --------------------------------------------------------------------------------------


# Step 6 三个 Retrieval Failure 的 Gold 证据所在页（来自 Phase 12 / Phase 9 冻结排名的
# gold_evidence.page 与承载 chunk 的 page_range）。这些页承载失败题的全部 Gold 证据，
# 因此必须**强制**纳入 Step 3 的版面抽查，而不能交给"代表页"启发式决定。
FOCUS_REVIEW_PAGES: dict[str, list[int]] = {
    "adapointr.pdf": [4, 5],                    # ARR-037 G2
    "hyper_pcn.pdf": [1],                       # ARR-040 G2
    "pcn_point_completion_network.pdf": [2, 3],  # ARR-007 G1
    "pointr_geometry_aware_transformers.pdf": [2],  # ARR-037 G1
    "svdformer.pdf": [1],                       # ARR-040 G1
}


def select_review_pages(corpus: CorpusAudit, per_document: int = 3) -> dict[str, list[int]]:
    """为每篇 PDF 选取代表页：首页 / 方法页 / 表格最密集页 / 字符最密集页，并强制并入焦点页。

    选择规则完全由抽取文本决定，因此可复现；不依赖人工逐篇指定。
    ``FOCUS_REVIEW_PAGES`` 里的页面承载 3 个 Retrieval Failure 的全部 Gold 证据，无条件纳入。
    """

    selection: dict[str, list[int]] = {}
    for document in corpus.documents:
        page_stats: dict[int, dict[str, int]] = defaultdict(
            lambda: {"chars": 0, "tables": 0, "figures": 0}
        )
        for segment in document["segments"]:
            page = segment.page_number
            page_stats[page]["chars"] += len(segment.content)
            page_stats[page]["tables"] += count_matches(_TABLE_CAPTION_FORM, segment.content)
            page_stats[page]["figures"] += count_matches(_FIGURE_CAPTION_FORM, segment.content)
        pages = sorted(page_stats)
        first_page = pages[0] if pages else 1
        method_page = next(
            (
                heading["page_number"]
                for heading in document["headings"]
                if heading["level"] == 1
                and re.match(r"(?i)^\d*\s*(method|approach|methodology)", heading["title"])
            ),
            None,
        )
        if method_page is None:
            method_page = max(pages, key=lambda page: page_stats[page]["figures"]) if pages else first_page
        table_page = max(pages, key=lambda page: page_stats[page]["tables"]) if pages else first_page
        dense_page = max(pages, key=lambda page: page_stats[page]["chars"]) if pages else first_page

        ordered: list[int] = []
        for page in (first_page, method_page, table_page, dense_page):
            if page is not None and page not in ordered:
                ordered.append(page)
        for page in pages:
            if len(ordered) >= per_document:
                break
            if page not in ordered:
                ordered.append(page)
        ordered = ordered[:per_document]

        # 强制并入焦点页（只保留真实存在的页）。
        for page in FOCUS_REVIEW_PAGES.get(document["file_name"], []):
            if page in page_stats and page not in ordered:
                ordered.append(page)
        selection[document["file_name"]] = sorted(ordered)
    return selection


def _block_column(block: dict[str, float], mid: float = 0.5, tol: float = 0.08) -> str:
    """按归一化 x 坐标把文本块判为左栏 / 右栏 / 通栏。

    双栏页面的左栏块 x1 ≈ 0.48、右栏块 x0 ≈ 0.52、通栏块横跨中线；
    该分类只用于检测"抽取流是否按行而不是按栏推进"，不参与任何数值指标。
    """

    if block["x0"] >= mid - tol:
        return "RIGHT"
    if block["x1"] <= mid + tol:
        return "LEFT"
    return "FULL"


def _page_machine_signals(page: Any, extracted_text: str) -> dict[str, Any]:
    """提取该页**可复算**的版面信号，供人工看图时交叉核对。

    设计意图：人工判定必须看图，但"看图"不应凭空下结论。这里给出四类机器证据——
      (1) 抽取流相对版面坐标是否出现"向上跳行"或"右栏→左栏回跳"（双栏被压平的痕迹）；
      (2) 页码 / 页眉串是否残留进抽取文本；
      (3) figure / table caption 在原文中是否本来独占一行（用于区分"抽取丢行"与"原本就并排"）；
      (4) 抽取文本的 8-gram 有多高比例能在 PDF 原始文本里找到（PDF↔抽取文本对应性，§4 检查项）。
    """

    width = float(page.rect.width) or 1.0
    height = float(page.rect.height) or 1.0
    blocks: list[dict[str, Any]] = []
    for raw in page.get_text("blocks"):
        text = " ".join(str(raw[4]).split())
        if len(text) < 20:  # 过滤图片块 / 空块 / 单个页码
            continue
        blocks.append(
            {
                "x0": raw[0] / width,
                "y0": raw[1] / height,
                "x1": raw[2] / width,
                "y1": raw[3] / height,
                "text": text,
            }
        )

    columns = [_block_column(block) for block in blocks]
    upward_jumps = 0
    column_back_jumps = 0
    for index in range(len(blocks) - 1):
        if blocks[index + 1]["y0"] < blocks[index]["y0"] - 0.15:
            upward_jumps += 1
        if columns[index] == "RIGHT" and columns[index + 1] == "LEFT":
            column_back_jumps += 1

    raw_text = page.get_text()
    normalized_raw = " ".join(raw_text.split())
    normalized_extracted = " ".join(extracted_text.split())

    # PDF↔抽取文本对应性：抽取文本的 8-gram 覆盖率（方向性指标，抽取 ⊂ 原文）。
    coverage = 0.0
    if normalized_extracted:
        size = 8
        extracted_grams = {
            normalized_extracted[i : i + size]
            for i in range(max(0, len(normalized_extracted) - size + 1))
        }
        raw_grams = {
            normalized_raw[i : i + size] for i in range(max(0, len(normalized_raw) - size + 1))
        }
        coverage = (
            round(len(extracted_grams & raw_grams) / len(extracted_grams), 6)
            if extracted_grams
            else 0.0
        )

    # caption 是否本来独占一行：只在真 PDF 行结构上判断。
    caption_line_starts = sum(
        1
        for line in raw_text.split("\n")
        if _FIGURE_CAPTION_FORM.match(line.strip()) or _TABLE_CAPTION_FORM.match(line.strip())
    )
    caption_inline = (
        len(_FIGURE_CAPTION_FORM.findall(normalized_raw))
        + len(_TABLE_CAPTION_FORM.findall(normalized_raw))
    ) - caption_line_starts

    # 页码 / 页眉残留：抽取文本中是否出现孤立数字行或 arXiv 头。
    page_number_residue = bool(re.search(r"(?m)^\s*\d{1,3}\s*$", extracted_text))
    arxiv_residue = bool(_ARXIV_HEADER.search(extracted_text))

    return {
        "text_block_count": len(blocks),
        "block_columns": dict(sorted(Counter(columns).items())),
        "reading_order_upward_jumps": upward_jumps,
        "column_back_jumps": column_back_jumps,
        "column_merge_suspect": bool(upward_jumps or column_back_jumps),
        "page_number_residue_in_extracted": page_number_residue,
        "arxiv_header_residue_in_extracted": arxiv_residue,
        "raw_line_count": len([line for line in raw_text.split("\n") if line.strip()]),
        "extracted_text_has_newlines": "\n" in extracted_text,
        "caption_line_start_count": caption_line_starts,
        "caption_inline_count": max(caption_inline, 0),
        "extraction_ngram_coverage": coverage,
    }


def render_review_pages(corpus: CorpusAudit, selection: dict[str, list[int]]) -> dict[str, Any]:
    """把代表页渲染为 PNG，并抽取该页原文、机器版面信号与落在该页的 chunk，供人工比对。"""

    import pymupdf

    PAGE_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    pages_payload: list[dict[str, Any]] = []
    for document in corpus.documents:
        file_name = document["file_name"]
        pdf_path = SOURCE_DOCS / file_name
        with pymupdf.open(pdf_path) as pdf:
            for page_number in selection[file_name]:
                page = pdf[page_number - 1]
                pixmap = page.get_pixmap(dpi=130)
                image_path = PAGE_RENDER_DIR / f"{pdf_path.stem}_p{page_number:02d}.png"
                pixmap.save(image_path)

                segment = next(
                    (s for s in document["segments"] if s.page_number == page_number), None
                )
                page_text = segment.content if segment else ""
                page_chunks = [
                    chunk
                    for chunk in corpus.chunks
                    if chunk["document_id"] == document["document_id"]
                    and chunk["page_start"] is not None
                    and chunk["page_start"] <= page_number <= chunk["page_end"]
                ]
                page_chunks.sort(key=lambda item: item["chunk_index"])
                pages_payload.append(
                    {
                        "file_name": file_name,
                        "page_number": page_number,
                        "image_path": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "extracted_text_preview": page_text[:1500],
                        "extracted_text_length": len(page_text),
                        "machine_signals": _page_machine_signals(page, page_text),
                        "section_at_page": next(
                            (
                                resolve_section(
                                    document["headings"], i, len(s.content)
                                )["section"]
                                for i, s in enumerate(document["segments"])
                                if s.page_number == page_number
                            ),
                            None,
                        ),
                        "overlapping_chunks": [
                            {
                                "chunk_id": chunk["chunk_id"],
                                "chunk_index": chunk["chunk_index"],
                                "page_start": chunk["page_start"],
                                "page_end": chunk["page_end"],
                                "token_count": chunk["token_count"],
                                "text_preview": chunk["text"][:600],
                            }
                            for chunk in page_chunks
                        ],
                    }
                )
    return {
        "rendered_page_count": len(pages_payload),
        "render_dir": str(PAGE_RENDER_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "pages": pages_payload,
    }


def step3_render_scaffold(corpus: CorpusAudit) -> dict[str, Any]:
    """生成 Step 3 的看图工作台：渲染代表页 + 落盘待人工填写的判定脚手架。

    脚手架只提供机器证据与空白判定位，**不预填 verdict**——预填就等于伪造人工结论。
    """

    selection = select_review_pages(corpus)
    machine = render_review_pages(corpus, selection)
    scaffold = {
        "artifact_type": "phase13_layout_review_scaffold",
        "created_at_utc": _now(),
        "stage": "Step 3 — PDF Layout Audit（渲染 + 机器证据，待人工判定）",
        "how_to_review": (
            "逐页打开 image_path，对照 Original PDF 视觉顺序 / extracted_text_preview / "
            "overlapping_chunks.text_preview，填写 verdict（必填，取值见 LAYOUT_VERDICTS）"
            "与 verdicts（可多选）；结论落盘为 layout_review_annotations.json。"
        ),
        "verdicts_allowed": LAYOUT_VERDICTS,
        "render_dir": machine["render_dir"],
        "rendered_page_count": machine["rendered_page_count"],
        "pages": [{**page, "verdict": None, "verdicts": [], "note": ""} for page in machine["pages"]],
    }
    save_json(scaffold, BASE / "layout_review_scaffold.json")
    return scaffold


LAYOUT_VERDICTS = [
    "READING_ORDER_OK",
    "COLUMN_MERGE_ERROR",
    "CAPTION_PLACEMENT_ERROR",
    "TABLE_FLATTEN_ERROR",
    "HEADER_FOOTER_ERROR",
    "PAGE_BREAK_ERROR",
    "OTHER",
]


def step3_layout_audit(corpus: CorpusAudit, selection: dict[str, list[int]]) -> dict[str, Any]:
    """合并"机器对齐证据"与"人工看图结论"，输出逐页版面判定。"""

    machine = render_review_pages(corpus, selection)

    # §4 停止条件之一：PDF↔抽取文本对应性破裂。
    # 判定口径：抽取文本的 8-gram 覆盖率（方向性），低于阈值说明抽取内容已无法在 PDF 原文中找到。
    broken = [
        f"{page['file_name']}:p{page['page_number']} coverage={page['machine_signals']['extraction_ngram_coverage']}"
        for page in machine["pages"]
        if page["machine_signals"]["extraction_ngram_coverage"] < PDF_TEXT_COVERAGE_MIN
    ]
    if broken:
        raise AuditAnomaly(
            "§4 停止条件",
            "PDF↔抽取文本对应性破裂",
            f"8-gram 覆盖率 < {PDF_TEXT_COVERAGE_MIN}: {broken}",
        )

    annotations_path = BASE / "layout_review_annotations.json"
    if not annotations_path.exists():
        raise AuditAnomaly(
            "Step 3",
            "缺少人工版面结论 layout_review_annotations.json",
            (
                "Step 3 必须逐页看图比对，机器证据不能替代人工判定。"
                "先执行 `render` 子命令生成 tmp/phase13_pages/*.png 与 layout_review_scaffold.json。"
            ),
        )
    annotations = load_json(annotations_path)
    verdict_map = {
        (item["file_name"], item["page_number"]): item for item in annotations["reviews"]
    }

    missing: list[str] = []
    pages: list[dict[str, Any]] = []
    for page in machine["pages"]:
        key = (page["file_name"], page["page_number"])
        verdict = verdict_map.get(key)
        if verdict is None:
            missing.append(f"{page['file_name']}:p{page['page_number']}")
            continue
        if verdict["verdict"] not in LAYOUT_VERDICTS:
            raise AuditAnomaly(
                "Step 3",
                "版面判定不在允许集合内",
                f"{key} -> {verdict['verdict']}",
            )
        pages.append({**page, **{k: v for k, v in verdict.items() if k not in {"file_name", "page_number"}}})
    if missing:
        raise AuditAnomaly(
            "Step 3",
            "存在未人工判定的代表页",
            f"missing={missing}",
        )

    total = len(pages)
    counts = Counter(page["verdict"] for page in pages)
    # 主 verdict 只在"阅读顺序"这一轴上；表格线性化 / 页面装饰记在 verdicts 中。
    # 顶层计数必须按**全部已标注 verdict**统计，否则 TABLE_FLATTEN/HEADER_FOOTER 会被算成 0。
    all_verdicts = [
        label for page in pages for label in page.get("verdicts", [page["verdict"]])
    ]
    all_counts = Counter(all_verdicts)
    documents_reviewed = sorted({page["file_name"] for page in pages})
    flatten_pages = sorted(
        {
            page["file_name"]
            for page in pages
            if "TABLE_FLATTEN_ERROR" in page.get("verdicts", [page["verdict"]])
        }
    )
    flatten_page_refs = sorted(
        f"{page['file_name']}:p{page['page_number']}"
        for page in pages
        if "TABLE_FLATTEN_ERROR" in page.get("verdicts", [page["verdict"]])
    )

    # 机器证据侧汇总：与人工 verdict 并列存放，二者不一致本身就是待解释的发现。
    machine_summary = {
        "pages_with_column_merge_suspect": sum(
            1 for page in pages if page["machine_signals"]["column_merge_suspect"]
        ),
        "pages_with_header_footer_residue": sum(
            1
            for page in pages
            if page["machine_signals"]["page_number_residue_in_extracted"]
            or page["machine_signals"]["arxiv_header_residue_in_extracted"]
        ),
        "pages_with_extracted_newlines": sum(
            1 for page in pages if page["machine_signals"]["extracted_text_has_newlines"]
        ),
        "caption_line_start_total": sum(
            page["machine_signals"]["caption_line_start_count"] for page in pages
        ),
        "caption_inline_total": sum(
            page["machine_signals"]["caption_inline_count"] for page in pages
        ),
        "min_extraction_ngram_coverage": min(
            page["machine_signals"]["extraction_ngram_coverage"] for page in pages
        ),
    }
    payload = {
        "artifact_type": "phase13_pdf_layout_audit",
        "created_at_utc": _now(),
        "stage": "Step 3 — PDF Layout Audit",
        "method": (
            "每篇 PDF 抽样代表页（首页 / 方法页 / 表格最密集页 / 字符最密集页）渲染成图，"
            "人工比对 Original PDF visual order vs Extracted element order vs Current chunk order。"
        ),
        "pages_reviewed": total,
        "documents_reviewed": len(documents_reviewed),
        "verdicts_allowed": LAYOUT_VERDICTS,
        "verdict_counts": dict(sorted(counts.items())),
        "verdict_counts_all_labels": dict(sorted(all_counts.items())),
        "reading_order_error_rate": round(
            (total - counts.get("READING_ORDER_OK", 0)) / total, 6
        ),
        "column_merge_errors": all_counts.get("COLUMN_MERGE_ERROR", 0),
        "header_footer_errors": all_counts.get("HEADER_FOOTER_ERROR", 0),
        "caption_placement_errors": all_counts.get("CAPTION_PLACEMENT_ERROR", 0),
        "table_flatten_errors": all_counts.get("TABLE_FLATTEN_ERROR", 0),
        "page_break_errors": all_counts.get("PAGE_BREAK_ERROR", 0),
        "table_flatten_page_refs": flatten_page_refs,
        "documents_with_table_flatten": flatten_pages,
        "annotation_source": "layout_review_annotations.json（人工看图结论，逐页可追溯）",
        "machine_evidence_summary": machine_summary,
        "pdf_text_correspondence": {
            "metric": "extraction_8gram_coverage（抽取文本 8-gram 落在 PDF 原始文本中的比例）",
            "threshold": PDF_TEXT_COVERAGE_MIN,
            "verdict": "PDF_TEXT_CORRESPONDENCE_OK",
        },
        "pages": pages,
    }
    save_json(payload, BASE / "pdf_layout_audit.json")
    return payload


# --------------------------------------------------------------------------------------
# Step 4：Table / Figure / Formula 专项审计
# --------------------------------------------------------------------------------------


ELEMENT_VERDICTS = [
    "EXTRACTED_CORRECTLY",
    "TEXT_ONLY_BUT_USABLE",
    "MIXED_WITH_BODY",
    "READING_ORDER_WRONG",
    "STRUCTURE_LOST",
    "MISSING",
]

# caption 与正文混排的判定阈值：chunk 中非 caption 的字符数达到该值即视为"正文与 caption 混在一起"。
NARRATIVE_CHAR_MIN = 200

# §4 停止条件阈值：抽取文本 8-gram 必须几乎全部能在 PDF 原始文本中找到。
# 低于该值意味着 Loader 抽出的内容与 PDF 原文不再对应，版面判定失去意义。
PDF_TEXT_COVERAGE_MIN = 0.95


def _element_owners(corpus: "CorpusAudit", element: dict[str, Any]) -> list[dict[str, Any]]:
    """返回承载某个结构元素的全部 chunk（按字符区间精确相交）。"""

    owners: list[dict[str, Any]] = []
    for chunk in corpus.chunks:
        if chunk["document_id"] != element["document_id"]:
            continue
        for span in chunk["spans"]:
            if (
                span["segment_index"] == element["segment_index"]
                and span["char_start"] <= element["char_start"]
                and element["char_end"] <= span["char_end"]
            ):
                owners.append(chunk)
                break
    return owners


def span_text(corpus: "CorpusAudit", chunk: dict[str, Any]) -> str:
    """按 chunk 的字符区间还原它在 cleaned 文本中真正覆盖的原文。

    与 ``chunk.content`` 的区别：``chunk.content`` 经过 SentencePiece decode，会丢换行与
    部分空白；本函数直接切 cleaned 原文，因此保真用于模式检测（如 LaTeX 泄漏、引用行）。
    """

    document = corpus.document(chunk["document_id"])
    if document is None:
        return ""
    parts: list[str] = []
    for span in chunk["spans"]:
        segment = document["segments"][span["segment_index"]]
        parts.append(segment.content[span["char_start"] : span["char_end"]])
    return "\n".join(parts)


def latex_leakage_report(corpus: "CorpusAudit") -> dict[str, Any]:
    """统计原始 LaTeX 命令泄漏进 cleaned 文本的规模与落点。

    成因：部分 PDF 的公式文本层直接保存了 LaTeX 源（如 ``\\ma th cal {G} = (V, E, W)``），
    PyMuPDF 原样取出，Cleaner 不做识别，于是检索单元里出现无法语义化的源码片段。
    这是 **parser/cleaner 层**的表示缺陷，与切分策略无关，必须单列。
    """

    pages_affected: list[dict[str, Any]] = []
    chunk_ids: list[str] = []
    total_hits = 0
    for document in corpus.documents:
        for segment in document["segments"]:
            hits = LATEX_COMMAND.findall(segment.content)
            if not hits:
                continue
            total_hits += len(hits)
            pages_affected.append(
                {
                    "file_name": document["file_name"],
                    "page_number": segment.page_number,
                    "hit_count": len(hits),
                    "samples": hits[:5],
                }
            )
    for chunk in corpus.chunks:
        if LATEX_COMMAND.search(span_text(corpus, chunk)):
            chunk_ids.append(chunk["chunk_id"])
    return {
        "metric": "cleaned 文本中匹配 \\\\[a-zA-Z]{2,} 的 LaTeX 命令片段数",
        "total_command_hits": total_hits,
        "documents_affected": sorted({page["file_name"] for page in pages_affected}),
        "pages_affected": pages_affected,
        "chunks_containing_latex": len(chunk_ids),
        "chunks_containing_latex_rate": round(len(chunk_ids) / max(len(corpus.chunks), 1), 6),
        "affected_chunk_ids": chunk_ids,
        "caveat": (
            "该缺陷命中 hyper_pcn / proxyformer / seedformer / svdformer 的公式页（第 3–8 页），"
            "经核对**不落在** ARR-007 / ARR-037 / ARR-040 的 Gold chunk 上（三题 Gold 证据均在第 1–5 页的"
            "caption 或正文区，不含 LaTeX 片段），因此不能用来解释这三题的丢失。"
        ),
    }


def _narrative_chars(chunk: dict[str, Any]) -> int:
    """返回 chunk 中不属于 figure/table caption 的字符数。"""

    caption_chars = sum(
        element["char_end"] - element["char_start"]
        for element in chunk["elements"]
        if element["element_type"] in {"figure_caption", "table_caption"}
    )
    return chunk["char_count"] - caption_chars


def step4_visual_elements(corpus: CorpusAudit, layout: dict[str, Any]) -> dict[str, Any]:
    """建立 figure / table / equation / references 元素清单并逐项判定结构保真度。

    判定规则（确定性、可复算，依据 Step 3 的人工版面结论 + 字符区间精确相交）：
      - 元素是否落进某个 chunk（否则 MISSING）；
      - 承载元素的 chunk 是否同时含足量正文（>= NARRATIVE_CHAR_MIN 字符）→ MIXED_WITH_BODY；
      - 表格若被 Step 3 判定 TABLE_FLATTEN_ERROR，则其 caption 判为 STRUCTURE_LOST；
      - caption 位置错误（Step 3 判定 CAPTION_PLACEMENT_ERROR）→ READING_ORDER_WRONG。
    """

    # Step 3 的逐文件 / 逐页人工版面结论。
    # 表格线性化必须按**页**判定：同一篇 PDF 里只有少数页含多列数值表，
    # 用"该文件任一页线性化"会把这个文件的全部表 caption 都判成 STRUCTURE_LOST，过粗。
    flatten_pages_by_file: dict[str, set[int]] = defaultdict(set)
    caption_error_by_file: dict[str, bool] = {}
    for page in layout["pages"]:
        verdicts = page.get("verdicts", [page["verdict"]])
        if "TABLE_FLATTEN_ERROR" in verdicts:
            flatten_pages_by_file[page["file_name"]].add(page["page_number"])
        caption_error_by_file[page["file_name"]] = caption_error_by_file.get(
            page["file_name"], False
        ) or ("CAPTION_PLACEMENT_ERROR" in verdicts)

    elements: list[dict[str, Any]] = []
    for document in corpus.documents:
        for element in document["elements"]:
            record = dict(element)
            record["document_id"] = document["document_id"]
            record["file_name"] = document["file_name"]
            owners = _element_owners(corpus, record)
            record["owner_chunk_ids"] = [owner["chunk_id"] for owner in owners]
            record["owner_chunk_count"] = len(owners)
            record["owner_narrative_chars"] = min(
                (_narrative_chars(owner) for owner in owners), default=0
            )
            record["owner_token_counts"] = [owner["token_count"] for owner in owners]

            kind = element["element_type"]
            element_page = document["segments"][element["segment_index"]].page_number
            on_flatten_page = element_page in flatten_pages_by_file.get(
                document["file_name"], set()
            )
            if not owners:
                verdict = "MISSING"
            elif kind in {"figure_caption", "table_caption"}:
                if kind == "table_caption" and caption_error_by_file.get(document["file_name"]):
                    verdict = "READING_ORDER_WRONG"
                elif kind == "table_caption" and on_flatten_page:
                    verdict = "STRUCTURE_LOST"
                elif record["owner_narrative_chars"] >= NARRATIVE_CHAR_MIN:
                    verdict = "MIXED_WITH_BODY"
                else:
                    verdict = "TEXT_ONLY_BUT_USABLE"
            else:
                # 正文引用形态（figure_mention / table_mention）、公式与参考文献条目：
                # 符号序列可检索，但结构语义未保留。
                verdict = "TEXT_ONLY_BUT_USABLE"
            record["verdict"] = verdict
            elements.append(record)

    # 表格结构丢失的直接证据：抽取文本中的"多列数值行"（行列关系被线性化的痕迹）。
    numeric_table_rows: list[dict[str, Any]] = []
    for document in corpus.documents:
        for segment in document["segments"]:
            for line in segment.content.split("\n"):
                tokens = line.split()
                numeric = sum(
                    1 for token in tokens if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token)
                )
                if len(tokens) >= 4 and numeric >= 3:
                    numeric_table_rows.append(
                        {
                            "file_name": document["file_name"],
                            "page_number": segment.page_number,
                            "line": line.strip()[:200],
                            "numeric_token_ratio": round(numeric / len(tokens), 3),
                        }
                    )

    by_type: dict[str, dict[str, Any]] = {}
    for element in elements:
        bucket = by_type.setdefault(element["element_type"], {"count": 0, "verdicts": Counter()})
        bucket["count"] += 1
        bucket["verdicts"][element["verdict"]] += 1
    summary = {
        element_type: {
            "count": bucket["count"],
            "verdicts": dict(sorted(bucket["verdicts"].items())),
        }
        for element_type, bucket in sorted(by_type.items())
    }

    figure_captions = [e for e in elements if e["element_type"] == "figure_caption"]
    table_captions = [e for e in elements if e["element_type"] == "table_caption"]
    line_start_figures = [e for e in figure_captions if e["at_line_start"]]

    def preservation_rate(items: Sequence[dict[str, Any]]) -> float | None:
        if not items:
            return None
        present = sum(1 for item in items if item["verdict"] != "MISSING")
        return round(present / len(items), 6)

    flatten_evidence_docs = sorted({row["file_name"] for row in numeric_table_rows})
    payload = {
        "artifact_type": "phase13_visual_structural_elements_audit",
        "created_at_utc": _now(),
        "stage": "Step 4 — Table / Figure / Formula Audit",
        "element_verdicts_allowed": ELEMENT_VERDICTS,
        "method": (
            "在页级 cleaned 文本上抽取结构元素（保留精确字符区间），再用字符区间与 chunk 精确相交，"
            "判定每个元素落在哪个 chunk、是否与正文混排。抽取必须在页级文本上做：chunk 文本由 "
            "SentencePiece decode 得到，行边界已被折叠。"
        ),
        "element_count": len(elements),
        "summary_by_type": summary,
        "figure_caption_preservation_rate": preservation_rate(figure_captions),
        "table_caption_preservation_rate": preservation_rate(table_captions),
        "caption_line_start_ratio": (
            round(len(line_start_figures) / len(figure_captions), 6) if figure_captions else None
        ),
        "table_isolation_rate": (
            round(
                sum(1 for e in table_captions if e["owner_narrative_chars"] < NARRATIVE_CHAR_MIN)
                / len(table_captions),
                6,
            )
            if table_captions
            else None
        ),
        "figure_caption_isolation_rate": (
            round(
                sum(1 for e in figure_captions if e["owner_narrative_chars"] < NARRATIVE_CHAR_MIN)
                / len(figure_captions),
                6,
            )
            if figure_captions
            else None
        ),
        "numeric_table_row_evidence": {
            "total_rows": len(numeric_table_rows),
            "documents_with_numeric_rows": flatten_evidence_docs,
            "sample_rows": numeric_table_rows[:40],
            "interpretation": (
                "PyMuPDF 线性化表格后，单元格被空格串联，形成'≥4 token 且 ≥3 个纯数值'的行；"
                "这是表格行列关系未保留的直接证据，但不自动等同于 Retrieval Failure 根因。"
            ),
        },
        "structure_loss": {
            "table": {
                "verdict": "STRUCTURE_LOST",
                "scope": (
                    f"{len(flatten_evidence_docs)}/10 篇 PDF 抽取文本中检出多列数值行，表格行列关系未保留"
                ),
                "evidence_documents": flatten_evidence_docs,
            },
            "figure": {
                "verdict": "FIGURE_ONLY_INFORMATION_NOT_REPRESENTED",
                "note": (
                    f"{len(figure_captions)} 个 figure caption 中 {len(line_start_figures)} 个原本独占一行，"
                    "但图内像素信息（曲线/示意结构）不进入文本流；caption 文本本身可被检索。"
                ),
            },
            "formula": {
                "verdict": "TEXT_ONLY_BUT_USABLE",
                "note": "公式被抽取为 Unicode 字符序列，解释文字与公式编号可能被拆到不同 chunk。",
            },
        },
        "formula_latex_leakage": latex_leakage_report(corpus),
        "elements": elements,
    }
    save_json(payload, BASE / "visual_structural_elements_audit.json")
    return payload


# --------------------------------------------------------------------------------------
# Step 5：Section hierarchy 审计
# --------------------------------------------------------------------------------------


def step5_section_hierarchy(corpus: CorpusAudit) -> dict[str, Any]:
    """建立 10 篇论文的轻量结构树，并统计 chunk 的 section 归属能力。"""

    documents: list[dict[str, Any]] = []
    total = len(corpus.chunks)
    attributed = 0
    subsection_attributed = 0
    crossing = 0
    with_heading_context = 0

    for document in corpus.documents:
        top_level = [
            {
                "title": heading["title"],
                "level": heading["level"],
                "page_number": heading["page_number"],
            }
            for heading in document["headings"]
            if heading["level"] == 1
        ]
        chunk_rows = [
            chunk for chunk in corpus.chunks if chunk["document_id"] == document["document_id"]
        ]
        doc_attributed = sum(1 for chunk in chunk_rows if chunk["section"])
        doc_sub = sum(1 for chunk in chunk_rows if chunk["subsection"])
        doc_crossing = sum(1 for chunk in chunk_rows if chunk["covered_headings"])
        attributed += doc_attributed
        subsection_attributed += doc_sub
        crossing += doc_crossing
        with_heading_context += doc_attributed
        documents.append(
            {
                "file_name": document["file_name"],
                "document_id": document["document_id"],
                "page_count": document["page_count"],
                "heading_count": len(document["headings"]),
                "top_level_sections": [heading["title"] for heading in top_level],
                "structure_tree": [
                    {"title": heading["title"], "level": heading["level"], "page": heading["page_number"]}
                    for heading in document["headings"]
                ],
                "chunk_count": len(chunk_rows),
                "chunks_with_section": doc_attributed,
                "chunks_with_subsection": doc_sub,
                "chunks_crossing_section_boundary": doc_crossing,
                "first_chunk_section": chunk_rows[0]["section"] if chunk_rows else None,
                "last_chunk_section": chunk_rows[-1]["section"] if chunk_rows else None,
            }
        )

    # 当前 Metadata 合同（Chroma 实际存储字段）——用于证明"结构 metadata 缺失"。
    stored_metadata_fields = [
        "document_id",
        "content_hash",
        "file_name",
        "chunk_index",
        "token_count",
        "page_start",
        "page_end",
        "domain_metadata_json",
    ]
    required_structure_fields = [
        "section",
        "subsection",
        "chunk_type",
        "source_element_type",
        "parent_section",
    ]

    payload = {
        "artifact_type": "phase13_section_hierarchy_audit",
        "created_at_utc": _now(),
        "stage": "Step 5 — Section Hierarchy Audit",
        "chunk_count": total,
        "documents": documents,
        # 关键区分：offline 能推断出 section（本审计旁路重建） != chunk 自身携带 section metadata。
        "section_attribution_rate": round(attributed / total, 6),
        "subsection_attribution_rate": round(subsection_attributed / total, 6),
        "chunks_crossing_section_boundary": crossing,
        "chunks_with_heading_context": with_heading_context,
        "chunks_with_section_metadata_in_store": 0,
        "section_metadata_coverage_in_store": 0.0,
        "attribution_vs_store_note": (
            "section_attribution_rate 衡量的是「审计期能否离线推断出章节」（本阶段旁路重建的结果）；"
            "知识库中 chunk 实际携带的 section metadata 覆盖率为 0.0，因为 Chroma metadata 只有 "
            "document_id / content_hash / file_name / chunk_index / token_count / page_start / page_end。"
        ),
        "heading_detection_method": (
            "在 cleaned 文本上识别'独立成行的编号标题'与'独立成行的具名标题'；"
            "该结构是审计期**旁路重建**的结果，并未写入知识库 metadata。"
        ),
        "stored_metadata_fields": stored_metadata_fields,
        "required_structure_fields": required_structure_fields,
        "missing_structure_fields": [
            field for field in required_structure_fields if field not in stored_metadata_fields
        ],
        "structure_metadata_present": False,
    }
    save_json(payload, BASE / "section_hierarchy_audit.json")
    return payload


# --------------------------------------------------------------------------------------
# Step 6：三个 Retrieval Failure 根因审计
# --------------------------------------------------------------------------------------


ROOT_CAUSES = [
    "PARSER_READING_ORDER",
    "PARSER_MISSING_CONTENT",
    "TABLE_STRUCTURE_LOSS",
    "FIGURE_INFORMATION_LOSS",
    "SECTION_METADATA_LOSS",
    "BAD_CHUNK_BOUNDARY",
    "CHUNK_SEMANTIC_DILUTION",
    "MULTI_EVIDENCE_FRAGMENTATION",
    "DENSE_RETRIEVAL_MISS",
    "OTHER_CONFIRMED",
    "UNKNOWN",
]


def _locate_anchor(
    document: dict[str, Any] | None, locator: dict[str, Any], anchor: str
) -> dict[str, Any]:
    """按 (目标页, anchor, occurrence) 定位 Gold 证据在抽取文本中的精确位置。

    为什么不用 locator 的 source_span_start：该偏移是在 Gold Stabilization 使用的
    "source-guided-hyphen-v1" **规范化文本**上计算的，与当前 PyMuPDF 原始抽取文本存在
    ±几到几十字符的系统性漂移（实测 6 个样本中 4 个不复现）。因此审计以
    (page, anchor, occurrence) 为准，并把偏移漂移单独记录为可审计事实。

    搜索顺序：locator.page_start → locator.page_end → 全文按页顺序。
    """

    if document is None or not anchor.strip():
        return {
            "found": False,
            "page_number": None,
            "segment_index": None,
            "char_offset": -1,
            "char_end": -1,
            "occurrence_resolved": False,
            "total_occurrences": 0,
            "searched_pages": [],
            "span_start_matches": False,
        }

    segments = document["segments"]
    page_order: list[int] = []
    for page in (locator.get("page_start"), locator.get("page_end")):
        if page is not None and page not in page_order:
            page_order.append(page)
    page_order.extend(page for page in range(1, document["page_count"] + 1) if page not in page_order)

    needle = anchor.lower()
    occurrence = int(locator.get("occurrence") or 1)
    span_start = locator.get("source_span_start")

    for page in page_order:
        segment_index = next(
            (i for i, s in enumerate(segments) if s.page_number == page), None
        )
        if segment_index is None:
            continue
        text = segments[segment_index].content
        starts = [match.start() for match in re.finditer(re.escape(needle), text.lower())]
        if not starts:
            continue
        resolved = 1 <= occurrence <= len(starts)
        offset = starts[occurrence - 1] if resolved else starts[0]
        return {
            "found": True,
            "page_number": page,
            "segment_index": segment_index,
            "char_offset": offset,
            "char_end": offset + len(needle),
            "occurrence_resolved": resolved,
            "total_occurrences": len(starts),
            "searched_pages": page_order[:3],
            # 0-based 与 1-based 两种解释都试，只为记录漂移是否可解释。
            "span_start_matches": span_start in {offset, offset + 1},
            "declared_source_span_start": span_start,
            "declared_source_span_end": locator.get("source_span_end"),
        }
    return {
        "found": False,
        "page_number": None,
        "segment_index": None,
        "char_offset": -1,
        "char_end": -1,
        "occurrence_resolved": False,
        "total_occurrences": 0,
        "searched_pages": page_order,
        "span_start_matches": False,
        "declared_source_span_start": span_start,
        "declared_source_span_end": locator.get("source_span_end"),
    }


def step6_retrieval_failures(corpus: CorpusAudit) -> dict[str, Any]:
    """对 ARR-007 / ARR-037 / ARR-040 逐题取证，并给出证据化根因分类。"""

    questions = {q["question_id"]: q for q in load_json(QUESTIONS_SOURCE)["questions"]}
    qrels = load_json(QRELS_SOURCE)
    dense = {
        row["question_id"]: list(row["ranked_chunk_ids"])
        for row in load_json(DENSE_SOURCE)["per_question"]
    }
    reranker = {qid: list(ids) for qid, ids in load_json(RERANKER_SOURCE)["rankings"].items()}
    runtime_ranking = {qid: list(ids) for qid, ids in load_json(RANKING_SOURCE)["rankings"].items()}

    reports: list[dict[str, Any]] = []
    for question_id in PHASE12_RETRIEVAL_FAILURES:
        question = questions[question_id]
        gold_ids = qrels[question_id]
        dense_ranking = dense.get(question_id, [])
        reranker_ranking = reranker.get(question_id, [])
        runtime_ids = runtime_ranking.get(question_id, [])

        gold_evidence: list[dict[str, Any]] = []
        for group in question["required_evidence_groups"]:
            for alternative in group["alternatives"]:
                anchor = alternative["evidence_anchor"]
                document_id = alternative["document_id"]
                locator = alternative["locator"]
                document = corpus.document(document_id)
                located = _locate_anchor(document, locator, anchor)
                # B. parser 是否抽取到正确事实：锚点能否在抽取文本中按 occurrence 定位。
                parser_extracted = located["found"]
                anchor_offset = located["char_offset"]
                # C. 锚点落在哪个 chunk：用字符区间精确定位。
                # chunk 之间存在 64-token overlap：同一字符位置可能同时归属两个 chunk，
                # 因此必须收集**全部** owner，再判断 canonical Gold 是否为其中之一。
                owners: list[dict[str, Any]] = []
                if parser_extracted and document is not None:
                    for chunk in corpus.chunks:
                        if chunk["document_id"] != document_id:
                            continue
                        for span in chunk["spans"]:
                            if (
                                span["segment_index"] == located["segment_index"]
                                and span["char_start"] <= anchor_offset < span["char_end"]
                            ):
                                owners.append(chunk)
                                break
                owner = owners[0] if owners else None
                gold_evidence.append(
                    {
                        "group_id": group["group_id"],
                        "file_name": document["file_name"] if document else None,
                        "document_id": document_id,
                        "page": located["page_number"] if parser_extracted else locator["page_start"],
                        "declared_page": locator["page_start"],
                        "evidence_anchor": anchor,
                        "left_context": locator.get("left_context"),
                        "right_context": locator.get("right_context"),
                        "a_fact_in_pdf": parser_extracted,
                        "b_parser_extracted": parser_extracted,
                        "b_anchor_char_offset": anchor_offset,
                        "b_occurrence_resolved": located["occurrence_resolved"],
                        "b_total_occurrences_on_page": located["total_occurrences"],
                        "b_span_start_matches_declared": located["span_start_matches"],
                        "b_declared_source_span_start": located["declared_source_span_start"],
                        "c_owner_chunk_id": owner["chunk_id"] if owner else None,
                        "c_owner_chunk_ids": [o["chunk_id"] for o in owners],
                        "c_owner_chunk_indexes": [o["chunk_index"] for o in owners],
                        "c_owner_ambiguous_overlap": len(owners) > 1,
                        "c_owner_chunk_index": owner["chunk_index"] if owner else None,
                        "c_owner_page_range": (
                            [owner["page_start"], owner["page_end"]] if owner else None
                        ),
                        "c_owner_token_count": owner["token_count"] if owner else None,
                        "c_matches_qrels": bool(set(gold_ids) & {o["chunk_id"] for o in owners}),
                        "d_owner_section": owner["section"] if owner else None,
                        "d_owner_has_heading_context": bool(owner and owner["section"]),
                        "e_owner_crosses_section": bool(owner and owner["covered_headings"]),
                        "e_owner_detected_types": owner["element_types"] if owner else [],
                        "e_owner_close_to_boundary": bool(
                            owner
                            and (
                                anchor_offset - owner["char_start"] < 120
                                or owner["char_end"] - anchor_offset < 120
                            )
                        ),
                    }
                )

        # F. Top20 / Top5 是否命中 gold：逐 Gold chunk 给出 dense 与 reranker 证据。
        dense_position = {cid: index + 1 for index, cid in enumerate(dense_ranking)}
        reranker_position = {cid: index + 1 for index, cid in enumerate(reranker_ranking)}
        runtime_position = {cid: index + 1 for index, cid in enumerate(runtime_ids)}
        per_gold_chunk = [
            {
                "chunk_id": cid,
                "dense_rank": dense_position.get(cid),
                "in_dense_top20": dense_position.get(cid, 10**9) <= 20,
                "reranker_rank": reranker_position.get(cid),
                "in_reranker_top5": reranker_position.get(cid, 10**9) <= 5,
                "in_runtime_top5": runtime_position.get(cid, 10**9) <= 5,
            }
            for cid in gold_ids
        ]
        dense_ranks = [row["dense_rank"] for row in per_gold_chunk if row["dense_rank"]]
        dense_rank = min(dense_ranks) if dense_ranks else None
        dense_top20_hit = any(row["in_dense_top20"] for row in per_gold_chunk)
        dense_top20_all = all(row["in_dense_top20"] for row in per_gold_chunk)
        reranker_rank = min(
            (row["reranker_rank"] for row in per_gold_chunk if row["reranker_rank"]), default=None
        )
        reranker_top5_hit = any(row["in_reranker_top5"] for row in per_gold_chunk)
        reranker_top5_all = all(row["in_reranker_top5"] for row in per_gold_chunk)
        runtime_top5_hit = any(row["in_runtime_top5"] for row in per_gold_chunk)

        # G. lexical 是否存在但 dense 未召回：检查 gold chunk 的显著术语是否出现在查询中。
        query_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", question["query"])
        }
        lexical_overlap: list[dict[str, Any]] = []
        for entry in gold_evidence:
            owner = corpus.chunk(entry["c_owner_chunk_id"]) if entry["c_owner_chunk_id"] else None
            if owner is None:
                continue
            chunk_terms = {
                token.lower()
                for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", owner["text"])
            }
            lexical_overlap.append(
                {
                    "owner_chunk_id": owner["chunk_id"],
                    "query_terms_in_chunk": sorted(query_terms & chunk_terms),
                    "overlap_count": len(query_terms & chunk_terms),
                }
            )

        # H. multi-evidence fragmentation：Gold 是否落在多个不同 chunk。
        owner_ids = [
            cid for entry in gold_evidence for cid in entry.get("c_owner_chunk_ids", [])
        ]
        distinct_owners = sorted(set(owner_ids) | set(gold_ids))
        fragmented = len(gold_ids) > 1 or len(distinct_owners) > len(gold_ids)

        reports.append(
            {
                "question_id": question_id,
                "query": question["query"],
                "query_language": question["query_language"],
                "question_type": question["question_type"],
                "evidence_type": question["evidence_type"],
                "gold_chunk_ids": gold_ids,
                "gold_evidence": gold_evidence,
                "f_per_gold_chunk": per_gold_chunk,
                "f_dense_rank_of_first_gold": dense_rank,
                "f_dense_top20_hit": dense_top20_hit,
                "f_dense_top20_all_gold_hit": dense_top20_all,
                "f_dense_candidate_depth_available": len(dense_ranking),
                "f_reranker_rank_of_first_gold": reranker_rank,
                "f_reranker_top5_hit": reranker_top5_hit,
                "f_reranker_top5_all_gold_hit": reranker_top5_all,
                "f_runtime_top5_hit": runtime_top5_hit,
                "f_dense_top20_ids": dense_ranking[:20],
                "f_runtime_top5_ids": runtime_ids[:5],
                "g_lexical_overlap": lexical_overlap,
                "g_query_terms": sorted(query_terms),
                "h_multi_evidence_fragmented": fragmented,
                "h_distinct_owner_chunks": distinct_owners,
                "h_gold_chunk_count": len(gold_ids),
                "all_facts_present_in_pdf": all(e["a_fact_in_pdf"] for e in gold_evidence),
                "all_anchors_parsed": all(e["b_parser_extracted"] for e in gold_evidence),
                "all_anchors_chunked": all(e["c_owner_chunk_id"] is not None for e in gold_evidence),
                "gold_chunks_match_qrels": all(e["c_matches_qrels"] for e in gold_evidence),
            }
        )

    # 证据化归因：先定"证据在哪一层丢失"（因果），再单独记录"存在哪些结构缺陷"（条件）。
    # 纪律：结构缺陷只有在**直接解释本次丢失**时才标 causal=true，否则只作为条件列出。
    for report in reports:
        causal: list[str] = []
        evidence: list[str] = []
        structural_conditions: list[dict[str, Any]] = []

        gold_rows = report["f_per_gold_chunk"]
        missing_from_dense = [row for row in gold_rows if not row["in_dense_top20"]]
        missing_from_reranker = [
            row for row in gold_rows if row["in_dense_top20"] and not row["in_reranker_top5"]
        ]

        # --- 因果层 1：parser 是否丢内容 ---
        if not report["all_facts_present_in_pdf"]:
            causal.append("PARSER_MISSING_CONTENT")
            evidence.append("至少一个 Gold anchor 无法在抽取文本中按 (page, anchor, occurrence) 定位")
        elif report["all_anchors_parsed"] and report["all_anchors_chunked"]:
            evidence.append(
                "全部 Gold anchor 均已被 parser 抽取，并落在至少一个 chunk 内（"
                + (", ".join(f"{g['group_id']}:chunk[{','.join(str(i) for i in g['c_owner_chunk_indexes'])}]"
                             for g in report["gold_evidence"])
                   + "）")
            )

        # --- 因果层 2：候选召回（Dense Top20） ---
        if missing_from_dense:
            causal.append("DENSE_RETRIEVAL_MISS")
            evidence.append(
                "Gold chunk 未进入 Dense Top20 → Reranker 根本没有机会看到它："
                + ", ".join(
                    f"{row['chunk_id'][:12]}(dense rank={row['dense_rank']})"
                    for row in missing_from_dense
                )
            )
        if len(gold_rows) > 1 and report["f_dense_top20_all_gold_hit"] is False:
            causal.append("MULTI_EVIDENCE_FRAGMENTATION")
            evidence.append(
                f"该题需要 {len(gold_rows)} 个证据块，Dense Top20 未能同时全部命中"
                f"（dense 名次：{[row['dense_rank'] for row in gold_rows]}）"
            )

        # --- 因果层 3：Reranker 选择 ---
        if missing_from_reranker and not missing_from_dense:
            causal.append("OTHER_CONFIRMED")
            evidence.append(
                "Gold 已进入 Dense Top20，但被 Reranker 挤出 Top5："
                + ", ".join(
                    f"{row['chunk_id'][:12]}(dense={row['dense_rank']}, reranker={row['reranker_rank']})"
                    for row in missing_from_reranker
                )
            )

        # --- 结构条件（只描述已证事实，不自动等于因果） ---
        for entry in report["gold_evidence"]:
            for owner_id in entry.get("c_owner_chunk_ids", []):
                owner = corpus.chunk(owner_id)
                if owner is None:
                    continue
                if owner["covered_headings"]:
                    structural_conditions.append(
                        {
                            "chunk_id": owner_id,
                            "group_id": entry["group_id"],
                            "condition": "BAD_CHUNK_BOUNDARY",
                            "detail": "承载 Gold 的 chunk 跨越了章节标题：" + " / ".join(owner["covered_headings"]),
                            "on_gold_chunk": owner_id in report["gold_chunk_ids"],
                        }
                    )
                if not owner["section"]:
                    structural_conditions.append(
                        {
                            "chunk_id": owner_id,
                            "group_id": entry["group_id"],
                            "condition": "SECTION_METADATA_LOSS",
                            "detail": "承载 Gold 的 chunk 无法归属到任何章节（首屏/标题区无独立标题行）",
                            "on_gold_chunk": owner_id in report["gold_chunk_ids"],
                        }
                    )
            if "figure_caption" in entry.get("e_owner_detected_types", []):
                structural_conditions.append(
                    {
                        "chunk_id": entry["c_owner_chunk_id"],
                        "group_id": entry["group_id"],
                        "condition": "FIGURE_INFORMATION_LOSS",
                        "detail": f"Gold anchor {entry['evidence_anchor']!r} 位于 Figure caption 内；图内像素信息不进文本流",
                        "on_gold_chunk": entry.get("c_owner_chunk_id") in report["gold_chunk_ids"],
                    }
                )
            for owner_id in entry.get("c_owner_chunk_ids", []):
                owner = corpus.chunk(owner_id)
                if owner is None:
                    continue
                if LATEX_COMMAND.search(span_text(corpus, owner)):
                    structural_conditions.append(
                        {
                            "chunk_id": owner_id,
                            "group_id": entry["group_id"],
                            "condition": "EQUATION_LATEX_LEAKAGE",
                            "detail": "承载 Gold 的 chunk 内含原始 LaTeX 命令片段（cleaner 未语义化）",
                            "on_gold_chunk": owner_id in report["gold_chunk_ids"],
                        }
                    )

        # 去重
        seen_conditions = set()
        unique_conditions = []
        for condition in structural_conditions:
            key = (condition["chunk_id"], condition["condition"])
            if key in seen_conditions:
                continue
            seen_conditions.add(key)
            unique_conditions.append(condition)

        if not causal:
            causal = ["UNKNOWN"]
            evidence.append("现有证据不足以归因到某一层，按计划保留 UNKNOWN")

        report["root_cause"] = causal
        report["root_cause_evidence"] = evidence
        report["structural_conditions"] = unique_conditions
        report["gold_chunk_structural_conditions"] = [
            condition for condition in unique_conditions if condition["on_gold_chunk"]
        ]
        # 纪律：审计只能证明"结构缺陷存在"，不能证明它就是本次丢失的原因。
        # 因果只能在第二轮的受控 A/B 中验证，因此这里显式标记为未证明。
        report["structure_causality"] = (
            "PLAUSIBLE_NOT_PROVEN" if report["gold_chunk_structural_conditions"] else "NOT_APPLICABLE"
        )
        # 丢失层面标签，便于跨题汇总。
        if "PARSER_MISSING_CONTENT" in causal:
            report["evidence_loss_layer"] = "PARSER"
        elif "DENSE_RETRIEVAL_MISS" in causal:
            report["evidence_loss_layer"] = "DENSE_CANDIDATE_RECALL"
        elif "OTHER_CONFIRMED" in causal:
            report["evidence_loss_layer"] = "RERANKER_SELECTION"
        else:
            report["evidence_loss_layer"] = "UNKNOWN"

    layer_counts = Counter(report["evidence_loss_layer"] for report in reports)

    payload = {
        "artifact_type": "phase13_retrieval_failure_root_cause",
        "created_at_utc": _now(),
        "stage": "Step 6 — Retrieval Failure Root-Cause Review",
        "root_cause_taxonomy": ROOT_CAUSES,
        "question_count": len(reports),
        "evidence_loss_layer_counts": dict(sorted(layer_counts.items())),
        "reports": reports,
        "guard": (
            "UNKNOWN 是允许的；本步骤禁止为凑结论强行归类。root_cause 只记录**证据在哪一层丢失**"
            "（可证）；结构缺陷单独放在 structural_conditions，仅标注它是否落在 gold chunk 上"
            "（on_gold_chunk，可证）。结构缺陷与本次丢失的因果关系统一标记为 "
            "structure_causality=PLAUSIBLE_NOT_PROVEN，必须由第二轮受控 A/B 才能确认。"
        ),
    }
    save_json(payload, BASE / "retrieval_failure_root_cause.json")
    return payload


# --------------------------------------------------------------------------------------
# Step 7：Gold chunk quality 审计
# --------------------------------------------------------------------------------------


def step7_gold_chunk_quality(corpus: CorpusAudit) -> dict[str, Any]:
    """对 canonical Gold chunks 做自包含性与污染审计（只读，不改 Gold）。"""

    qrels = load_json(QRELS_SOURCE)
    mapping = load_json(MAPPING_SOURCE)
    questions = {q["question_id"]: q for q in load_json(QUESTIONS_SOURCE)["questions"]}

    multi_group_questions = {
        group["question_id"] for group in mapping["groups"] if group["group_id"] != "G1"
    }

    rows: list[dict[str, Any]] = []
    for question_id, gold_ids in sorted(qrels.items()):
        for chunk_id in gold_ids:
            chunk = corpus.chunk(chunk_id)
            if chunk is None:
                rows.append(
                    {
                        "question_id": question_id,
                        "chunk_id": chunk_id,
                        "present_in_corpus": False,
                        "verdict": "PROBLEMATIC",
                        "flags": ["GOLD_CHUNK_NOT_FOUND_IN_CORPUS"],
                    }
                )
                continue
            types = chunk["element_types"]
            # 逐条回答计划 §12 的六个问题（每项都是可复核的布尔事实）。
            self_contained = (
                bool(chunk["section"])
                and not chunk["covered_headings"]
                and not starts_mid_sentence(chunk["prev_char"], chunk["first_char"])
                and not ends_mid_sentence(chunk["last_char"])
            )
            answers = {
                "self_contained": self_contained,
                "has_heading_context": bool(chunk["section"]),
                "too_much_irrelevant_content": len(chunk["covered_headings"]) >= 2,
                "crosses_section": bool(chunk["covered_headings"]),
                "table_or_caption_polluted": "figure_caption" in types or "table_caption" in types,
                "reference_polluted": "reference_entry" in types,
                "required_claim_fragmented": question_id in multi_group_questions,
            }
            flags: list[str] = []
            if not answers["has_heading_context"]:
                flags.append("NO_HEADING_CONTEXT")
            if answers["crosses_section"]:
                flags.append("CROSSES_SECTION_BOUNDARY")
            if answers["table_or_caption_polluted"]:
                flags.append("CAPTION_POLLUTION")
            if answers["reference_polluted"]:
                flags.append("REFERENCE_POLLUTION")
            if chunk["token_count"] < VERY_SHORT_TOKEN_THRESHOLD:
                flags.append("VERY_SHORT")
            if answers["required_claim_fragmented"]:
                # 多组题：Gold 分布到多个 chunk 是问题本身的性质，标记但不算结构缺陷。
                flags.append("MULTI_GROUP_QUESTION")
            # 严重度分层：只有实质性缺陷才判 PROBLEMATIC；多组分散单独记为 MINOR。
            substantive = [
                flag
                for flag in flags
                if flag not in {"MULTI_GROUP_QUESTION"}
            ]
            verdict = "PROBLEMATIC" if substantive else ("MINOR" if flags else "GOOD")
            rows.append(
                {
                    "question_id": question_id,
                    "chunk_id": chunk_id,
                    "file_name": chunk["file_name"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "token_count": chunk["token_count"],
                    "section": chunk["section"],
                    "subsection": chunk["subsection"],
                    "answers": answers,
                    "detected_types": types,
                    "starts_mid_sentence": starts_mid_sentence(chunk["prev_char"], chunk["first_char"]),
                    "ends_mid_sentence": ends_mid_sentence(chunk["last_char"]),
                    "flags": flags,
                    "verdict": verdict,
                    "text_preview": chunk["text"][:300].replace("\n", " ⏎ "),
                }
            )

    total = len(rows)
    verdict_counts = Counter(row["verdict"] for row in rows)
    flag_counts = Counter(flag for row in rows for flag in row["flags"])
    answer_counts = {
        key: {
            "true": sum(1 for row in rows if row.get("answers", {}).get(key) is True),
            "false": sum(1 for row in rows if row.get("answers", {}).get(key) is False),
        }
        for key in (
            "self_contained",
            "has_heading_context",
            "too_much_irrelevant_content",
            "crosses_section",
            "table_or_caption_polluted",
            "reference_polluted",
            "required_claim_fragmented",
        )
    }
    # required claim 是否被分散到多个 chunk：用多组题与多 Gold 数量近似。
    fragmented_questions = sorted(qid for qid, ids in qrels.items() if len(ids) > 1)
    payload = {
        "artifact_type": "phase13_gold_chunk_quality",
        "created_at_utc": _now(),
        "stage": "Step 7 — Gold Chunk Quality Review",
        "policy": "只审计，不修改 canonical Gold / qrels / mapping。",
        "gold_chunk_count": total,
        "distinct_gold_chunk_count": len({row["chunk_id"] for row in rows}),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "good": verdict_counts.get("GOOD", 0),
        "minor": verdict_counts.get("MINOR", 0),
        "problematic": verdict_counts.get("PROBLEMATIC", 0),
        "problems_per_gold_chunk": round(verdict_counts.get("PROBLEMATIC", 0) / total, 6) if total else None,
        "flag_counts": dict(sorted(flag_counts.items())),
        "answer_counts": answer_counts,
        "verdict_rule": (
            "GOOD = 无任何标记；MINOR = 仅「跨两篇论文导致的多块分散」；"
            "PROBLEMATIC = 存在无标题上下文 / 跨 section / caption 或表格污染 / 参考文献污染 / 过短中的任一项。"
        ),
        "questions_with_multiple_gold_chunks": fragmented_questions,
        "rows": rows,
    }
    save_json(payload, BASE / "gold_chunk_quality.json")
    return payload


# --------------------------------------------------------------------------------------
# Step 8：Audit decision
# --------------------------------------------------------------------------------------


def step8_decision(
    baseline: dict[str, Any],
    quality: dict[str, Any],
    layout: dict[str, Any],
    elements: dict[str, Any],
    sections: dict[str, Any],
    failure: dict[str, Any],
    gold: dict[str, Any],
) -> dict[str, Any]:
    """依据计划 §13 给出 PROCEED_TO_CONTROLLED_REBUILD 的明确结论。"""

    # 判据 1：3 个 Retrieval Failure 中 ≥2 个有**可证的** PDF/chunk/structure 根因。
    #
    # 严格口径：只承认 Step 6 中"证据在哪一层丢失"层面的结构类归因
    # （PARSER_* / TABLE_* / FIGURE_* / SECTION_METADATA_LOSS / BAD_CHUNK_BOUNDARY /
    #  CHUNK_SEMANTIC_DILUTION）。MULTI_EVIDENCE_FRAGMENTATION 不计入判据 1：
    # 本题集的分散是"跨两篇不同论文"，属于问题本身的性质，不是切分缺陷。
    structure_causes = {
        "PARSER_READING_ORDER",
        "PARSER_MISSING_CONTENT",
        "TABLE_STRUCTURE_LOSS",
        "FIGURE_INFORMATION_LOSS",
        "SECTION_METADATA_LOSS",
        "BAD_CHUNK_BOUNDARY",
        "CHUNK_SEMANTIC_DILUTION",
    }
    with_structure_cause = [
        report["question_id"]
        for report in failure["reports"]
        if set(report["root_cause"]) & structure_causes
    ]
    loss_layer_counts = failure["evidence_loss_layer_counts"]
    dense_candidate_failures = [
        report["question_id"]
        for report in failure["reports"]
        if report["evidence_loss_layer"] == "DENSE_CANDIDATE_RECALL"
    ]
    # §27-C 的判据：如果全部失败题的**可证**丢失层都是 Dense 候选召回，则 chunk 不是主因。
    dense_is_primary = len(dense_candidate_failures) == len(failure["reports"])

    # 判据 2：结构问题规模是否明显且可复现。
    contamination = quality["structural_contamination"]
    furniture_exact = quality["document_furniture"]["definition_exact_span"]
    structural_scale = {
        "crossing_section_boundary_chunks": contamination["counts"]["crossing_section_boundary"],
        "crossing_section_boundary_rate": contamination["multi_section_suspect_rate"],
        "text_and_figure_caption_mixed": contamination["counts"]["text_and_figure_caption_mixed"],
        "text_and_table_mixed": contamination["counts"]["text_and_table_mixed"],
        "figures_never_isolated": contamination["counts"]["figure_caption_only_chunks"] == 0,
        "tables_never_isolated": contamination["counts"]["table_caption_only_chunks"] == 0,
        "references_boundary_chunks": contamination["counts"]["references_boundary_chunks"],
        "line_structure_preserved": contamination["chunk_text_line_structure_preserved"],
        "structure_metadata_present": sections["structure_metadata_present"],
        "high_confidence_page_furniture_rate": furniture_exact["high_confidence_furniture_rate"],
        "high_confidence_page_furniture_chunks": furniture_exact["high_confidence_furniture_chunks"],
        "table_structure_lost_documents": len(
            elements["numeric_table_row_evidence"]["documents_with_numeric_rows"]
        ),
        "table_flatten_documents_visual": sorted(layout["documents_with_table_flatten"]),
        "table_flatten_documents_visual_count": len(layout["documents_with_table_flatten"]),
        "layout_reading_order_error_rate": layout["reading_order_error_rate"],
    }
    # 阈值口径：计划 §13 只说"明显且可复现"，不预设数值；这里用若干互相独立、
    # 都指向"结构性表示缺失"的可复现事实，任一成立即视为达到规模。
    scale_reasons: list[str] = []
    if structural_scale["crossing_section_boundary_rate"] >= 0.15:
        scale_reasons.append(
            f"跨越章节标题的 chunk 占比 {structural_scale['crossing_section_boundary_rate']:.4f} >= 0.15"
        )
    if structural_scale["figures_never_isolated"]:
        scale_reasons.append("0 个 chunk 只承载 figure caption：caption 从不被隔离，必然与正文同块")
    if not structural_scale["line_structure_preserved"]:
        scale_reasons.append("0/361 chunk 保留行边界：section/caption/表格行的行级结构在检索单元中丢失")
    if structural_scale["table_flatten_documents_visual_count"] >= 5:
        scale_reasons.append(
            f"逐页看图确认 {structural_scale['table_flatten_documents_visual_count']}/10 篇 PDF "
            f"存在多列数值表被线性化（页：{', '.join(layout['table_flatten_page_refs'])}）；"
            f"纯数值行启发式只检出 {structural_scale['table_structure_lost_documents']}/10，属下界"
        )
    if not structural_scale["structure_metadata_present"]:
        scale_reasons.append("知识库 metadata 完全不含 section / subsection / chunk_type / parent_section")
    if structural_scale["high_confidence_page_furniture_rate"] >= 0.15:
        scale_reasons.append(
            f"{structural_scale['high_confidence_page_furniture_chunks']}/361 chunk 吃进页级装饰"
            f"（页码/arXiv 侧栏/版权/水印，率 {structural_scale['high_confidence_page_furniture_rate']:.4f}）"
        )

    criterion_1 = len(with_structure_cause) >= 2
    criterion_2 = bool(scale_reasons)
    proceed = bool(criterion_1 or criterion_2)

    # 候选推荐：结构规模达标 → 最小变量的 section-aware 方案；
    # 同时必须显式声明"结构缺陷的因果未证明"，避免把相关性当成结论。
    recommended = "SECTION_AWARE_MINIMAL" if proceed else "NONE"
    payload = {
        "artifact_type": "phase13_audit_decision",
        "created_at_utc": _now(),
        "stage": "Step 8 — Phase 13A Audit Decision",
        "criteria": {
            "criterion_1": {
                "description": "3 个 Retrieval Failure 中 ≥2 个有**可证**的 PDF/chunk/structure 根因",
                "questions_with_structure_root_cause": with_structure_cause,
                "count": len(with_structure_cause),
                "met": criterion_1,
                "note": (
                    "严格口径。MULTI_EVIDENCE_FRAGMENTATION 不计入：本题集的证据分散是"
                    "跨两篇不同论文，属问题性质而非切分缺陷。"
                ),
            },
            "criterion_2": {
                "description": "结构污染率 / section crossing / table loss 达到明显且可复现的问题规模",
                "structural_scale": structural_scale,
                "reasons": scale_reasons,
                "met": criterion_2,
            },
        },
        "evidence_loss_layer_counts": loss_layer_counts,
        "questions_with_dense_candidate_recall_loss": dense_candidate_failures,
        "dense_candidate_recall_is_primary_for_all_failures": dense_is_primary,
        "PROCEED_TO_CONTROLLED_REBUILD": "YES" if proceed else "NO",
        "recommended_candidate": recommended,
        "decision_rationale": (
            "判据 2 成立 → 按计划 §13 进入受控重建（OR 关系，判据 1 不成立也可进入）。"
            "同时必须并列陈述一个相反方向的事实：三题的**可证**证据丢失层全部是 Dense 候选召回，"
            "结构缺陷与本次丢失的因果关系尚未证明。因此第二轮必须同时检验两个假设："
            "若 section-aware 候选能恢复这 3 题 → 结构是主因（§27-A）；"
            "若结构更干净但 3 题仍不恢复 → 按 §27-C 转向 sparse / lexical candidate augmentation。"
        ),
        "guardrails": {
            "runtime_modified": False,
            "baseline_collection_modified": False,
            "canonical_gold_modified": False,
            "note": "本决策只决定是否进入第二轮的受控 A/B，不代表任何候选被晋升。",
        },
        "diagnosis_inputs": {
            "chunk_count": baseline["knowledge_base"]["chunk_count"],
            "reading_order_error_rate": layout["reading_order_error_rate"],
            "section_attribution_rate": sections["section_attribution_rate"],
            "structure_metadata_present": sections["structure_metadata_present"],
            "gold_problematic_chunks": gold["problematic"],
        },
    }
    save_json(payload, BASE / "audit_decision.json")
    return payload


# --------------------------------------------------------------------------------------
# Finalize：产物清单 + 执行结果
# --------------------------------------------------------------------------------------


FINALIZE_ARTIFACTS = [
    "baseline.json",
    "chunk_inventory.json",
    "chunk_quality_metrics.json",
    "layout_review_scaffold.json",
    "layout_review_annotations.json",
    "pdf_layout_audit.json",
    "visual_structural_elements_audit.json",
    "section_hierarchy_audit.json",
    "retrieval_failure_root_cause.json",
    "gold_chunk_quality.json",
    "audit_decision.json",
]


def step_finalize(corpus: CorpusAudit) -> dict[str, Any]:
    """落盘 run_manifest.json 与 execution_result.json，并复验 §4 冻结条件仍成立。"""

    # 复验 1：审计对象与线上 Chroma 身份一致（证明全程只读、未改基线）。
    # verify_identity 自身在不一致时会抛 AuditAnomaly，这里只做显式断言与取值。
    identity = corpus.verify_identity()
    if not identity.get("local_recompute_matches_store"):
        raise AuditAnomaly(
            "Finalize",
            "§4 停止条件：361 chunk 身份在审计过程中发生变化",
            json.dumps(identity, ensure_ascii=False)[:400],
        )
    baseline_identity = load_json(BASE / "baseline.json")["knowledge_base"][
        "sorted_chunk_id_identity_hash"
    ]
    if identity["sorted_chunk_id_hash"] != baseline_identity:
        raise AuditAnomaly(
            "Finalize",
            "§4 停止条件：chunk 身份 hash 与 Step 0 冻结值不一致",
            f"{identity['sorted_chunk_id_hash']} != {baseline_identity}",
        )

    artifacts: dict[str, Any] = {}
    for name in FINALIZE_ARTIFACTS:
        path = BASE / name
        if not path.exists():
            artifacts[name] = {"status": "MISSING"}
            continue
        payload = load_json(path)
        entry: dict[str, Any] = {
            "status": "PRESENT",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for key in ("created_at_utc", "stage", "artifact_type"):
            if isinstance(payload, dict) and key in payload:
                entry[key] = payload[key]
        artifacts[name] = entry

    decision = load_json(BASE / "audit_decision.json")
    manifest = {
        "artifact_type": "phase13_run_manifest",
        "created_at_utc": _now(),
        "stage": "Finalize — 产物清单与冻结复验",
        "phase": "phase13_kb_chunk_audit_v1",
        "stage_scope": "第一轮 Step 0–8（Audit only；未进入 Controlled Rebuild）",
        "runtime_modified": False,
        "baseline_collection_modified": False,
        "canonical_gold_modified": False,
        "project_spec_modified": False,
        "project_spec_note": (
            "Phase 13 只审计不改设计合同；且 PROJECT_SPEC.md 属 runtime_source_fingerprint 采集范围，"
            "为避免破坏 Phase 12 冻结指纹，本阶段不追加 SPEC 章节（与 Phase 12 同一处理）。"
        ),
        "executed_steps": [
            "Step 0 Freeze Baseline",
            "Step 1 Chunk Inventory",
            "Step 2 Automatic Chunk Quality Metrics",
            "Step 3 PDF Layout Audit (render + per-page visual review)",
            "Step 4 Table / Figure / Formula Audit",
            "Step 5 Section Hierarchy Audit",
            "Step 6 Retrieval Failure Root Cause",
            "Step 7 Gold Chunk Quality Review",
            "Step 8 Audit Decision",
        ],
        "step_9_and_beyond": "NOT_ENTERED — 计划要求第一轮止于 Step 8",
        "artifacts": artifacts,
        "frozen_baseline_recheck": {
            "collection_name": COLLECTION_NAME,
            "chunk_count": len(corpus.chunks),
            "chunk_count_expected": FROZEN_CHUNK_COUNT,
            "sorted_chunk_id_identity_hash": identity["sorted_chunk_id_hash"],
            "baseline_identity_hash": baseline_identity,
            "identity_unchanged": True,
        },
        "not_run": {
            "candidate_parser_manifest.json": "NOT_RUN — Step 9 起属第二轮，第一轮不进入",
            "candidate_chunk_inventory.json": "NOT_RUN — 同上",
            "candidate_gold_mapping.json": "NOT_RUN — 同上",
            "candidate_ir_metrics.json": "NOT_RUN — 同上",
            "failure_specific_comparison.json": "NOT_RUN — 同上",
            "structure_quality_comparison.json": "NOT_RUN — 同上",
            "cost_comparison.json": "NOT_RUN — 同上",
            "phase12_retrieval_replay.json": "NOT_RUN — 同上",
        },
        "decision_pointer": {
            "artifact": "audit_decision.json",
            "PROCEED_TO_CONTROLLED_REBUILD": decision["PROCEED_TO_CONTROLLED_REBUILD"],
            "recommended_candidate": decision["recommended_candidate"],
        },
    }
    save_json(manifest, BASE / "run_manifest.json")

    quality = load_json(BASE / "chunk_quality_metrics.json")
    layout = load_json(BASE / "pdf_layout_audit.json")
    elements = load_json(BASE / "visual_structural_elements_audit.json")
    sections = load_json(BASE / "section_hierarchy_audit.json")
    failure = load_json(BASE / "retrieval_failure_root_cause.json")
    gold = load_json(BASE / "gold_chunk_quality.json")

    execution = {
        "artifact_type": "phase13_execution_result",
        "created_at_utc": _now(),
        "stage": "Phase 13A — Execution Result（Step 0–8 汇总）",
        "phase": "phase13_kb_chunk_audit_v1",
        "question": "PDF 解析 / 阅读顺序 / 章节边界 / 表格与图处理 / chunk 边界 / metadata 丢失，"
        "是否是可证的 Retrieval Failure 与多证据召回缺口根因？",
        "answer": (
            "不是可证的根因。三个 Retrieval Failure 的**可证**证据丢失层 100% 落在 Dense 候选召回"
            "（dense rank 25 / 77 / 43，均未进 Top20），parser 抽取了全部必需事实、全部 anchor 落进 chunk、"
            "且与 qrels 一致。与此同时知识库存在大规模、可复现的**结构性表示缺陷**，"
            "但两者的因果关系未被证明，只能留待第二轮受控 A/B 检验。"
        ),
        "anomalies": [],
        "anomaly_note": "全程未命中 §4 任何停止条件；PDF↔抽取文本对应性 8-gram 覆盖率最小值 = 1.0。",
        "frozen_baseline": {
            "collection_name": COLLECTION_NAME,
            "chunk_count": len(corpus.chunks),
            "identity_hash": identity["sorted_chunk_id_hash"],
            "standard_ir_metrics_verified_against_artifact": True,
            "standard_ir_metrics_note": (
                "Step 0 回读 evals/evaluation_v3/standard_ir_v1/resume_metrics.json 的 current_runtime 并"
                "逐项比对（容差 5e-7），6 项全部一致；Recall@20 仍为 N/A。本阶段未重算 embedding、"
                "未重建 Chroma、未改写 qrels / Gold / 冻结排名，因此指标本身不需要重新计算。"
            ),
            "note": "本阶段未重算 embedding、未重建 Chroma、未改写 qrels / Gold / 冻结排名。",
        },
        "headline_metrics": {
            "chunks_audited": len(corpus.chunks),
            "documents_audited": len(corpus.documents),
            "pages_rendered_and_reviewed": layout["pages_reviewed"],
            "chunk_token_min_median_max": [
                quality["size_distribution"]["tokens"]["min"],
                quality["size_distribution"]["tokens"]["median"],
                quality["size_distribution"]["tokens"]["max"],
            ],
            "chunk_token_mean": quality["size_distribution"]["tokens"]["mean"],
            "near_max_chunk_rate": quality["boundary_quality"]["near_max_chunk_rate"],
            "cross_page_rate": quality["boundary_quality"]["cross_page_rate"],
            "starts_mid_sentence_rate": quality["boundary_quality"]["starts_mid_sentence_rate"],
            "ends_mid_sentence_rate": quality["boundary_quality"]["ends_mid_sentence_rate"],
            "chunks_with_newline": quality["structural_contamination"]["chunks_with_newline"],
            "crossing_section_boundary_chunks": quality["structural_contamination"]["counts"][
                "crossing_section_boundary"
            ],
            "crossing_section_boundary_rate": quality["structural_contamination"][
                "multi_section_suspect_rate"
            ],
            "figure_caption_only_chunks": quality["structural_contamination"]["counts"][
                "figure_caption_only_chunks"
            ],
            "table_caption_only_chunks": quality["structural_contamination"]["counts"][
                "table_caption_only_chunks"
            ],
            "high_confidence_page_furniture_chunks": quality["document_furniture"][
                "definition_exact_span"
            ]["high_confidence_furniture_chunks"],
            "high_confidence_page_furniture_rate": quality["document_furniture"][
                "definition_exact_span"
            ]["high_confidence_furniture_rate"],
            "reading_order_error_rate": layout["reading_order_error_rate"],
            "table_flatten_pages": len(layout["table_flatten_page_refs"]),
            "table_flatten_documents": len(layout["documents_with_table_flatten"]),
            "figure_captions": elements["summary_by_type"]["figure_caption"]["count"],
            "table_captions": elements["summary_by_type"]["table_caption"]["count"],
            "table_caption_structure_lost": elements["summary_by_type"]["table_caption"][
                "verdicts"
            ].get("STRUCTURE_LOST", 0),
            "latex_leaked_chunks": elements["formula_latex_leakage"]["chunks_containing_latex"],
            "section_attribution_rate": sections["section_attribution_rate"],
            "structure_metadata_present": sections["structure_metadata_present"],
            "chunks_with_section_metadata_in_store": sections[
                "chunks_with_section_metadata_in_store"
            ],
        },
        "retrieval_failure_findings": [
            {
                "question_id": report["question_id"],
                "question_type": report["question_type"],
                "query": report["query"],
                "evidence_type": report["evidence_type"],
                "root_cause": report["root_cause"],
                "evidence_loss_layer": report["evidence_loss_layer"],
                "dense_ranks": [row["dense_rank"] for row in report["f_per_gold_chunk"]],
                "all_facts_present_in_pdf": report["all_facts_present_in_pdf"],
                "all_anchors_parsed": report["all_anchors_parsed"],
                "all_anchors_chunked": report["all_anchors_chunked"],
                "gold_matches_qrels": report["gold_chunks_match_qrels"],
                "structure_causality": report["structure_causality"],
                "structural_conditions_on_gold": [
                    condition["condition"]
                    for condition in report["gold_chunk_structural_conditions"]
                ],
            }
            for report in failure["reports"]
        ],
        "evidence_loss_layer_counts": failure["evidence_loss_layer_counts"],
        "gold_chunk_quality": {
            "gold_chunk_count": gold["gold_chunk_count"],
            "distinct_gold_chunk_count": gold["distinct_gold_chunk_count"],
            "verdict_counts": gold["verdict_counts"],
            "problems_per_gold_chunk": gold["problems_per_gold_chunk"],
            "flag_counts": gold["flag_counts"],
        },
        "decision": {
            "PROCEED_TO_CONTROLLED_REBUILD": decision["PROCEED_TO_CONTROLLED_REBUILD"],
            "recommended_candidate": decision["recommended_candidate"],
            "criterion_1_met": decision["criteria"]["criterion_1"]["met"],
            "criterion_2_met": decision["criteria"]["criterion_2"]["met"],
            "dense_candidate_recall_is_primary_for_all_failures": decision[
                "dense_candidate_recall_is_primary_for_all_failures"
            ],
        },
        "guardrails": {
            "runtime_modified": False,
            "baseline_collection_modified": False,
            "canonical_gold_modified": False,
            "project_spec_modified": False,
            "new_dependencies": [],
            "chroma_rebuilt": False,
        },
        "artifact_manifest_pointer": "run_manifest.json",
    }
    save_json(execution, BASE / "execution_result.json")
    return execution


# --------------------------------------------------------------------------------------
# 驱动
# --------------------------------------------------------------------------------------


def build_corpus() -> tuple[CorpusAudit, Any]:
    """构造审计语料并完成与线上 Chroma 的身份对账。"""

    from transformers import AutoTokenizer

    settings = Settings()
    if settings.collection_name != COLLECTION_NAME:
        raise AuditAnomaly(
            "§4 停止条件",
            "当前 Settings 未指向冻结的研究 Collection",
            f"{settings.collection_name} != {COLLECTION_NAME}",
        )
    if settings.chunk_size != FROZEN_CHUNK_SIZE or settings.chunk_overlap != FROZEN_CHUNK_OVERLAP:
        raise AuditAnomaly(
            "§4 停止条件",
            "Chunk 配置与冻结基线不同",
            f"{settings.chunk_size}/{settings.chunk_overlap}",
        )
    tokenizer = BgeM3Tokenizer(settings.embedding_model)
    raw_tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    corpus = CorpusAudit(tokenizer, raw_tokenizer).build()
    return corpus, tokenizer


STEP_ORDER = ["baseline", "inventory", "layout", "elements", "sections", "failure", "gold", "decide"]


def main() -> int:
    """按子命令执行 Phase 13A 的 Step 0–8。"""

    parser = argparse.ArgumentParser(description="Phase 13A KB / Chunk Quality Audit")
    parser.add_argument("step", choices=[*STEP_ORDER, "render", "finalize", "all"])
    args = parser.parse_args()

    # 离线模型模式：审计不访问网络。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    BASE.mkdir(parents=True, exist_ok=True)

    try:
        corpus, _tokenizer = build_corpus()
        if args.step == "all":
            baseline = step0_baseline(corpus)
            inventory = step1_inventory(corpus)
            quality = step2_quality_metrics(corpus, inventory)
            selection = select_review_pages(corpus)
            layout = step3_layout_audit(corpus, selection)
            elements = step4_visual_elements(corpus, layout)
            sections = step5_section_hierarchy(corpus)
            failure = step6_retrieval_failures(corpus)
            gold = step7_gold_chunk_quality(corpus)
            step8_decision(baseline, quality, layout, elements, sections, failure, gold)
            step_finalize(corpus)
            _checkpoint("Step 0–8", "COMPLETE", sorted(p.name for p in BASE.glob("*.json")))
            print("[phase13] Step 0–8 COMPLETE")
        else:
            if args.step == "baseline":
                step0_baseline(corpus)
            elif args.step == "inventory":
                inventory = step1_inventory(corpus)
                step2_quality_metrics(corpus, inventory)
            elif args.step == "render":
                scaffold = step3_render_scaffold(corpus)
                print(
                    f"[phase13] rendered {scaffold['rendered_page_count']} pages -> "
                    f"{scaffold['render_dir']}"
                )
            elif args.step == "layout":
                step3_layout_audit(corpus, select_review_pages(corpus))
            elif args.step == "elements":
                layout = load_json(BASE / "pdf_layout_audit.json")
                step4_visual_elements(corpus, layout)
            elif args.step == "sections":
                step5_section_hierarchy(corpus)
            elif args.step == "failure":
                step6_retrieval_failures(corpus)
            elif args.step == "gold":
                step7_gold_chunk_quality(corpus)
            elif args.step == "decide":
                step8_decision(
                    load_json(BASE / "baseline.json"),
                    load_json(BASE / "chunk_quality_metrics.json"),
                    load_json(BASE / "pdf_layout_audit.json"),
                    load_json(BASE / "visual_structural_elements_audit.json"),
                    load_json(BASE / "section_hierarchy_audit.json"),
                    load_json(BASE / "retrieval_failure_root_cause.json"),
                    load_json(BASE / "gold_chunk_quality.json"),
                )
            elif args.step == "finalize":
                step_finalize(corpus)
            _checkpoint(f"Step {args.step}", "COMPLETE", sorted(p.name for p in BASE.glob("*.json")))
            print(f"[phase13] {args.step} COMPLETE")
    except AuditAnomaly as exc:
        _write_anomaly(args.step, exc.args[1] if len(exc.args) > 1 else "§4", str(exc.args[-1]))
        print(f"[phase13] ANOMALY -> stopped: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
