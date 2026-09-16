"""Phase 13A Knowledge Base / Chunk Quality Audit 的纯逻辑与产物一致性测试。

覆盖范围：
  1. 纯函数：章节标题识别、章节归属、结构元素抽取、句切分、分布、chunk 边界启发式；
  2. 页面装饰识别：页码区间、单调性自证、跨文档防误匹配、纯数字重复串排除；
  3. Gold anchor 定位语义：(page, anchor, occurrence) 优先于已漂移的 source_span_start；
  4. 产物一致性：Step 0–8 的 JSON 之间的关键不变量与 §12/§13 的判定口径。

纪律：
  - 不访问网络、不加载真实模型；需要文档/区间的地方全部用构造对象；
  - 只读产物，不写入、不修改任何冻结 artifact。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.run_phase13_kb_chunk_audit import (
    FROZEN_CHUNK_COUNT,
    LATEX_COMMAND,
    LAYOUT_VERDICTS,
    PDF_TEXT_COVERAGE_MIN,
    _block_column,
    _locate_anchor,
    detect_document_furniture,
    detect_headings,
    distribution,
    ends_mid_sentence,
    extract_structural_elements,
    furniture_hit,
    is_mid_word,
    page_level_furniture_spans,
    page_number_sequence_report,
    resolve_section,
    span_intersects_furniture,
    split_sentences,
    starts_mid_sentence,
)


PHASE13 = Path(__file__).parents[1] / "evals/evaluation_v3/phase13_kb_chunk_audit_v1"


def _segment(content: str, page_number: int) -> SimpleNamespace:
    """构造与 DocumentSegment 同形的最小对象（只需 content / page_number）。"""

    return SimpleNamespace(content=content, page_number=page_number)


def _document(segments: list, document_id: str = "doc-1") -> dict:
    """构造 CorpusAudit 内部使用的 document 记录。"""

    return {
        "document_id": document_id,
        "file_name": "sample.pdf",
        "segments": segments,
        "page_count": len(segments),
    }


def _chunk(document_id: str, spans: list[tuple[int, int, int]], chunk_id: str = "c-1") -> dict:
    """构造带字符区间的 chunk 记录（spans 为 (segment_index, start, end)）。"""

    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "spans": [
            {"segment_index": index, "char_start": start, "char_end": end}
            for index, start, end in spans
        ],
    }


# --------------------------------------------------------------------------------------
# 1. 章节标题识别与归属
# --------------------------------------------------------------------------------------


def test_detect_headings_recognises_numbered_and_named_headings() -> None:
    """编号标题与具名标题都应被识别，并给出正确的层级。"""

    segments = [
        _segment("Abstract\nSome text.\n1 Introduction\nBody text.\n3.1 Set-to-Set\nMore.", 1)
    ]
    headings = detect_headings(segments)
    titles = [heading["title"] for heading in headings]
    assert "1 Introduction" in titles
    assert "3.1 Set-to-Set" in titles
    assert "Abstract" in titles
    levels = {heading["title"]: heading["level"] for heading in headings}
    assert levels["1 Introduction"] == 1
    assert levels["3.1 Set-to-Set"] == 2


def test_detect_headings_rejects_duplicate_major_number_and_duplicate_named_title() -> None:
    """一级编号重复、具名标题重复都必须剔除（实测的表格单元格 / 表头误检）。"""

    segments = [
        _segment("4 Note that this is a table cell\n4 Point Completion Network\nMethods\n", 1),
        _segment("Methods\n", 2),
    ]
    headings = detect_headings(segments)
    titles = [heading["title"] for heading in headings]
    assert titles.count("4 Point Completion Network") == 1
    assert "4 Note that this is a table cell" not in titles
    # "Methods" 出现两次 → 判定为表格列头，全部排除。
    assert "Methods" not in titles


def test_detect_headings_rejects_citation_math_and_trailing_punctuation() -> None:
    """含引用特征、数学符号、或以标点结尾的行不得当作标题。"""

    segments = [
        _segment(
            "3.2 Zhang et al. 2019\nThe row reads,\n5 Size Ydetail: M ×3\n2 Valid Heading\n",
            1,
        )
    ]
    titles = [heading["title"] for heading in detect_headings(segments)]
    assert "2 Valid Heading" in titles
    assert not any("et al" in title for title in titles)
    assert not any("×" in title for title in titles)
    assert not any(title.endswith(",") for title in titles)


def test_detect_headings_rejects_major_number_above_twenty() -> None:
    """主编号 > 20 一律不是章节标题。"""

    segments = [_segment("25 Some Long Caption Line\n1 Introduction\n", 1)]
    titles = [heading["title"] for heading in detect_headings(segments)]
    assert titles == ["1 Introduction"]


def test_resolve_section_uses_latest_heading_before_offset() -> None:
    """章节归属取该位置之前最近的标题；更深的标题作为 subsection。"""

    segments = [_segment("1 Introduction\nbody\n1.2 Details\nmore body\n", 1)]
    headings = detect_headings(segments)
    first = resolve_section(headings, 0, segments[0].content.index("body"))
    assert first["section"] == "1 Introduction"
    assert first["subsection"] is None

    second = resolve_section(headings, 0, segments[0].content.index("more body"))
    assert second["section"] == "1 Introduction"
    assert second["subsection"] == "1.2 Details"
    assert second["section_path"] == ["1 Introduction", "1.2 Details"]


def test_resolve_section_returns_empty_before_first_heading() -> None:
    """首个标题之前的内容没有章节归属（首屏/标题区）。"""

    segments = [_segment("Title line\nauthors\n1 Introduction\nbody\n", 1)]
    headings = detect_headings(segments)
    result = resolve_section(headings, 0, 0)
    assert result == {"section": None, "subsection": None, "section_path": []}


# --------------------------------------------------------------------------------------
# 2. 结构元素抽取
# --------------------------------------------------------------------------------------


def test_extract_structural_elements_marks_caption_forms_and_line_start() -> None:
    """caption 形态（Figure N: 大写）应被识别，并记录是否位于行首。"""

    segments = [
        _segment("Figure 2: PCN architecture. The encoder abstracts.\nnoise\n", 1)
    ]
    elements = extract_structural_elements(_document(segments))
    captions = [item for item in elements if item["element_type"] == "figure_caption"]
    assert captions, "应至少抽出一个 figure_caption"
    assert captions[0]["caption_form"] is True
    assert captions[0]["at_line_start"] is True
    assert captions[0]["page_number"] == 1


def test_extract_structural_elements_keeps_exact_char_spans() -> None:
    """元素必须带精确字符区间，且区间内容可回切原文。"""

    content = "Table 3: Results on ShapeNet.\nbody text\n"
    segments = [_segment(content, 5)]
    element = next(
        item
        for item in extract_structural_elements(_document(segments))
        if item["element_type"] == "table_caption"
    )
    # caption 形态要求 "N:" 后紧跟大写字母，因此区间末端包含该字母。
    assert content[element["char_start"] : element["char_end"]] == "Table 3: R"
    assert element["at_line_start"] is True
    assert element["page_number"] == 5


def test_extract_structural_elements_detects_equation_number_line() -> None:
    """独立成行的公式编号应被识别为 equation 元素。"""

    segments = [_segment("loss\n(3)\n", 2)]
    kinds = {
        item["element_type"] for item in extract_structural_elements(_document(segments))
    }
    assert "equation" in kinds


def test_latex_command_recognises_raw_latex_leakage() -> None:
    """原始 LaTeX 泄漏片段必须能被识别（Phase 13 Step 4 追加发现）。"""

    # 实测形态一：一整条命令被空格拆散成多段，只有首段带反斜杠。
    assert LATEX_COMMAND.findall(r"\ma th cal {G} = (V, E, W), (1)") == ["\\ma"]
    # 实测形态二：多条命令各自带反斜杠（svdformer / hyper_pcn 的 cases 环境）。
    assert LATEX_COMMAND.findall(r"\beg i n { ca ses} 1, & \text {if}") == [
        "\\beg",
        "\\text",
    ]
    assert LATEX_COMMAND.search("plain unicode equation G = (V, E, W)") is None


# --------------------------------------------------------------------------------------
# 3. 文本启发式与分布
# --------------------------------------------------------------------------------------


def test_boundary_helpers_behave_as_documented() -> None:
    """切词 / 切句 / 句末中断三条启发式的边界行为。"""

    assert is_mid_word("e", "x") is True
    assert is_mid_word(" ", "x") is False
    assert is_mid_word("", "x") is False

    # 起点小写且前字符不是句末标点 → 句子中间。
    assert starts_mid_sentence("d", "i") is True
    assert starts_mid_sentence(".", "i") is False
    assert starts_mid_sentence("d", "I") is False

    assert ends_mid_sentence("e") is True
    assert ends_mid_sentence(".") is False
    assert ends_mid_sentence("") is False


def test_split_sentences_returns_empty_for_blank_input() -> None:
    """空白输入返回空列表，不产生空句子。"""

    assert split_sentences("   ") == []
    assert len(split_sentences("First one. Second one!")) == 2


def test_distribution_reports_all_required_quantiles() -> None:
    """分布函数必须输出 §7.1 要求的全部分位点。"""

    result = distribution([1, 2, 3, 4])
    assert set(result) == {"min", "p25", "median", "p75", "p90", "p95", "max", "mean"}
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["median"] == 2.5
    assert distribution([])["max"] == 0.0


def test_block_column_classifies_left_right_and_full_width_blocks() -> None:
    """栏分类：左栏 / 右栏 / 通栏。"""

    assert _block_column({"x0": 0.08, "x1": 0.47}) == "LEFT"
    assert _block_column({"x0": 0.52, "x1": 0.92}) == "RIGHT"
    assert _block_column({"x0": 0.10, "x1": 0.90}) == "FULL"


# --------------------------------------------------------------------------------------
# 4. 页面装饰（页码 / 页眉 / 侧栏 / 水印）
# --------------------------------------------------------------------------------------


def test_furniture_hit_documents_its_structural_blind_spot() -> None:
    """furniture_hit 按行比较，在无换行文本上恒为 False —— 这是被显式记录的缺陷。"""

    furniture = detect_document_furniture(
        [
            {
                "document_id": "d",
                "segments": [_segment(f"Page Header\nbody {index}\n", index) for index in range(1, 6)],
            }
        ]
    )
    assert furniture["d"], "跨页重复串应被检出"

    # 带换行的原文能命中。
    assert furniture_hit("Page Header\nbody", furniture["d"]) is True
    # chunk 文本已被 decode 折叠掉换行 → 结构性恒为 False。
    assert furniture_hit("Page Header body", furniture["d"]) is False


def test_page_level_furniture_spans_finds_first_and_last_line_page_number() -> None:
    """页码：首非空行或末非空行且整行只由 1–5 位数字组成。"""

    document = _document(
        [
            _segment("1\nTitle\nbody text here\n", 1),
            _segment("body text here\n\n7663\n", 2),
        ]
    )
    spans = page_level_furniture_spans(document)
    numbers = [item for item in spans if item["kind"] == "PAGE_NUMBER"]
    assert [(item["page_number"], item["text"], item["line_position"]) for item in numbers] == [
        (1, "1", "first"),
        (2, "7663", "last"),
    ]


def test_page_level_furniture_spans_ignores_isolated_number_inside_body() -> None:
    """正文中间的孤立数字行不得被当成页码。"""

    document = _document([_segment("Title\nbody\n12\nmore body\nlast line here\n", 1)])
    spans = page_level_furniture_spans(document)
    assert [item for item in spans if item["kind"] == "PAGE_NUMBER"] == []


def test_page_number_sequence_report_validates_monotonicity() -> None:
    """页码序列单调性自证：恒定偏移 + 严格递增 → monotonic 为 True。"""

    document = _document(
        [_segment(f"body {index}\n{9465 + index}\n", index) for index in range(1, 6)]
    )
    spans = page_level_furniture_spans(document)
    report = page_number_sequence_report(spans)
    assert report["monotonic"] is True
    assert report["inversions"] == 0
    assert report["distinct_offsets"] == [9465]
    assert report["page_number_candidate_count"] == 5


def test_page_level_furniture_spans_excludes_pure_digit_repeated_strings() -> None:
    """纯数字重复串（归一化为 # / #.#）必须排除，避免小节号被当成页眉。"""

    document = _document([_segment("3.1\nbody text here\n", 1)])
    spans = page_level_furniture_spans(document, repeated_furniture=["#.#", "page #"])
    assert all(item["kind"] != "REPEATED_FURNITURE" for item in spans)


def test_page_level_furniture_spans_detects_repeated_running_header() -> None:
    """含字母的跨页重复页眉（如论文短标题）应被识别为 REPEATED_FURNITURE。"""

    document = _document([_segment("AdaPoinTr\nbody text here\n", 1)])
    spans = page_level_furniture_spans(document, repeated_furniture=["adapointr"])
    assert [item["kind"] for item in spans] == ["REPEATED_FURNITURE"]
    assert spans[0]["text"] == "AdaPoinTr"


def test_page_level_furniture_spans_excludes_author_running_header_as_citation() -> None:
    """已知取舍：作者型页眉（"J. Wang et al."）会被引用特征过滤掉，属可接受的漏检。

    该过滤是为了剔除跨页重复出现的参考文献 venue 串；副作用是同时漏掉作者页眉。
    因此"跨页重复串"口径既过召回（venue 串）又欠召回（作者页眉），
    只能作为启发式，不能作为 §7.4 的正式结论来源。
    """

    document = _document([_segment("J. Wang et al.\nbody text here\n", 1)])
    spans = page_level_furniture_spans(document, repeated_furniture=["j. wang et al."])
    assert all(item["kind"] != "REPEATED_FURNITURE" for item in spans)


def test_page_level_furniture_spans_detects_arxiv_sidebar() -> None:
    """arXiv 侧栏按模式识别并记录精确区间。"""

    document = _document([_segment("Title\narXiv:2301.04545v1 [cs.CV] 11 Jan 2023\nbody here\n", 1)])
    spans = page_level_furniture_spans(document)
    kinds = {item["kind"] for item in spans}
    assert "ARXIV_SIDEBAR" in kinds


def test_span_intersects_furniture_respects_document_boundary() -> None:
    """区间相交必须校验 document_id：segment_index 只在单篇文档内唯一。"""

    furniture = [
        {
            "kind": "PAGE_NUMBER",
            "document_id": "other-doc",
            "segment_index": 0,
            "char_start": 0,
            "char_end": 1,
            "text": "1",
        }
    ]
    chunk = _chunk("doc-1", [(0, 0, 50)])
    assert span_intersects_furniture(chunk, furniture) == []

    furniture[0]["document_id"] = "doc-1"
    assert len(span_intersects_furniture(chunk, furniture)) == 1


def test_span_intersects_furniture_detects_real_overlap_only() -> None:
    """只有真正相交的区间才算命中。"""

    furniture = [
        {
            "kind": "PAGE_NUMBER",
            "document_id": "doc-1",
            "segment_index": 0,
            "char_start": 100,
            "char_end": 101,
            "text": "7",
        }
    ]
    assert span_intersects_furniture(_chunk("doc-1", [(0, 0, 99)]), furniture) == []
    assert len(span_intersects_furniture(_chunk("doc-1", [(0, 50, 120)]), furniture)) == 1


# --------------------------------------------------------------------------------------
# 5. Gold anchor 定位语义
# --------------------------------------------------------------------------------------


def test_locate_anchor_resolves_by_page_and_occurrence() -> None:
    """定位以 (page, anchor, occurrence) 为准，不依赖已漂移的 source_span_start。"""

    segments = [
        _segment("cover page\n", 1),
        _segment("Figure 2: PCN architecture\nbody\nFigure 2: PCN architecture again\n", 2),
    ]
    document = _document(segments)
    locator = {"page_start": 2, "page_end": 2, "occurrence": 2, "source_span_start": 999}

    result = _locate_anchor(document, locator, "Figure 2: PCN architecture")
    assert result["found"] is True
    assert result["page_number"] == 2
    assert result["occurrence_resolved"] is True
    assert result["total_occurrences"] == 2
    assert result["char_offset"] > 0
    # source_span_start 已漂移 → 必须被记录为不匹配，而不是当作真值。
    assert result["span_start_matches"] is False
    assert result["declared_source_span_start"] == 999


def test_locate_anchor_falls_back_to_other_pages_when_declared_page_misses() -> None:
    """声明页找不到时按页顺序回退；找不到则返回 found=False。"""

    document = _document([_segment("nothing here\n", 1), _segment("needle text\n", 2)])
    hit = _locate_anchor(document, {"page_start": 1, "page_end": 1}, "needle")
    assert hit["found"] is True
    assert hit["page_number"] == 2

    miss = _locate_anchor(document, {"page_start": 1}, "absent anchor")
    assert miss["found"] is False
    assert miss["char_offset"] == -1


def test_locate_anchor_handles_missing_document_and_blank_anchor() -> None:
    """缺文档或空 anchor 时安全返回未找到。"""

    for document, anchor in ((None, "x"), (_document([_segment("a\n", 1)]), "   ")):
        result = _locate_anchor(document, {}, anchor)
        assert result["found"] is False
        assert result["total_occurrences"] == 0


# --------------------------------------------------------------------------------------
# 6. 产物一致性（只读）
# --------------------------------------------------------------------------------------


def _load(name: str) -> dict:
    path = PHASE13 / name
    assert path.exists(), f"缺少 Phase 13 产物：{name}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase13_required_artifacts_exist() -> None:
    """Step 0–8 的必需产物必须全部存在。"""

    required = [
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
        "run_manifest.json",
        "execution_result.json",
    ]
    missing = [name for name in required if not (PHASE13 / name).exists()]
    assert missing == []


def test_phase13_baseline_identity_is_frozen() -> None:
    """基线冻结：361 chunk 且身份 hash 与 finalize 复验一致。"""

    baseline = _load("baseline.json")
    manifest = _load("run_manifest.json")
    knowledge_base = baseline["knowledge_base"]
    assert knowledge_base["chunk_count"] == FROZEN_CHUNK_COUNT
    assert knowledge_base["local_recompute_matches_store"] is True

    recheck = manifest["frozen_baseline_recheck"]
    assert recheck["identity_unchanged"] is True
    assert recheck["sorted_chunk_id_identity_hash"] == knowledge_base["sorted_chunk_id_identity_hash"]


def test_phase13_guardrails_declare_no_mutation() -> None:
    """边界声明：Runtime / 基线 / Gold / SPEC 全部未修改，且无新增依赖。"""

    manifest = _load("run_manifest.json")
    execution = _load("execution_result.json")
    for payload in (manifest, execution["guardrails"]):
        assert payload["runtime_modified"] is False
        assert payload["baseline_collection_modified"] is False
        assert payload["canonical_gold_modified"] is False
    assert manifest["project_spec_modified"] is False
    assert execution["guardrails"]["new_dependencies"] == []
    assert execution["guardrails"]["chroma_rebuilt"] is False


def test_phase13_did_not_enter_controlled_rebuild() -> None:
    """第一轮必须止于 Step 8：candidate_* 全部 NOT_RUN，且未产出任何 candidate 文件。"""

    manifest = _load("run_manifest.json")
    assert manifest["step_9_and_beyond"] == "NOT_ENTERED — 计划要求第一轮止于 Step 8"
    assert all(value.startswith("NOT_RUN") for value in manifest["not_run"].values())
    assert not list(PHASE13.glob("candidate_*.json"))


def test_phase13_pdf_text_correspondence_not_anomalous() -> None:
    """§4 对应性检查：8-gram 覆盖率最小值必须不低于阈值。"""

    layout = _load("pdf_layout_audit.json")
    assert layout["pdf_text_correspondence"]["verdict"] == "PDF_TEXT_CORRESPONDENCE_OK"
    observed = layout["machine_evidence_summary"]["min_extraction_ngram_coverage"]
    assert observed >= PDF_TEXT_COVERAGE_MIN


def test_phase13_layout_verdicts_cover_every_rendered_page() -> None:
    """每一张抽查页都必须有人工 verdict，且落在允许集合内。"""

    layout = _load("pdf_layout_audit.json")
    annotations = _load("layout_review_annotations.json")
    reviewed = {
        (item["file_name"], item["page_number"]) for item in annotations["reviews"]
    }
    audited = {(page["file_name"], page["page_number"]) for page in layout["pages"]}
    assert audited == reviewed
    assert layout["pages_reviewed"] == len(audited)
    for page in layout["pages"]:
        assert page["verdict"] in LAYOUT_VERDICTS
        assert set(page["verdicts"]) <= set(LAYOUT_VERDICTS)


def test_phase13_reading_order_verdict_is_ok_for_all_pages() -> None:
    """本次审计的核心反直觉结论：阅读顺序不是缺陷（0 栏间乱序）。"""

    layout = _load("pdf_layout_audit.json")
    assert layout["reading_order_error_rate"] == 0.0
    assert layout["column_merge_errors"] == 0
    assert layout["verdict_counts_all_labels"]["READING_ORDER_OK"] == layout["pages_reviewed"]


def test_phase13_table_flatten_documents_cover_whole_corpus() -> None:
    """逐页看图确认表格线性化覆盖全部 10 篇 PDF，且页数口径自洽。"""

    layout = _load("pdf_layout_audit.json")
    documents = layout["documents_with_table_flatten"]
    assert len(documents) == 10
    assert layout["table_flatten_errors"] == len(layout["table_flatten_page_refs"])
    assert {ref.split(":")[0] for ref in layout["table_flatten_page_refs"]} == set(documents)


def test_phase13_furniture_metric_corrects_the_legacy_zero() -> None:
    """页面装饰污染：旧口径恒为 0（缺陷），高置信口径必须显著大于 0。"""

    quality = _load("chunk_quality_metrics.json")
    furniture = quality["document_furniture"]
    assert furniture["definition_repeated_string"]["STATUS"] == "UNRELIABLE_STRUCTURALLY_ALWAYS_ZERO"
    assert furniture["definition_repeated_string"]["header_footer_contamination_rate"] == 0.0

    exact = furniture["definition_exact_span"]
    assert exact["high_confidence_furniture_chunks"] > 0
    assert exact["high_confidence_furniture_rate"] > 0.15
    # 页码识别必须自证单调：所有检出页码的文档都不得有逆序。
    for file_name, report in exact["page_number_sequence_validation"].items():
        if report["page_number_candidate_count"]:
            assert report["monotonic"] is True, file_name
            assert len(report["distinct_offsets"]) == 1, file_name


def test_phase13_chunk_text_has_no_line_structure() -> None:
    """行结构在检索单元中彻底消失：0/361 chunk 含换行。"""

    quality = _load("chunk_quality_metrics.json")
    contamination = quality["structural_contamination"]
    assert contamination["chunks_with_newline"] == 0
    assert contamination["chunk_text_line_structure_preserved"] is False
    assert quality["chunk_count"] == FROZEN_CHUNK_COUNT


def test_phase13_no_caption_is_ever_isolated() -> None:
    """caption 从不被隔离：只承载 caption 的 chunk 数为 0。"""

    quality = _load("chunk_quality_metrics.json")
    counts = quality["structural_contamination"]["counts"]
    assert counts["figure_caption_only_chunks"] == 0
    assert counts["table_caption_only_chunks"] == 0
    elements = _load("visual_structural_elements_audit.json")
    assert elements["figure_caption_isolation_rate"] == 0.0
    assert elements["table_isolation_rate"] == 0.0


def test_phase13_section_metadata_is_absent_from_store() -> None:
    """章节 metadata 在知识库中覆盖率为 0；离线归属率不得被当作 store coverage。"""

    sections = _load("section_hierarchy_audit.json")
    assert sections["structure_metadata_present"] is False
    assert sections["chunks_with_section_metadata_in_store"] == 0
    assert sections["section_metadata_coverage_in_store"] == 0.0
    assert sections["section_attribution_rate"] > 0.9
    for field in ("section", "subsection", "chunk_type", "source_element_type", "parent_section"):
        assert field in sections["missing_structure_fields"]


def test_phase13_all_retrieval_failures_lose_evidence_at_dense_candidate_recall() -> None:
    """三题的 evidence_loss_layer 必须全部是 DENSE_CANDIDATE_RECALL，且事实全在 PDF 内。"""

    failure = _load("retrieval_failure_root_cause.json")
    assert failure["evidence_loss_layer_counts"] == {"DENSE_CANDIDATE_RECALL": 3}
    assert "UNKNOWN" in failure["root_cause_taxonomy"]

    questions = {report["question_id"]: report for report in failure["reports"]}
    assert set(questions) == {"ARR-007", "ARR-037", "ARR-040"}
    for report in questions.values():
        assert report["evidence_loss_layer"] == "DENSE_CANDIDATE_RECALL"
        assert report["all_facts_present_in_pdf"] is True
        assert report["all_anchors_parsed"] is True
        assert report["all_anchors_chunked"] is True
        assert report["gold_chunks_match_qrels"] is True
        # 结构缺陷只能标记为"可能但未证明"，不得声称因果。
        if report["gold_chunk_structural_conditions"]:
            assert report["structure_causality"] == "PLAUSIBLE_NOT_PROVEN"
        # 每题都至少有一个 Gold 未进 Dense Top20，才可能落在候选召回层。
        assert any(row["in_dense_top20"] is False for row in report["f_per_gold_chunk"])


def test_phase13_structural_conditions_never_claim_causality() -> None:
    """任何被标记为落在 gold chunk 上的结构条件，都必须同时声明因果未证明。"""

    failure = _load("retrieval_failure_root_cause.json")
    for report in failure["reports"]:
        for condition in report["structural_conditions"]:
            assert condition["on_gold_chunk"] in (True, False)
            assert "condition" in condition and "detail" in condition


def test_phase13_decision_proceeds_via_scale_criterion_only() -> None:
    """判据 1 不成立、判据 2 成立 → PROCEED=YES；且推荐方案为 SECTION_AWARE_MINIMAL。"""

    decision = _load("audit_decision.json")
    criteria = decision["criteria"]
    assert criteria["criterion_1"]["met"] is False
    assert criteria["criterion_1"]["count"] == 0
    assert criteria["criterion_2"]["met"] is True
    assert len(criteria["criterion_2"]["reasons"]) >= 5
    assert decision["PROCEED_TO_CONTROLLED_REBUILD"] == "YES"
    assert decision["recommended_candidate"] == "SECTION_AWARE_MINIMAL"
    assert decision["dense_candidate_recall_is_primary_for_all_failures"] is True
    assert decision["guardrails"]["runtime_modified"] is False


def test_phase13_gold_chunk_quality_answers_all_six_questions() -> None:
    """Step 7 必须显式回答 §12 的六个问题，并给出三级判定计数。"""

    gold = _load("gold_chunk_quality.json")
    for key in (
        "self_contained",
        "has_heading_context",
        "too_much_irrelevant_content",
        "crosses_section",
        "table_or_caption_polluted",
        "reference_polluted",
        "required_claim_fragmented",
    ):
        counts = gold["answer_counts"][key]
        assert set(counts) == {"true", "false"}
        assert counts["true"] + counts["false"] == gold["gold_chunk_count"]

    verdicts = gold["verdict_counts"]
    assert sum(verdicts.values()) == gold["gold_chunk_count"]
    # 本次审计结论：Gold chunk 普遍不自包含。
    assert gold["answer_counts"]["self_contained"]["true"] == 0


def test_phase13_latex_leakage_is_measured_but_excluded_from_failure_gold() -> None:
    """LaTeX 泄漏必须有量化记录，且不得被用来解释三题丢失（三题 Gold 不含该片段）。"""

    elements = _load("visual_structural_elements_audit.json")
    leakage = elements["formula_latex_leakage"]
    assert leakage["total_command_hits"] > 0
    assert leakage["documents_affected"]
    assert "不落在" in leakage["caveat"]

    failure = _load("retrieval_failure_root_cause.json")
    for report in failure["reports"]:
        on_gold = [
            condition
            for condition in report["gold_chunk_structural_conditions"]
            if condition["condition"] == "EQUATION_LATEX_LEAKAGE"
        ]
        assert on_gold == [], report["question_id"]


def test_phase13_manifest_records_artifact_hashes() -> None:
    """manifest 必须给出每个产物的 sha256，便于后续复算与审计。"""

    manifest = _load("run_manifest.json")
    assert len(manifest["artifacts"]) >= 11
    for name, entry in manifest["artifacts"].items():
        assert entry["status"] == "PRESENT", name
        assert len(entry["sha256"]) == 64
    assert manifest["executed_steps"][0].startswith("Step 0")
    assert manifest["executed_steps"][-1].startswith("Step 8")


@pytest.mark.parametrize(
    "artifact_name",
    [
        "baseline.json",
        "chunk_quality_metrics.json",
        "pdf_layout_audit.json",
        "section_hierarchy_audit.json",
        "retrieval_failure_root_cause.json",
        "gold_chunk_quality.json",
        "audit_decision.json",
        "execution_result.json",
    ],
)
def test_phase13_artifacts_are_valid_json_with_stage(artifact_name: str) -> None:
    """产物必须是可解析 JSON 且带 stage 字段，便于跨阶段追溯。"""

    payload = _load(artifact_name)
    assert isinstance(payload, dict)
    assert payload.get("stage")
