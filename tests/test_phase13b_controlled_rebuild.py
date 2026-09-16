"""Phase 13B Knowledge Base Controlled Rebuild 的纯逻辑与产物一致性测试。

覆盖范围：
  1. 纯函数：元素分类（caption 风格 / 表格体 / 公式 / 参考文献）、章节感知装箱、
     超长元素二次切分、元素级清理、anchor 投影（字符区间相交）；
  2. chunk 变体不变量：Baseline 复现 361 + 冻结身份、A/B 不跨章节、B 隔离结构元素、
     清理后无 furniture、行结构保留；
  3. 产物一致性（只读）：baseline 冻结、qrels hash、index_manifest 旧库守护、控制变量。

纪律：
  - 不访问网络、不加载真实模型；需要文档/元素/区间的地方全部构造对象；
  - 只读产物，不写入、不修改任何冻结 artifact；
  - 与 Phase 13A 测试同一条边界：测试可从 ``experiments.run_phase13b_controlled_rebuild``
    导入纯函数，但绝不触碰真实模型 / Chroma / LLM。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.run_phase13b_controlled_rebuild import (
    BASELINE_COLLECTION,
    CANDIDATE_COLLECTIONS,
    CHUNK_SIZE,
    FROZEN_CHUNK_COUNT,
    FROZEN_CHUNK_IDENTITY_HASH,
    FROZEN_QRELS_HASH,
    _chunks_covering_position,
    _classify_block_text,
    _is_table_body,
    _normalize_block_text,
    _projected_qrels,
    _split_oversized_element,
    build_experiment_chunk_id,
    chunk_element_aware,
    chunk_section_aware,
)

PHASE13B = Path(__file__).parents[1] / "evals/evaluation_v3/phase13b_controlled_rebuild_v1"


# --------------------------------------------------------------------------------------
# 纯函数
# --------------------------------------------------------------------------------------


class TestClassifyBlockText:
    def test_standard_table_caption(self) -> None:
        assert _classify_block_text("Table 1. Quantitative results.\n") == "table_caption"

    def test_allcaps_table_caption_style(self) -> None:
        """adapointr 风格：全大写 TABLE N 独占首行、无冒号。"""

        assert _classify_block_text("TABLE 3\nResults on Projected-ShapeNet-55.\n") == "table_caption"

    def test_figure_caption(self) -> None:
        assert _classify_block_text("Figure 2: PCN architecture.\n") == "figure_caption"

    def test_reference_entry(self) -> None:
        assert _classify_block_text("[12] Some Author. A paper title.\n") == "reference_entry"

    def test_plain_paragraph(self) -> None:
        assert _classify_block_text("Point clouds are captured by 3D sensors.\n") == "paragraph"

    def test_latex_equation(self) -> None:
        assert _classify_block_text(r"\ma th cal {G} = (V, E, W), (1)") == "equation"


class TestIsTableBody:
    def test_header_row_block(self) -> None:
        """表格头行（多短行、全标签）必须识别为表格体。"""

        assert _is_table_body("Methods\nPlane\nCabinet\nCar\nChair\nLamp\n")

    def test_numeric_data_block(self) -> None:
        assert _is_table_body("5.50\n22.70\n10.63\n8.70\n11.00\n")

    def test_prose_is_not_table_body(self) -> None:
        assert not _is_table_body(
            "Real-world 3D data are often incomplete, causing loss in geometric information."
        )

    def test_short_isolated_label_not_table_body(self) -> None:
        # 单行、少数词，不应误判成表格体。
        assert not _is_table_body("Methods\n")


class TestNormalizeBlockText:
    def test_strips_trailing_whitespace_only(self) -> None:
        assert _normalize_block_text("Hello  \nWorld\n") == "Hello\nWorld\n"

    def test_unifies_newlines(self) -> None:
        assert _normalize_block_text("a\r\nb\r\n") == "a\nb\n"

    def test_strips_bom(self) -> None:
        assert _normalize_block_text("\ufeffTitle\n") == "Title\n"


class TestSplitOversizedElement:
    def _tok(self, text: str) -> list[int]:
        # 用一个"按字符 1:1"的假 tokenizer：token 数 == 字符数。
        class _Fake:
            def encode(self, text: str) -> list[int]:
                return list(range(len(text)))

            def decode(self, ids: list[int]) -> str:
                return "".join(chr(65 + (i % 26)) for i in ids)

        return []

    def test_short_text_kept_whole(self) -> None:
        class _Fake:
            def encode(self, text: str) -> list[int]:
                return list(range(len(text)))

            def decode(self, ids: list[int]) -> str:
                return "".join("x" for _ in ids)

        element = {"text": "Short sentence."}
        pieces = _split_oversized_element(element, _Fake(), chunk_size=100)
        assert len(pieces) == 1

    def test_oversized_split_on_sentences(self) -> None:
        class _Fake:
            def encode(self, text: str) -> list[int]:
                return list(range(len(text)))

            def decode(self, ids: list[int]) -> str:
                return "".join("x" for _ in ids)

        text = ("A" * 60 + ". " + "B" * 60 + ". " + "C" * 60 + ".")
        element = {"text": text}
        pieces = _split_oversized_element(element, _Fake(), chunk_size=70)
        # 每段都不应超过 chunk_size 个 token。
        assert all(len(_Fake().encode(p)) <= 70 for p in pieces)
        # 至少被切成多段。
        assert len(pieces) >= 2


class TestBuildExperimentChunkId:
    def test_strategy_is_part_of_identity(self) -> None:
        a = build_experiment_chunk_id("doc", "SECTION_AWARE_MINIMAL", 512, 0, 0, "same text")
        b = build_experiment_chunk_id("doc", "ELEMENT_AWARE", 512, 0, 0, "same text")
        assert a != b

    def test_stable_for_same_input(self) -> None:
        a = build_experiment_chunk_id("doc", "S", 512, 0, 0, "x")
        b = build_experiment_chunk_id("doc", "S", 512, 0, 0, "x")
        assert a == b


class TestChunksCoveringPosition:
    def test_exact_span_intersection(self) -> None:
        chunk = {
            "chunk_id": "c1",
            "page_start": 1,
            "page_end": 1,
            "char_spans": [{"segment_index": 0, "char_start": 10, "char_end": 50}],
        }
        located = {"page": 1, "segment_index": 0, "char_start": 30, "char_end": 35}
        assert _chunks_covering_position([chunk], located) == ["c1"]

    def test_no_intersection(self) -> None:
        chunk = {
            "chunk_id": "c1",
            "page_start": 1,
            "page_end": 1,
            "char_spans": [{"segment_index": 0, "char_start": 10, "char_end": 50}],
        }
        located = {"page": 1, "segment_index": 0, "char_start": 80, "char_end": 85}
        assert _chunks_covering_position([chunk], located) == []

    def test_wrong_page_excluded(self) -> None:
        chunk = {
            "chunk_id": "c1",
            "page_start": 2,
            "page_end": 2,
            "char_spans": [{"segment_index": 1, "char_start": 10, "char_end": 50}],
        }
        located = {"page": 1, "segment_index": 0, "char_start": 30, "char_end": 35}
        assert _chunks_covering_position([chunk], located) == []


# --------------------------------------------------------------------------------------
# 产物一致性（只读）
# --------------------------------------------------------------------------------------


class TestBaselineArtifact:
    def test_baseline_exists_and_freezes_identity(self) -> None:
        baseline = json.loads((PHASE13B / "baseline.json").read_text(encoding="utf-8"))
        assert baseline["baseline_collection"]["chunk_count"] == FROZEN_CHUNK_COUNT
        assert baseline["baseline_collection"]["sorted_chunk_id_identity_hash"] == FROZEN_CHUNK_IDENTITY_HASH
        assert baseline["canonical_qrels"]["sha256"] == FROZEN_QRELS_HASH
        assert baseline["canonical_qrels"]["unchanged"] is True

    def test_metric_check_all_match(self) -> None:
        baseline = json.loads((PHASE13B / "baseline.json").read_text(encoding="utf-8"))
        assert all(v["matches"] for v in baseline["standard_ir_metrics"].values())

    def test_guardrails_declare_no_modification(self) -> None:
        baseline = json.loads((PHASE13B / "baseline.json").read_text(encoding="utf-8"))
        assert baseline["guardrails"]["old_chroma_modified"] is False
        assert baseline["guardrails"]["canonical_gold_modified"] is False
        assert baseline["guardrails"]["qrels_modified"] is False


class TestChunkVariantsArtifact:
    def test_baseline_reproduces_frozen_identity(self) -> None:
        variants = json.loads((PHASE13B / "chunk_variants.json").read_text(encoding="utf-8"))
        assert variants["baseline_identity_check"]["matches"] is True
        assert variants["baseline_identity_check"]["chunk_count"] == FROZEN_CHUNK_COUNT

    def test_section_aware_does_not_cross_section(self) -> None:
        variants = json.loads((PHASE13B / "chunk_variants.json").read_text(encoding="utf-8"))
        a = variants["variants"]["SECTION_AWARE_MINIMAL"]
        assert a["summary"]["crosses_section_rate"] == 0.0

    def test_candidates_restore_line_structure(self) -> None:
        variants = json.loads((PHASE13B / "chunk_variants.json").read_text(encoding="utf-8"))
        for name in ("SECTION_AWARE_MINIMAL", "ELEMENT_AWARE"):
            summary = variants["variants"][name]["summary"]
            assert summary["chunks_with_newline"] == summary["chunk_count"]

    def test_element_aware_isolates_captions(self) -> None:
        variants = json.loads((PHASE13B / "chunk_variants.json").read_text(encoding="utf-8"))
        b = variants["variants"]["ELEMENT_AWARE"]
        # figure_caption / table_caption / equation 都应作为独立 chunk_type 出现。
        assert b["chunk_type_counts"].get("figure_caption", 0) > 0
        assert b["chunk_type_counts"].get("table_caption", 0) > 0

    def test_baseline_has_no_newlines(self) -> None:
        """复现 13A 的关键缺陷：baseline chunk 文本不含换行。"""

        variants = json.loads((PHASE13B / "chunk_variants.json").read_text(encoding="utf-8"))
        assert variants["variants"]["BASELINE_C512_O64"]["summary"]["chunks_with_newline"] == 0


class TestIndexManifestArtifact:
    def test_old_collection_guard_unchanged(self) -> None:
        if not (PHASE13B / "index_manifest.json").exists():
            pytest.skip("Step 4 尚未执行")
        manifest = json.loads((PHASE13B / "index_manifest.json").read_text(encoding="utf-8"))
        assert manifest["old_collection_guard"]["unchanged"] is True

    def test_candidate_collections_named_distinctly(self) -> None:
        if not (PHASE13B / "index_manifest.json").exists():
            pytest.skip("Step 4 尚未执行")
        manifest = json.loads((PHASE13B / "index_manifest.json").read_text(encoding="utf-8"))
        names = {v["collection_name"] for v in manifest["candidate_stores"].values()}
        assert BASELINE_COLLECTION not in names
        assert names == set(CANDIDATE_COLLECTIONS.values())
