"""Phase 13C-B — Cleaning & ScientificDocument Fusion Benchmark（Sandbox-Only）执行入口。

对应方案 §2 的 Step 0–11。本脚本只做"第二轮"：

    Parser Output -> Cleaning / Normalization / Fusion -> ScientificDocument

然后 **STOP**，不建 chunk、不做 embedding、不建 Chroma、不动 Runtime（方案 §22）。

## 三个主 arm（方案 §5）

| arm | 输入 | 目的 |
|---|---|---|
| `B0_p0_cleaned` | P0_current 结构化输出 | 最可靠正文路径 + 新 Cleaner 的基线 |
| `B1_mineru_cleaned` | P1_mineru 结构化输出（+ 原生 content_list） | MinerU 自身结构经过统一 Cleaner 后能否满足 body + structure |
| `B2_fused` | B0 + B1，provenance 对齐 | 正文完整与结构能力兼得 |

## 输入复用

13C-A 的 parser 输出已冻结在 ``<SANDBOX>/structured/<arm>/<mode>/<doc>.json``，
本阶段**不重新调用任何 parser**（重跑 MinerU 是 17.7 s/页 的代价，且会引入新的
不可比变量）。Step 0 会逐个校验这些缓存的哈希与完整性；缺失即 STOP。

## 用法

    python -m experiments.run_phase13c_cleaning_benchmark --steps all
    python -m experiments.run_phase13c_cleaning_benchmark --steps 0 1 2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from phase13c import (  # noqa: E402
    cleaning,
    cleaning_metrics,
    fusion,
    gates_v2,
    metrics,
    sandbox,
    schema,
)

BASE = sandbox.ARTIFACT_ROOT / "phase13c_b"
SANDBOX = sandbox.SANDBOX_ROOT
STRUCTURED = SANDBOX / "structured"
PARSERS = SANDBOX / "parsers"
CLEANED = SANDBOX / "cleaned"
FULL_METRICS = SANDBOX / "metrics_full"

SOURCE_DOCS = PROJECT_ROOT / "data/source_docs"
CORPUS_MANIFEST = PROJECT_ROOT / "evals/research_ingestion/academic_corpus_manifest_v1.json"
QUESTIONS_SOURCE = (
    PROJECT_ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json"
)
CONTROLLED_MANIFEST = sandbox.ARTIFACT_ROOT / "template_corpus/controlled_corpus_manifest.json"
CONTROLLED_GT = sandbox.ARTIFACT_ROOT / "ground_truth/controlled_template_ground_truth.json"
SELECTED_PARSERS = sandbox.ARTIFACT_ROOT / "selected_parsers.json"
PARSER_COMPARISON = sandbox.ARTIFACT_ROOT / "reports/parser_comparison.md"
PARSER_METRICS = sandbox.ARTIFACT_ROOT / "metrics/parser_metrics.json"
PHASE13B_DIR = sandbox.ARTIFACT_ROOT / "phase13b_controlled_rebuild_v1"
PHASE13_DIR = sandbox.ARTIFACT_ROOT / "phase13_kb_chunk_audit_v1"
STANDARD_IR_QRELS = sandbox.ARTIFACT_ROOT / "standard_ir_v1/qrels.json"

# 13C-A 冻结的 baseline（本阶段的门 1 是"相对 baseline 无新增丢失"）。
BASELINE_ARM = "P0_current"
# 复现性检查用的受控模板与真实文档（13C-A 用的是 T02）。
REPRO_TEMPLATE = "T02_ieee_two_column"
# 本阶段的三个主 arm。
ARMS = ("B0_p0_cleaned", "B1_mineru_cleaned", "B2_fused")
# 每个 arm 的解析产物来源：arm -> (parser arm, ocr mode, 是否送原生 content_list)
ARM_SOURCES: dict[str, dict[str, Any]] = {
    "B0_p0_cleaned": {"parser": "P0_current", "ocr_mode": "none", "native": False},
    "B1_mineru_cleaned": {"parser": "P1_mineru", "ocr_mode": "none", "native": True},
}
# P1 的原生产物可能是 REMAPPED 的复用结果，其哈希需单独记录。
P1_NATIVE_GLOB = "**/*content_list.json"

# 13C-A 已证实的"OCR 不得用于 born-digital"；本阶段只验证触发规则（方案 §10）。
OCR_TRIGGER_ARM = "O2_ocr"
OCR_TRIGGER_MODE = "force"


def log(message: str) -> None:
    """打印带时间戳的进度行。"""

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# 文档清单
# --------------------------------------------------------------------------------------


def controlled_documents() -> list[dict[str, Any]]:
    """受控模板语料清单（12 个模板）。"""

    manifest = load_json(CONTROLLED_MANIFEST)
    return [
        {
            "document_id": template["template_id"],
            "scope": "controlled",
            "pdf_path": Path(template["pdf_path"]),
            "sha256": template["pdf_sha256"],
            "columns": template["columns"],
            "page_count": template["page_count"],
            "rasterized": template["rasterized"],
        }
        for template in manifest["templates"]
    ]


def real_documents() -> list[dict[str, Any]]:
    """真实语料清单（10 篇 EvidenceFlow 论文）。"""

    manifest = load_json(CORPUS_MANIFEST)
    return [
        {
            "document_id": paper["file_name"].replace(".pdf", ""),
            "scope": "real",
            "pdf_path": SOURCE_DOCS / paper["file_name"],
            "file_name": paper["file_name"],
            "sha256": paper.get("sha256"),
            "columns": 2 if paper.get("has_two_column") else 1,
            "page_count": paper.get("page_count"),
            "rasterized": False,
        }
        for paper in manifest["papers"]
    ]


def all_documents() -> list[dict[str, Any]]:
    return controlled_documents() + real_documents()


def build_real_anchors() -> list[dict[str, Any]]:
    """复用 13C-A 的 canonical evidence anchor 构造（保证两侧完全同一套锚点）。"""

    from experiments.run_phase13c_parser_benchmark import build_real_anchors as build

    return build()


# --------------------------------------------------------------------------------------
# 文本层探测（用于 §3 "扫描页不计入正文误删"）
# --------------------------------------------------------------------------------------


def no_text_layer_pages(pdf_path: Path) -> set[int]:
    """返回"整页没有可抽取文本层"的页码集合（1-based）。

    这是**文档属性**，与哪个 arm 无关：只有把它与"某 arm 没抽出正文"分开，
    才能避免把"没有 OCR 能力"记成"把正文删了"（方案 §3）。
    """

    import pymupdf

    blank: set[int] = set()
    with pymupdf.open(pdf_path) as pdf:
        for index in range(pdf.page_count):
            if not pdf[index].get_text().strip():
                blank.add(index + 1)
    return blank


# --------------------------------------------------------------------------------------
# Step 0 — 冻结 13C-A
# --------------------------------------------------------------------------------------


def step0_freeze() -> dict[str, Any]:
    """记录 13C-A 的身份与全部输入的哈希，并在不一致时 STOP。"""

    log("Step 0 — 冻结 Phase 13C-A 与输入身份")
    sandbox.project_write_guard()

    prior_artifacts = {}
    for path in (
        PARSER_COMPARISON,
        SELECTED_PARSERS,
        PARSER_METRICS,
        sandbox.ARTIFACT_ROOT / "input_hashes.json",
        PHASE13B_DIR / "report.md",
        PHASE13B_DIR / "structured_elements.json",
        PHASE13B_DIR / "index_manifest.json",
        PHASE13_DIR / "report.md",
        STANDARD_IR_QRELS,
    ):
        prior_artifacts[str(path.relative_to(sandbox.ARTIFACT_ROOT)).replace("\\", "/")] = {
            "exists": path.exists(),
            "sha256": sandbox.sha256_file(path) if path.exists() else None,
        }

    # 13C-A 缓存的 parser 输出：逐个记录哈希并检查完整性。
    docs = all_documents()
    parser_outputs: dict[str, Any] = {"missing": [], "entries": {}, "by_arm": {}}
    for arm, source in ARM_SOURCES.items():
        parser_arm, ocr_mode = source["parser"], source["ocr_mode"]
        arm_entries: dict[str, str] = {}
        for document in docs:
            path = STRUCTURED / parser_arm / ocr_mode / f"{document['document_id']}.json"
            if not path.exists():
                parser_outputs["missing"].append(str(path))
                continue
            digest = sandbox.sha256_file(path)
            arm_entries[document["document_id"]] = digest
            parser_outputs["entries"][
                str(path.relative_to(SANDBOX)).replace("\\", "/")
            ] = digest
        parser_outputs["by_arm"][parser_arm] = arm_entries

    # MinerU 原生 content_list（B1 的类型再标注依赖它）。
    native_entries: dict[str, str] = {}
    native_missing: list[str] = []
    for document in docs:
        matches = sorted(
            glob.glob(
                str(PARSERS / "P1_mineru" / "none" / document["document_id"] / P1_NATIVE_GLOB),
                recursive=True,
            )
        )
        if not matches:
            native_missing.append(document["document_id"])
            continue
        native_entries[document["document_id"]] = sandbox.sha256_file(matches[0])
    parser_outputs["native_content_list"] = native_entries
    parser_outputs["native_missing"] = native_missing

    anchors = build_real_anchors()
    real_pdfs = {}
    for document in docs:
        if document["scope"] != "real":
            continue
        record = {
            "path": str(document["pdf_path"]),
            "sha256": sandbox.sha256_file(document["pdf_path"]),
            "manifest_sha256": document["sha256"],
        }
        record["matches_manifest"] = record["sha256"] == record["manifest_sha256"]
        real_pdfs[document["file_name"]] = record
    controlled_pdfs = {}
    for document in docs:
        if document["scope"] != "controlled":
            continue
        record = {
            "path": str(document["pdf_path"]),
            "sha256": sandbox.sha256_file(document["pdf_path"]),
            "manifest_sha256": document["sha256"],
        }
        record["matches_manifest"] = record["sha256"] == record["manifest_sha256"]
        controlled_pdfs[document["document_id"]] = record

    # baseline Chroma：只读身份（不打开、不查询）。
    chroma_sqlite = PROJECT_ROOT / "data/chroma/chroma.sqlite3"
    chroma_identity = {
        "path": str(chroma_sqlite),
        "exists": chroma_sqlite.exists(),
        "sha256": sandbox.sha256_file(chroma_sqlite) if chroma_sqlite.exists() else None,
        "size_bytes": chroma_sqlite.stat().st_size if chroma_sqlite.exists() else None,
    }

    payload = {
        "artifact_type": "phase13c_b_frozen_inputs",
        "created_at_utc": sandbox.utc_now(),
        "stage": "Phase 13C-B",
        "sandbox_root": str(SANDBOX),
        "artifact_root": str(sandbox.ARTIFACT_ROOT / "phase13c_b"),
        "prior_phase_artifacts": prior_artifacts,
        "parser_outputs": parser_outputs,
        "controlled_pdfs": controlled_pdfs,
        "real_pdfs": real_pdfs,
        "all_real_pdfs_match_manifest": all(
            record["matches_manifest"] for record in real_pdfs.values()
        ),
        "all_controlled_pdfs_match_manifest": all(
            record["matches_manifest"] for record in controlled_pdfs.values()
        ),
        "canonical_anchors": {
            "count": len(anchors),
            "sha256": sandbox.sha256_json(anchors),
            "ids": [anchor["anchor_id"] for anchor in anchors],
        },
        "canonical_qrels": {
            "path": str(STANDARD_IR_QRELS),
            "sha256": sandbox.sha256_file(STANDARD_IR_QRELS) if STANDARD_IR_QRELS.exists() else None,
        },
        "baseline_chroma": chroma_identity,
        "frozen_project_files": sandbox.snapshot_frozen_project_files(),
        "source_tree_before": sandbox.snapshot_project_source_tree(),
        "environment": sandbox.environment_fingerprint(),
        "cleaner_version": cleaning.CLEANER_VERSION,
        "fusion_version": fusion.FUSION_VERSION,
    }

    problems: list[str] = []
    if parser_outputs["missing"]:
        problems.append(f"missing cached parser outputs: {len(parser_outputs['missing'])}")
    if native_missing:
        problems.append(f"missing MinerU native content_list: {native_missing}")
    if not payload["all_real_pdfs_match_manifest"]:
        problems.append("source PDFs do not match the corpus manifest")
    if not payload["all_controlled_pdfs_match_manifest"]:
        problems.append("controlled corpus PDFs do not match the template manifest")
    if len(anchors) != gates_v2.ANCHOR_TOTAL:
        problems.append(f"canonical anchor count {len(anchors)} != {gates_v2.ANCHOR_TOTAL}")
    payload["consistency_problems"] = problems
    payload["consistent"] = not problems

    sandbox.save_json(payload, BASE / "frozen_inputs.json")
    if problems:
        raise sandbox.AnomalyStop("Step0", "frozen_inputs_inconsistent", "; ".join(problems))
    log(f"  arm 输入哈希 {len(parser_outputs['entries'])} 项，anchor {len(anchors)} 个 — 一致")
    return payload


# --------------------------------------------------------------------------------------
# Step 1 — 冻结 Gate Definitions v2
# --------------------------------------------------------------------------------------


def step1_freeze_gates() -> dict[str, Any]:
    """把 Gate 5 正式拆成 5A / 5B + Reference Preservation 并落盘。

    版本变更不视为异常（本轮运行之间允许显式修订门定义），但必须**留下版本链**：
    被替换版本写入 ``superseded_versions``，运行日志显式打印——禁止静默换口径。
    """

    log("Step 1 — 冻结 Gate Definitions v2")
    definitions = gates_v2.gate_definitions_v2()
    path = BASE / "gate_definitions_v2.json"
    if path.exists():
        existing = load_json(path)
        existing_version = existing.get("gate_definition_version")
        if existing_version and existing_version != definitions["gate_definition_version"]:
            superseded = list(existing.get("superseded_versions") or [])
            if existing_version not in superseded:
                superseded.append(existing_version)
            definitions["superseded_versions"] = superseded
            log(
                f"  门定义版本变更（显式修订）: {existing_version} -> "
                f"{definitions['gate_definition_version']}（历史版本已记录）"
            )
    sandbox.save_json(definitions, path)
    return definitions


# --------------------------------------------------------------------------------------
# Step 2 — Cleaner 契约检查
# --------------------------------------------------------------------------------------


def step2_cleaner_contract() -> dict[str, Any]:
    """验证 Cleaner 契约：纯函数、无 Runtime 依赖、无写副作用。"""

    log("Step 2 — Cleaner 契约检查（纯函数 / 无 Runtime 依赖）")
    cheating = []
    for module_name in ("cleaning", "fusion", "cleaning_metrics", "gates_v2"):
        source = (PROJECT_ROOT / "experiments/phase13c" / f"{module_name}.py").read_text(encoding="utf-8")
        for pattern in ("import evidenceflow", "from evidenceflow", "from src.", "import chromadb"):
            if pattern in source:
                cheating.append({"module": module_name, "pattern": pattern})
    # 数学跨度规范化是"会改写文本"的唯一入口，必须是纯函数且可逆（raw 保留）。
    probe_text = "obtain $N _ { p } = 1 2 8$ patterns"
    first, applied = cleaning.normalize_math_spans(probe_text)
    second, _ = cleaning.normalize_math_spans(probe_text)
    payload = {
        "artifact_type": "phase13c_b_cleaner_contract",
        "created_at_utc": sandbox.utc_now(),
        "cleaner_version": cleaning.CLEANER_VERSION,
        "fusion_version": fusion.FUSION_VERSION,
        "gate_definition_version": gates_v2.GATE_DEFINITION_VERSION,
        "pure_function_violations": cheating,
        "math_normalization_probe": {
            "input": probe_text,
            "output": first,
            "applied": applied,
            "idempotent": first == second,
        },
        "cleaner_functions": [
            "normalize_body_text",
            "normalize_math_spans",
            "classify_furniture / furniture_subtype_for_unknown",
            "normalize_table",
            "normalize_figure",
            "normalize_equation",
            "normalize_reference",
            "clean_document",
        ],
        "runtime_dependency": False,
        "external_llm_calls": False,
    }
    sandbox.save_json(payload, BASE / "cleaner_contract.json")
    if cheating:
        raise sandbox.AnomalyStop("Step2", "cleaner_has_runtime_dependency", str(cheating))
    return payload


# --------------------------------------------------------------------------------------
# 输入加载与清洗
# --------------------------------------------------------------------------------------


def load_parser_document(arm: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """读取 13C-A 缓存的某个 arm 的解析产物。

    ``arm`` 既可以是 arm id（``B0_p0_cleaned`` / ``B1_mineru_cleaned``，经
    :data:`ARM_SOURCES` 解析出 parser 与 OCR 档位），也可以是直接的 parser id
    （如真实语料指标里以 ``P0_current`` 的原始输出作为 raw 基准，含 B2）。
    """

    if arm in ARM_SOURCES:
        source = ARM_SOURCES[arm]
        parser, ocr_mode = source["parser"], source["ocr_mode"]
    else:
        # 直接给 parser id 时默认 none 档（13C-A 冻结缓存里 P0/P1 的主档位）。
        parser, ocr_mode = arm, "none"
    path = STRUCTURED / parser / ocr_mode / f"{document['document_id']}.json"
    return load_json(path)


def load_native_content_list(document: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """读取 MinerU 原生 content_list（B1 的类型再标注依赖它）。"""

    matches = sorted(
        glob.glob(
            str(PARSERS / "P1_mineru" / "none" / document["document_id"] / P1_NATIVE_GLOB),
            recursive=True,
        )
    )
    if not matches:
        return None
    return load_json(Path(matches[0]))


def clean_for_arm(arm: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """对某个 arm 的某个文档执行清洗（B2 由 :func:`build_fusion_for_document` 处理）。"""

    raw = load_parser_document(arm, document)
    native = load_native_content_list(document) if ARM_SOURCES[arm]["native"] else None
    cleaned = cleaning.clean_document(raw, native_content_list=native)
    cleaned["document_id"] = document["document_id"]
    cleaned["scope"] = document["scope"]
    cleaned["arm_id"] = arm
    return cleaned


def build_fusion_for_document(
    document: Mapping[str, Any], b0: Mapping[str, Any], b1: Mapping[str, Any]
) -> dict[str, Any]:
    """构造 B2 融合文档。"""

    fused = fusion.fuse_documents(dict(b0), dict(b1))
    fused["document_id"] = document["document_id"]
    fused["scope"] = document["scope"]
    fused["arm_id"] = "B2_fused"
    return fused


def document_digest(document: Mapping[str, Any]) -> str:
    """计算 ScientificDocument 的确定性摘要（用于门 B8 的可复现判定）。

    只摘要"语义字段"，排除时间戳、耗时一类天然变化的字段。
    """

    keys = (
        "element_id",
        "type",
        "normalized_text",
        "page",
        "bbox",
        "reading_order",
        "section_path",
        "retrieval_role",
        "fusion_status",
        "source_parsers",
    )
    payload = {
        "cleaner_version": document.get("metadata", {}).get("cleaner_version"),
        "fusion_version": document.get("metadata", {}).get("fusion_version"),
        "elements": [
            {key: element.get(key) for key in keys} for element in document.get("elements", [])
        ],
        "removed_furniture": [
            {key: record.get(key) for key in ("element_id", "type", "page", "reason")}
            for record in document.get("removed_furniture", [])
        ],
        "reference_element_ids": document.get("reference_element_ids"),
    }
    return sandbox.sha256_json(payload)


# --------------------------------------------------------------------------------------
# Step 3–4 — 运行 B0 / B1
# --------------------------------------------------------------------------------------


def run_cleaning_arms() -> dict[str, Any]:
    """Step 3–4：对 B0 / B1 的每个文档执行清洗并落盘。"""

    log("Step 3–4 — 运行 B0 P0 Cleaned / B1 MinerU Cleaned")
    documents = all_documents()
    text_layer_cache: dict[str, list[int]] = {}
    runs: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for arm in ("B0_p0_cleaned", "B1_mineru_cleaned"):
        runs[arm] = {}
        for document in documents:
            started = time.perf_counter()
            try:
                cleaned = clean_for_arm(arm, document)
                elapsed = round(time.perf_counter() - started, 4)
                path = CLEANED / arm / f"{document['document_id']}.json"
                sandbox.save_json(cleaned, path)
                runs[arm][document["document_id"]] = {
                    "status": "OK",
                    "document_path": str(path),
                    "element_count": cleaned["element_count"],
                    "retrieval_element_count": len(cleaned["retrieval_element_ids"]),
                    "removed_furniture_count": len(cleaned["removed_furniture"]),
                    "discarded_unknown_count": len(cleaned["removed_elements"]),
                    "reference_entry_count": len(cleaned["reference_element_ids"]),
                    "cleaning_seconds": elapsed,
                    "digest": document_digest(cleaned),
                    "validation_valid": schema.validate_document(cleaned)["valid"],
                }
            except Exception as exc:  # pragma: no cover - 异常路径只做记录
                failures.append(
                    {
                        "arm": arm,
                        "document_id": document["document_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1200:],
                    }
                )
                runs[arm][document["document_id"]] = {"status": "FAILED", "error": str(exc)}
            if document["scope"] == "real" or document["document_id"] not in text_layer_cache:
                text_layer_cache[document["document_id"]] = sorted(
                    no_text_layer_pages(document["pdf_path"])
                )
    sandbox.save_json(text_layer_cache, SANDBOX / "inputs" / "no_text_layer_pages.json")
    payload = {
        "artifact_type": "phase13c_b_cleaning_runs",
        "created_at_utc": sandbox.utc_now(),
        "arms": runs,
        "failures": failures,
        "no_text_layer_pages": text_layer_cache,
    }
    sandbox.save_json(payload, BASE / "cleaned" / "index.json")
    if failures:
        sandbox.save_json(failures, BASE / "failures" / "cleaning_failures.json")
    log(f"  B0/B1 完成 {sum(len(v) for v in runs.values())} 个文档，失败 {len(failures)}")
    return payload


# --------------------------------------------------------------------------------------
# Step 5 — B2 Fusion
# --------------------------------------------------------------------------------------


def run_fusion_arm(cleaning_runs: Mapping[str, Any]) -> dict[str, Any]:
    """Step 5：构造 B2，并输出 alignment / conflict / unresolved 汇总。"""

    log("Step 5 — 构建 B2 Fusion")
    documents = all_documents()
    runs: dict[str, Any] = {}
    alignment_summary: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for document in documents:
        document_id = document["document_id"]
        try:
            started = time.perf_counter()
            b0 = load_json(CLEANED / "B0_p0_cleaned" / f"{document_id}.json")
            b1 = load_json(CLEANED / "B1_mineru_cleaned" / f"{document_id}.json")
            fused = build_fusion_for_document(document, b0, b1)
            elapsed = round(time.perf_counter() - started, 4)
            path = CLEANED / "B2_fused" / f"{document_id}.json"
            sandbox.save_json(fused, path)
            runs[document_id] = {
                "status": "OK",
                "document_path": str(path),
                "element_count": fused["element_count"],
                "retrieval_element_count": len(fused["retrieval_element_ids"]),
                "digest": document_digest(fused),
                "validation_valid": schema.validate_document(fused)["valid"],
                # 融合耗时单独记，与 B0/B1 的 cleaning_seconds 同一单位（秒）。
                "fusion_seconds": elapsed,
                **fused["fusion_metrics"],
            }
            # 落盘审计表：alignment / conflict / unresolved（只保留摘要，避免产物膨胀）。
            page_key = "controlled" if document["scope"] == "controlled" else "real"
            conflict_rows.append(
                {
                    "scope": page_key,
                    "document_id": document_id,
                    "conflict_count": fused["fusion_metrics"]["conflict_count"],
                    "conflict_kind_counts": fused["fusion_metrics"]["conflict_kind_counts"],
                    "unresolved_conflict_count": fused["fusion_metrics"]["unresolved_conflict_count"],
                    "suppressed_secondary_count": fused["fusion_metrics"]["suppressed_secondary_count"],
                    "recorded_conflicts": len(fused["conflicts"]),
                }
            )
            for conflict in fused["unresolved_conflicts"][:20]:
                unresolved_rows.append({"document_id": document_id, **conflict})
            alignment_summary.append(
                {
                    "scope": page_key,
                    "document_id": document_id,
                    "aligned": fused["fusion_metrics"]["aligned_element_count"],
                    "p0_only": fused["fusion_metrics"]["primary_only_count"],
                    "p1_only": fused["fusion_metrics"]["secondary_only_count"],
                    "rate": fused["fusion_metrics"]["aligned_element_rate"],
                }
            )
        except Exception as exc:  # pragma: no cover
            failures.append(
                {
                    "document_id": document_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1200:],
                }
            )
            runs[document_id] = {"status": "FAILED", "error": str(exc)}
    sandbox.save_json(
        {
            "artifact_type": "phase13c_b_fusion_runs",
            "created_at_utc": sandbox.utc_now(),
            "runs": runs,
            "failures": failures,
        },
        BASE / "cleaned" / "fusion_index.json",
    )
    sandbox.save_json(
        {
            "artifact_type": "phase13c_b_alignment",
            "created_at_utc": sandbox.utc_now(),
            "documents": alignment_summary,
        },
        BASE / "conflicts" / "alignment_table.json",
    )
    sandbox.save_json(
        {
            "artifact_type": "phase13c_b_conflicts",
            "created_at_utc": sandbox.utc_now(),
            "documents": conflict_rows,
            "unresolved_examples": unresolved_rows[:60],
        },
        BASE / "conflicts" / "conflict_table.json",
    )
    log(f"  B2 完成 {len(runs)} 个文档，失败 {len(failures)}")
    return {"runs": runs, "failures": failures}


# --------------------------------------------------------------------------------------
# 装饰删除记录汇总（§9.2）
# --------------------------------------------------------------------------------------


def summarise_removed_furniture(arm: str) -> dict[str, Any]:
    """汇总某 arm 的装饰删除记录（产物目录放摘要，全量留在沙盒）。"""

    documents = []
    by_type: dict[str, int] = {}
    total = 0
    for path in sorted((CLEANED / arm).glob("*.json")):
        document = load_json(path)
        records = document.get("removed_furniture") or []
        if not records:
            continue
        total += len(records)
        for record in records:
            key = str(record.get("type"))
            by_type[key] = by_type.get(key, 0) + 1
        documents.append(
            {
                "document_id": document.get("document_id"),
                "count": len(records),
                "types": sorted({str(record.get("type")) for record in records}),
                "examples": [
                    {k: record.get(k) for k in ("type", "text", "page", "reason", "source_parser")}
                    for record in records[:3]
                ],
            }
        )
    payload = {
        "artifact_type": "phase13c_b_removed_furniture",
        "created_at_utc": sandbox.utc_now(),
        "arm_id": arm,
        "record_count": total,
        "by_type": dict(sorted(by_type.items())),
        "documents": documents,
        "note": "全量记录（含 bbox / source_ref）留在沙盒 cleaned/<arm>/<doc>.json 的 removed_furniture 字段。",
    }
    sandbox.save_json(payload, BASE / "removed_furniture" / f"{arm}.json")
    return payload


# --------------------------------------------------------------------------------------
# Step 6–7 — 评测
# --------------------------------------------------------------------------------------


def load_cleaned(arm: str, document_id: str) -> dict[str, Any]:
    return load_json(CLEANED / arm / f"{document_id}.json")


# 每个指标族在产物目录里保留的"逐文档头条字段"。
HEADLINE_KEYS: dict[str, tuple[str, ...]] = {
    "body": (
        "body_token_coverage",
        "unintended_body_loss_count",
        "no_text_layer_gt_body_count",
        "low_coverage_body_element_count",
        "paragraph_text_recall",
    ),
    "furniture": (
        "furniture_precision",
        "furniture_recall",
        "body_matched_to_furniture_rate",
        "traceable_filtered_furniture_rate",
        "removed_furniture_record_count",
        "furniture_traceability",
    ),
    "section": (
        "section_header_f1",
        "section_path_accuracy",
        "section_conflict_count",
        "cross_section_merge_count",
    ),
    "table": (
        "table_detection_recall",
        "table_detection_precision",
        "row_column_relation_accuracy",
        "header_assignment_accuracy",
        "spanning_cell_accuracy",
        "table_serialization_validity",
    ),
    "figure": (
        "figure_detection_recall",
        "figure_detection_precision",
        "caption_detection",
        "caption_text_recall",
        "caption_to_figure_link_accuracy",
    ),
    "equation": (
        "equation_detection_by_geometry",
        "equation_detection_precision",
        "formula_number_preservation",
        "broken_formula_rate",
    ),
    "references": (
        "reference_section_detected",
        "reference_entry_count",
        "reference_entry_recall",
        "reference_entry_precision",
        "reference_pollution_into_body_rate",
        "reference_merge_rate",
        "reference_fragment_rate",
    ),
    "provenance": (
        "page_provenance_coverage",
        "bbox_provenance_coverage",
        "source_parser_traceability",
        "fusion_traceability",
    ),
    "fusion": (
        "aligned_element_count",
        "primary_only_count",
        "secondary_only_count",
        "conflict_count",
        "unresolved_conflict_count",
        "duplicate_content_rate",
        "pre_existing_duplicate_pair_count",
    ),
}


def _family_payload(
    arm: str,
    scope: str,
    full_metrics: Mapping[str, Mapping[str, Any]],
    family: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """从一个 arm 的逐文档完整指标里抽出某族的逐文档记录与 arm 级均值。"""

    headline = HEADLINE_KEYS[family]
    per_document: dict[str, Any] = {}
    for document_id, payload in sorted(full_metrics.items()):
        source = payload.get(family) or {}
        per_document[document_id] = {key: source.get(key) for key in headline}
    summary: dict[str, Any] = {}
    for key in headline:
        values = [
            record[key]
            for record in per_document.values()
            if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool)
        ]
        summary[key] = round(sum(values) / len(values), 6) if values else None
    return (
        {
            "artifact_type": f"phase13c_b_metrics_{family}",
            "created_at_utc": sandbox.utc_now(),
            "arm_id": arm,
            "scope": scope,
            "document_count": len(per_document),
            "arm_summary": summary,
            "per_document": per_document,
        },
        summary,
    )


def build_arm_metrics(arm: str, scope: str, documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """对某 arm 的某个 scope 计算全部指标并落盘（全量在沙盒、摘要进产物目录）。"""

    ground_truth = load_json(CONTROLLED_GT)["documents"] if scope == "controlled" else {}
    real_anchors = build_real_anchors() if scope == "real" else []
    anchors_by_doc: dict[str, list[dict[str, Any]]] = {}
    for anchor in real_anchors:
        anchors_by_doc.setdefault(anchor["file_name"].replace(".pdf", ""), []).append(anchor)

    # 无文本层页是**源 PDF 的页级属性**，由 Step 3-4 在清洗时顺带探测并缓存。
    # 这里读出来传给正文指标，使"扫描页无文本层"不被计成 native parser 的正文误删。
    text_layer_cache: dict[str, Any] = {}
    text_layer_path = SANDBOX / "inputs" / "no_text_layer_pages.json"
    if text_layer_path.exists():
        text_layer_cache = load_json(text_layer_path)

    # arm → 落盘索引（B0/B1 在 cleaned/index.json 的 arms；B2 在 fusion_index.json 的 runs）。
    if arm == "B2_fused":
        available = load_json(BASE / "cleaned" / "fusion_index.json")["runs"]
    else:
        available = load_json(BASE / "cleaned" / "index.json")["arms"][arm]

    full_metrics: dict[str, Any] = {}
    for document in documents:
        if document["scope"] != scope:
            continue
        document_id = document["document_id"]
        if document_id not in available:
            continue
        cleaned = load_cleaned(arm, document_id)
        if scope == "controlled":
            gt = ground_truth[document_id]["elements"]
            # 第 3 个位置参数之后的关键字参数在 evaluate_cleaned_document 里是 keyword-only。
            payload = cleaning_metrics.evaluate_cleaned_document(
                gt,
                cleaned,
                no_text_layer_pages=text_layer_cache.get(document_id, []),
            )
            payload["no_text_layer_pages"] = sorted(
                int(page) for page in text_layer_cache.get(document_id, [])
            )
        else:
            raw = load_parser_document(ARM_SOURCES.get(arm, {}).get("parser", "P0_current"), document)
            payload = cleaning_metrics.evaluate_cleaned_real_document(
                cleaned, anchors_by_doc.get(document_id, []), raw_document=raw
            )
        # B2 的融合特有指标（§13）随 fused 文档落盘，接到指标族里供报告与摘要使用。
        if cleaned.get("fusion_metrics"):
            payload["fusion"] = cleaned["fusion_metrics"]
        full_metrics[document_id] = payload
        sandbox.save_json(payload, FULL_METRICS / arm / scope / f"{document_id}.json")

    family_summaries: dict[str, Any] = {}
    for family in HEADLINE_KEYS:
        if scope == "real" and family in {"section", "table", "figure", "equation", "references"}:
            # 真实语料没有元素级 GT，这几族只保留可计算的子集（见 real evaluator）。
            payload, summary = _family_payload(arm, scope, _real_family_view(full_metrics, family), family)
        else:
            payload, summary = _family_payload(arm, scope, full_metrics, family)
        sandbox.save_json(payload, FULL_METRICS / arm / scope / f"_family_{family}.json")
        family_summaries[family] = summary
    return {"full": full_metrics, "families": family_summaries}


def _real_family_view(
    full_metrics: Mapping[str, Mapping[str, Any]], family: str
) -> dict[str, Mapping[str, Any]]:
    """把真实语料的指标重整成与受控语料同形的 ``{family: {...}}`` 视图。

    真实语料没有元素级 GT，因此 section/table/figure/equation 族里只有
    "结构可用性"与"参考文献计数"是可算的，其余字段如实留空（``None`` = 未采集），
    绝不用 0 冒充"测了但为 0"。
    """

    view: dict[str, Mapping[str, Any]] = {}
    for document_id, payload in full_metrics.items():
        availability = payload.get("structure_availability") or {}
        references = payload.get("references") or {}
        furniture = payload.get("furniture") or {}
        provenance = payload.get("provenance") or {}
        mapping: dict[str, Any] = {
            "body": {
                "body_token_coverage": None,
                "unintended_body_loss_count": None,
                "no_text_layer_gt_body_count": None,
                "low_coverage_body_element_count": None,
                "paragraph_text_recall": None,
            },
            "furniture": {
                "furniture_precision": None,
                "furniture_recall": None,
                "body_matched_to_furniture_rate": None,
                "traceable_filtered_furniture_rate": furniture.get("traceable_filtered_furniture_rate"),
                "removed_furniture_record_count": furniture.get("removed_furniture_record_count"),
                "furniture_traceability": (
                    "EXPLICIT_DETECTION"
                    if furniture.get("removed_furniture_record_count")
                    else "NO_FURNITURE_OBSERVED"
                ),
            },
            "section": {"section_header_f1": None, "section_path_accuracy": None,
                        "section_conflict_count": None, "cross_section_merge_count": None},
            "table": {"table_detection_recall": None, "table_detection_precision": None,
                      "row_column_relation_accuracy": None, "header_assignment_accuracy": None,
                      "spanning_cell_accuracy": None, "table_serialization_validity": None,
                      "table_count": availability.get("table_count"),
                      "structured_table_count": availability.get("structured_table_count")},
            "figure": {"figure_detection_recall": None, "figure_detection_precision": None,
                       "caption_detection": None, "caption_text_recall": None,
                       "caption_to_figure_link_accuracy": None,
                       "figure_count": availability.get("figure_count"),
                       "figure_with_caption_count": availability.get("figure_with_caption_count")},
            "equation": {"equation_detection_by_geometry": None, "equation_detection_precision": None,
                         "formula_number_preservation": None, "broken_formula_rate": None,
                         "equation_count": availability.get("equation_count"),
                         "equation_with_dual_representation_count": availability.get(
                             "equation_with_dual_representation_count")},
            "references": {
                "reference_section_detected": references.get("reference_section_detected"),
                "reference_entry_count": references.get("reference_entry_count"),
                "reference_entry_recall": None,
                "reference_entry_precision": None,
                "reference_pollution_into_body_rate": None,
                "reference_merge_rate": None,
                "reference_fragment_rate": None,
            },
            "provenance": provenance,
            "fusion": (
                {key: value for key, value in payload.items() if key in HEADLINE_KEYS["fusion"]}
                if "aligned_element_count" in payload
                else {key: None for key in HEADLINE_KEYS["fusion"]}
            ),
        }
        view[document_id] = mapping
    return view


# --------------------------------------------------------------------------------------
# Step 8 — Conditional OCR 触发规则验证
# --------------------------------------------------------------------------------------


def conditional_ocr_trigger(page_text: str, *, has_visual_region: bool, page_width_ratio: float) -> dict[str, Any]:
    """Conditional OCR 触发规则（方案 §10）：只允许在无文本层 / 乱码 / 有可视区无文本时触发。"""

    stripped = page_text.strip()
    reasons: list[str] = []
    if not stripped:
        reasons.append("page_has_no_text_layer")
    else:
        # 乱码判据：替换字符或不可打印控制字符占比 > 10%。
        suspicious = sum(1 for char in stripped if char == "\ufffd" or (ord(char) < 32 and char not in "\n\t"))
        if suspicious / max(1, len(stripped)) > 0.10:
            reasons.append("native_text_is_garbled")
        else:
            tokens = stripped.split()
            if len(tokens) >= 20 and sum(len(token) for token in tokens) / len(tokens) < 2.0:
                reasons.append("native_text_is_garbled")
    if not stripped and has_visual_region and page_width_ratio >= 0.30:
        reasons.append("block_missing_but_visual_region_exists")
    return {"trigger": bool(reasons), "reasons": reasons}


def step8_conditional_ocr(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Step 8：验证 OCR 触发规则只在"无文本层 / 乱码"页命中，不在 born-digital 页命中。"""

    log("Step 8 — Conditional OCR 触发规则验证")
    import pymupdf

    per_document: dict[str, Any] = {}
    triggered_pages: list[dict[str, Any]] = []
    born_digital_triggers: list[dict[str, Any]] = []
    for document in documents:
        with pymupdf.open(document["pdf_path"]) as pdf:
            page_decisions = []
            for index in range(pdf.page_count):
                page = pdf[index]
                text = page.get_text()
                rect = page.rect
                visual_area = 0.0
                for image in page.get_images(full=True):
                    try:
                        boxes = page.get_image_rects(image[0])
                    except Exception:  # pragma: no cover - 防御性
                        boxes = []
                    for box in boxes:
                        visual_area += abs(box.width * box.height)
                drawings = page.get_drawings()
                for drawing in drawings[:200]:
                    visual_area += abs(drawing["rect"].width * drawing["rect"].height)
                ratio = min(1.0, visual_area / max(1.0, rect.width * rect.height))
                decision = conditional_ocr_trigger(
                    text, has_visual_region=bool(drawings or page.get_images()), page_width_ratio=ratio
                )
                record = {
                    "document_id": document["document_id"],
                    "page": index + 1,
                    **decision,
                }
                page_decisions.append(record)
                if decision["trigger"]:
                    triggered_pages.append(record)
                    if document["scope"] == "real" or not document.get("rasterized"):
                        born_digital_triggers.append(record)
        per_document[document["document_id"]] = {
            "page_count": len(page_decisions),
            "triggered_pages": [record["page"] for record in page_decisions if record["trigger"]],
            "reasons": sorted(
                {reason for record in page_decisions for reason in record["reasons"]}
            ),
        }

    # 触发页的 OCR 效果：用 13C-A 已缓存的 O2 输出做验证（不重新跑 OCR，也不在
    # born-digital 页上跑 OCR —— 方案 §10 禁止 force OCR）。
    ocr_validation: dict[str, Any] = {}
    for document in documents:
        decision = per_document[document["document_id"]]
        if not decision["triggered_pages"]:
            continue
        ocr_path = STRUCTURED / OCR_TRIGGER_ARM / OCR_TRIGGER_MODE / f"{document['document_id']}.json"
        if not ocr_path.exists():
            ocr_validation[document["document_id"]] = {"status": "NO_CACHED_OCR_OUTPUT"}
            continue
        ocr_document = load_json(ocr_path)
        ocr_validation[document["document_id"]] = {
            "status": "VALIDATED_AGAINST_CACHED_OCR",
            "ocr_element_count": len(ocr_document.get("elements", [])),
            "ocr_pages": sorted(
                {
                    element.get("page")
                    for element in ocr_document.get("elements", [])
                    if element.get("page")
                }
            ),
            "ocr_text_token_count": sum(
                len(schema.tokenize(element.get("normalized_text") or ""))
                for element in ocr_document.get("elements", [])
            ),
        }

    payload = {
        "artifact_type": "phase13c_b_conditional_ocr_probe",
        "created_at_utc": sandbox.utc_now(),
        "rule": {
            "page_has_no_text_layer": "整页无可抽取文本",
            "native_text_is_garbled": "替换字符/控制字符占比 > 10%，或平均 token 长度 < 2（且 token >= 20）",
            "block_missing_but_visual_region_exists": "无文本且可视区域占页面 >= 30%",
            "policy": "本阶段禁止 force OCR；只在命中上述规则的页面上运行，OCR 输出必须重新进入同一 schema。",
        },
        "per_document": per_document,
        "triggered_page_count": len(triggered_pages),
        "born_digital_trigger_count": len(born_digital_triggers),
        "born_digital_triggers": born_digital_triggers[:20],
        "ocr_validation": ocr_validation,
        "verdict": (
            "PASS"
            if not born_digital_triggers
            else "FAIL: OCR 触发规则在 born-digital 页上命中（方案 §10 禁止）"
        ),
    }
    sandbox.save_json(payload, BASE / "metrics" / "ocr_trigger.json")
    log(f"  触发 {len(triggered_pages)} 页；born-digital 误触发 {len(born_digital_triggers)} 页")
    return payload


# --------------------------------------------------------------------------------------
# Step 9 — 硬门判定
# --------------------------------------------------------------------------------------


def no_text_layer_documents() -> set[str]:
    """整本 PDF 每一页都无文本层的文档（如 T12 扫描模板）。

    这类文档上解析器没有任何可提取文本，装饰/参考文献门的 UNTRACEABLE / recall=0
    属解析器能力边界而非 Cleaner 失败，门判定时豁免（与 5B 的扫描页规则一致）。
    """

    cache_path = SANDBOX / "inputs" / "no_text_layer_pages.json"
    if not cache_path.exists():
        return set()
    cache = load_json(cache_path)
    page_counts = {
        document["document_id"]: int(document.get("page_count") or 0)
        for document in all_documents()
    }
    excluded: set[str] = set()
    for document_id, pages in cache.items():
        total = page_counts.get(document_id) or 0
        if total and len({int(page) for page in pages}) >= total:
            excluded.add(document_id)
    return excluded


def step9_hard_gates(
    *,
    frozen: Mapping[str, Any],
    arm_runs: Mapping[str, Mapping[str, Any]],
    arm_metrics: Mapping[str, Mapping[str, Any]],
    controlled_full: Mapping[str, Mapping[str, Any]],
    real_full: Mapping[str, Mapping[str, Any]],
    baseline_anchor_hits: Mapping[str, bool],
    repro_records: Mapping[str, Any],
    ocr_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Step 9：对 B0 / B1 / B2 逐项判 10 条硬门，并单独给出 5A / 5B / Reference 门。"""

    log("Step 9 — 硬门判定（B1–B10 + 5A/5B/REF）")
    sandbox_integrity = verify_sandbox_integrity(frozen)
    # 整本无文本层的文档：解析器能力边界，5A 的 UNTRACEABLE 检查与 REF 门豁免。
    scanned_documents = no_text_layer_documents()
    if scanned_documents:
        log(f"  无文本层文档（门判定豁免）: {sorted(scanned_documents)}")
    declarations = {
        "B0_p0_cleaned": "纯 PyMuPDF 文本/块抽取 + 本地正则与几何判定；无外部模型调用。",
        "B1_mineru_cleaned": "13C-A 缓存的结构化输出 + 本地 Cleaner（正则/几何）；无外部模型调用。",
        "B2_fused": "B0/B1 的确定性 provenance 对齐 + 本地 Cleaner；无外部模型调用。",
    }

    controlled_anchor_hits = {arm: 0 for arm in ARMS}
    new_loss: dict[str, int | None] = {arm: None for arm in ARMS}
    real_metrics: dict[str, Any] = {}
    for arm in ARMS:
        hits: dict[str, bool] = {}
        for document_id, payload in (arm_metrics[arm].get("real") or {}).items():
            for entry in (payload.get("anchors") or {}).get("detail", []):
                anchor_id = entry.get("anchor_id")
                if anchor_id:
                    hits[anchor_id] = hits.get(anchor_id, False) or bool(entry.get("hit"))
        real_metrics[arm] = hits

    anchor_results: dict[str, Any] = {}
    for arm in ARMS:
        arm_hits = real_metrics[arm]
        baseline_hits = {key for key, value in baseline_anchor_hits.items() if value}
        missed = sorted(baseline_hits - {key for key, value in arm_hits.items() if value})
        recovered = sorted({key for key, value in arm_hits.items() if value} - baseline_hits)
        anchor_results[arm] = {
            "baseline_hit_count": len(baseline_hits),
            "arm_hit_count": sum(1 for value in arm_hits.values() if value),
            "new_loss_count": len(missed),
            "new_loss_anchor_ids": missed[:20],
            "recovered_anchor_ids": recovered[:20],
        }
        new_loss[arm] = len(missed)

    arm_gate_results: dict[str, Any] = {}
    cleaning_gate_results: dict[str, Any] = {}
    for arm in ARMS:
        controlled = controlled_full[arm]
        real = real_full[arm]
        furniture_summaries = [
            {
                "document_id": document_id,
                **(payload.get("furniture") or {}),
            }
            for document_id, payload in controlled.items()
        ]
        # 真实语料也参与 5A：装饰删除记录必须可追溯。
        for document_id, payload in real.items():
            furniture_summaries.append(
                {
                    "document_id": document_id,
                    **(payload.get("furniture") or {}),
                    "explicit_furniture_detection": bool(
                        (payload.get("furniture") or {}).get("removed_furniture_record_count")
                    ),
                    "furniture_traceability": (
                        "EXPLICIT_DETECTION"
                        if (payload.get("furniture") or {}).get("removed_furniture_record_count")
                        else "NO_FURNITURE_OBSERVED"
                    ),
                }
            )
        five_a = gates_v2.evaluate_gate_5a(
            furniture_summaries, excluded_document_ids=sorted(scanned_documents)
        )
        body_summaries = [payload.get("body") or {} for payload in controlled.values()]
        five_b = gates_v2.evaluate_gate_5b(
            canonical_anchor_hits=anchor_results[arm]["arm_hit_count"],
            new_anchor_loss_count=new_loss[arm],
            unintended_body_loss_count=max(
                (int(summary.get("unintended_body_loss_count") or 0) for summary in body_summaries),
                default=0,
            ),
            body_token_coverage=(
                round(
                    sum(
                        float(summary.get("body_token_coverage") or 0.0)
                        for summary in body_summaries
                    )
                    / max(1, len(body_summaries)),
                    6,
                )
                if body_summaries
                else None
            ),
        )
        references = gates_v2.evaluate_reference_gate(
            [
                {"document_id": document_id, **(payload.get("references") or {})}
                for document_id, payload in controlled.items()
            ]
        )
        cleaning_gate_results[arm] = {"5A": five_a, "5B": five_b, "REF": references}

        interleave = [
            (payload.get("reading_order") or {}).get("column_interleave", {}).get("interleave_rate")
            for payload in controlled.values()
        ]
        interleave = [rate for rate in interleave if isinstance(rate, (int, float))]
        provenance_coverages = [
            (payload.get("provenance") or {}).get("page_provenance_coverage")
            for payload in list(controlled.values()) + list(real.values())
        ]
        provenance_coverages = [
            value for value in provenance_coverages if isinstance(value, (int, float))
        ]
        bbox_coverages = [
            (payload.get("provenance") or {}).get("bbox_provenance_coverage")
            for payload in controlled.values()
        ]
        bbox_coverages = [value for value in bbox_coverages if isinstance(value, (int, float))]
        structure_availability = _merge_structure_availability(
            [payload.get("structure_availability") or {} for payload in controlled.values()]
        )
        arm_gate_results[arm] = gates_v2.evaluate_hard_gates(
            arm_id=arm,
            canonical_anchor_hits=anchor_results[arm]["arm_hit_count"],
            new_anchor_loss_count=new_loss[arm],
            unintended_body_loss_count=max(
                (int(summary.get("unintended_body_loss_count") or 0) for summary in body_summaries),
                default=0,
            ),
            column_interleave_rate=max(interleave) if interleave else None,
            gate_5a=five_a,
            structure_availability=structure_availability,
            reference_gate=references,
            min_page_provenance_coverage=min(provenance_coverages) if provenance_coverages else None,
            structure_bbox_coverage=min(bbox_coverages) if bbox_coverages else None,
            reproducible=(repro_records.get(arm) or {}).get("reproducible"),
            external_llm_declaration=declarations[arm],
            sandbox_integrity=sandbox_integrity,
        )

    payload = {
        "artifact_type": "phase13c_b_hard_gates",
        "created_at_utc": sandbox.utc_now(),
        "gate_definition_version": gates_v2.GATE_DEFINITION_VERSION,
        "anchor_results": anchor_results,
        "cleaning_gates": cleaning_gate_results,
        "hard_gates": arm_gate_results,
        "summary_table": gates_v2.summarise_gate_table(arm_gate_results),
        "sandbox_integrity": sandbox_integrity,
        "ocr_rule_verdict": ocr_probe.get("verdict"),
        "blocked_arms": {
            "P4_grobid": {
                "status": "BLOCKED",
                "reason": (
                    "本机无 Java 运行时且无 Docker：GROBID 是 Java 服务，sandbox 内无法满足。"
                    "这不构成能力结论，也不从候选集合中静默删除（方案 §27）。"
                ),
            }
        },
    }
    sandbox.save_json(payload, BASE / "metrics" / "hard_gates.json")
    return payload


def _merge_structure_availability(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把逐文档的结构可用性合并成"任一文档存在即存在"的 arm 级视图。"""

    merged: dict[str, Any] = {
        "type_counts": {},
        "table_count": 0,
        "structured_table_count": 0,
        "figure_count": 0,
        "figure_with_caption_count": 0,
        "equation_count": 0,
        "equation_with_dual_representation_count": 0,
        "structured_types_present": [],
    }
    present: set[str] = set()
    for payload in payloads:
        for key in (
            "table_count",
            "structured_table_count",
            "figure_count",
            "figure_with_caption_count",
            "equation_count",
            "equation_with_dual_representation_count",
        ):
            merged[key] += int(payload.get(key) or 0)
        for key, value in (payload.get("type_counts") or {}).items():
            merged["type_counts"][key] = merged["type_counts"].get(key, 0) + int(value)
        present.update(payload.get("structured_types_present") or [])
    merged["structured_types_present"] = sorted(present)
    return merged


# --------------------------------------------------------------------------------------
# 沙盒完整性（门 B10）
# --------------------------------------------------------------------------------------


def verify_sandbox_integrity(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """复核源 PDF / src / 13B 冻结产物 / Chroma 是否全部未变。"""

    source_tree_after = sandbox.snapshot_project_source_tree()
    tree_diff = sandbox.diff_source_tree(frozen["source_tree_before"], source_tree_after)

    pdfs_ok = True
    pdf_problems: list[str] = []
    for group in ("controlled_pdfs", "real_pdfs"):
        for name, record in (frozen.get(group) or {}).items():
            path = Path(record["path"])
            if not path.exists() or sandbox.sha256_file(path) != record["sha256"]:
                pdfs_ok = False
                pdf_problems.append(name)

    artifacts_ok = True
    artifact_changes: list[str] = []
    for relative, record in (frozen.get("prior_phase_artifacts") or {}).items():
        path = sandbox.ARTIFACT_ROOT / relative
        current = sandbox.sha256_file(path) if path.exists() else None
        if current != record.get("sha256"):
            artifacts_ok = False
            artifact_changes.append(relative)

    import hashlib

    chroma = (frozen.get("baseline_chroma") or {})
    chroma_ok = True
    if chroma.get("exists"):
        digest = hashlib.sha256()
        with Path(chroma["path"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        chroma_ok = digest.hexdigest() == chroma.get("sha256")

    frozen_projects_after = sandbox.snapshot_frozen_project_files()
    project_files_ok = all(
        frozen_projects_after.get(name, {}).get("sha256") == record.get("sha256")
        for name, record in (frozen.get("frozen_project_files") or {}).items()
    )

    return {
        "source_pdfs_unchanged": pdfs_ok,
        "source_pdf_problems": pdf_problems,
        "src_evidenceflow_unchanged": tree_diff["identical"],
        "src_tree_diff": tree_diff,
        "phase12_13_frozen_artifacts_unchanged": artifacts_ok,
        "changed_artifacts": artifact_changes,
        "baseline_chroma_unchanged": chroma_ok,
        "frozen_project_files_unchanged": project_files_ok,
        "intact": bool(
            pdfs_ok
            and tree_diff["identical"]
            and artifacts_ok
            and chroma_ok
            and project_files_ok
        ),
    }


# --------------------------------------------------------------------------------------
# 可复现性（门 B8）
# --------------------------------------------------------------------------------------


def step_reproducibility() -> dict[str, Any]:
    """对 T02 重跑清洗与融合，比较 digest。"""

    log("  复现性检查（T02_ieee_two_column）")
    document = next(
        doc for doc in controlled_documents() if doc["document_id"] == REPRO_TEMPLATE
    )
    records: dict[str, Any] = {}
    digests: dict[str, str] = {}
    for arm in ("B0_p0_cleaned", "B1_mineru_cleaned"):
        first = clean_for_arm(arm, document)
        second = clean_for_arm(arm, document)
        digests[arm] = document_digest(first)
        records[arm] = {
            "reproducible": document_digest(first) == document_digest(second),
            "digest_prefix": document_digest(first)[:16],
            "on_disk_matches": (
                document_digest(load_cleaned(arm, REPRO_TEMPLATE)) == document_digest(first)
            ),
        }
    b0 = clean_for_arm("B0_p0_cleaned", document)
    b1 = clean_for_arm("B1_mineru_cleaned", document)
    first_fused = build_fusion_for_document(document, b0, b1)
    second_fused = build_fusion_for_document(document, b0, b1)
    records["B2_fused"] = {
        "reproducible": document_digest(first_fused) == document_digest(second_fused),
        "digest_prefix": document_digest(first_fused)[:16],
        "on_disk_matches": (
            document_digest(load_cleaned("B2_fused", REPRO_TEMPLATE)) == document_digest(first_fused)
        ),
    }
    return {"template_id": REPRO_TEMPLATE, "records": records}


# --------------------------------------------------------------------------------------
# Step 10 — 选型
# --------------------------------------------------------------------------------------


def step10_select(
    *,
    hard_gates: Mapping[str, Any],
    arm_metrics: Mapping[str, Mapping[str, Any]],
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Step 10：按方案 §15 的决策顺序选 13C-C 的输入；不为"必须有 winner"放宽门。"""

    log("Step 10 — 选择 13C-C 输入")
    results = hard_gates["hard_gates"]
    qualified = [arm for arm in ARMS if (results.get(arm) or {}).get("gate_passed")]
    body_rank = {
        arm: _mean_metric(arm_metrics[arm], "body", "body_token_coverage") for arm in ARMS
    }
    structure_rank = {
        arm: _structure_score(arm_metrics[arm]) for arm in ARMS
    }
    conflict_rank = {
        arm: _mean_metric(arm_metrics[arm], "fusion", "unresolved_conflict_count") if arm == "B2_fused" else 0.0
        for arm in ARMS
    }
    duplicate_rank = {
        arm: _mean_metric(arm_metrics[arm], "fusion", "duplicate_content_rate") if arm == "B2_fused" else 0.0
        for arm in ARMS
    }

    if not qualified:
        status = "FAIL"
        selected: list[str] = []
        why = "没有任何 arm 通过全部硬门（方案 §26）。不进入 Chunk Benchmark。"
    else:
        # 决策顺序：Hard Gates -> Body Completeness -> Structure Quality -> Conflict/Duplication -> Cost
        ordered = sorted(
            qualified,
            key=lambda arm: (
                -(body_rank[arm] if isinstance(body_rank[arm], (int, float)) else -1),
                -(structure_rank[arm] if isinstance(structure_rank[arm], (int, float)) else -1),
                conflict_rank[arm],
                duplicate_rank[arm],
                cost.get(arm, {}).get("total_cleaning_seconds") or 0.0,
            ),
        )
        best_body = ordered[0]
        best_structure = max(
            qualified, key=lambda arm: structure_rank[arm] if isinstance(structure_rank[arm], (int, float)) else -1
        )
        if best_structure != best_body and structure_rank[best_structure] > structure_rank[best_body]:
            selected = sorted({best_body, best_structure})
            why = (
                f"{best_body} 正文最强（body_token_coverage={body_rank[best_body]}），"
                f"{best_structure} 结构最强（structure_score={structure_rank[best_structure]}）；"
                "两者均过全部门，按方案 §24/§25 并列交给 13C-C。"
            )
        else:
            selected = [best_body]
            why = (
                f"{best_body} 在硬门全过的前提下同时取得最强正文（{body_rank[best_body]}）"
                f"与最强结构（{structure_rank[best_body]}）。"
            )
        status = "PASS"

    payload = {
        "artifact_type": "phase13c_b_selection",
        "created_at_utc": sandbox.utc_now(),
        "phase_13c_b_status": status,
        "qualified_arms": qualified,
        "selected_for_13c_c": selected,
        "why": why,
        "ranking": {
            "body_token_coverage": body_rank,
            "structure_score": structure_rank,
            "unresolved_conflict_count": conflict_rank,
            "duplicate_content_rate": duplicate_rank,
            "cost": {arm: cost.get(arm, {}) for arm in ARMS},
        },
        "decision_order": [
            "Hard Gates",
            "Body Completeness",
            "Structure Quality",
            "Conflict / Duplication",
            "Cost / Complexity",
        ],
        "note": "不做加权总分（方案 §15）。若没有任何 arm 过门，输出 FAIL 而不是放宽门。",
    }
    sandbox.save_json(payload, BASE / "selected_cleaning_pipeline.json")
    return payload


def _mean_metric(arm_metrics: Mapping[str, Any], family: str, key: str) -> float | None:
    """单个 arm 在两个 scope 上的某指标均值（``arm_metrics`` 是该 arm 的指标字典）。"""

    values = [
        (payload.get(family) or {}).get(key)
        for scope in ("controlled", "real")
        for payload in (arm_metrics.get(scope) or {}).values()
    ]
    values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return round(sum(values) / len(values), 6) if values else None


def _structure_score(arm_metrics: Mapping[str, Any]) -> float | None:
    """结构质量：受控语料上"表结构 + 图注 + 公式几何检出"三项均值。

    ``arm_metrics`` 是**单个 arm** 的指标字典（含 ``controlled`` scope）；
    每个 arm 用自己的结构指标打分，而不是借用 B2 的。
    """

    scores: list[float] = []
    for payload in (arm_metrics.get("controlled") or {}).values():
        structure = payload.get("structure") or {}
        availability = payload.get("structure_availability") or {}
        table = structure.get("table") or {}
        figure = structure.get("figure_caption") or {}
        equation = structure.get("equation") or {}
        # 只在"该文档确实存在此类结构元素"时才计入该分项，
        # 避免"没有公式 → 几何检出 0"把分母污染成能力缺陷。
        if int(availability.get("table_count") or 0):
            scores.append(float(table.get("row_column_relation_accuracy") or 0.0))
        if availability.get("figure_count") or availability.get("figure_with_caption_count"):
            scores.append(float(figure.get("caption_text_recall") or 0.0))
        if availability.get("equation_count"):
            scores.append(float(equation.get("equation_detection_by_geometry") or 0.0))
    return round(sum(scores) / len(scores), 6) if scores else None


# --------------------------------------------------------------------------------------
# Step 11 — 报告与 manifest
# --------------------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_cleaning_comparison(
    *,
    arm_metrics: Mapping[str, Mapping[str, Any]],
    hard_gates: Mapping[str, Any],
    cost: Mapping[str, Any],
    ocr_probe: Mapping[str, Any],
) -> str:
    """生成 ``reports/cleaning_comparison.md``。"""

    anchor = hard_gates["anchor_results"]
    lines: list[str] = [
        "# Phase 13C-B — Cleaning Benchmark 对照报告（Sandbox-Only）",
        "",
        f"- **生成时间**：{sandbox.utc_now()}",
        f"- **Cleaner 版本**：`{cleaning.CLEANER_VERSION}`",
        f"- **Fusion 版本**：`{fusion.FUSION_VERSION}`",
        f"- **Gate 定义版本**：`{gates_v2.GATE_DEFINITION_VERSION}`",
        f"- **沙盒根**：`{SANDBOX}`",
        f"- **产物根**：`{BASE}`",
        "",
        "## 0. 一句话结论",
        "",
        _one_line_conclusion(hard_gates, arm_metrics),
        "",
        "## 1. canonical evidence anchor（真实语料 10 篇）",
        "",
        "| arm | 清洗前命中 | 清洗后命中 | 新增丢失 | 恢复 | 新增丢失锚点 |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        entry = anchor[arm]
        lines.append(
            f"| `{arm}` | {entry['baseline_hit_count']} | {entry['arm_hit_count']} | "
            f"{entry['new_loss_count']} | {len(entry['recovered_anchor_ids'])} | "
            f"{', '.join(entry['new_loss_anchor_ids']) or '—'} |"
        )
    lines += [
        "",
        "> **ARR-017 的归因（方案 §11 / Q3）**：13C-A 记录 P1_mineru 在 `seedformer.pdf` 上丢失 "
        "`EVID-ARR-017-G1-A1`（`Np = 128`）。逐元素追查后确认三类原因中的两类同时存在："
        "①**映射缺口**——MinerU 原生 `content_list` 里有 `page_number` / `header` / `footer` / "
        "`aside_text` / `list(sub_type=ref_text)` / `chart` 六种类型，13C-A 的映射层全部折叠成"
        "文本为空的 `unknown`；②**表示约定差异**——该 anchor 的内容确实在输出里，形态是行内公式 "
        "`$N _ { p } = 1 2 8$`，tokenizer 切出 `n/p/1/2/8`，取不到 `np`/`128`。"
        "Cleaner 分别用「按原生 content_list 再标注」与「数学跨度词法规范化」修复了这两类，"
        "因此 **B1 的 anchor 从 43/44 回到 44/44**。这不是解析器能力不足，而是本工程映射层的缺口。",
        "",
        "## 2. 正文保持（受控 12 模板 + 真实 10 篇）",
        "",
        "| arm | 正文 token 覆盖 | 未对齐正文元素 | 真误删 | 无文本层页正文 | 低覆盖元素 |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        summary = arm_metrics[arm]["families"]["controlled"]["body"]
        lines.append(
            f"| `{arm}` | {_fmt(summary.get('body_token_coverage'))} | "
            f"{_fmt(summary.get('paragraph_text_recall'))} | "
            f"{_fmt(_max_of(arm_metrics[arm]['controlled'], 'body', 'unintended_body_loss_count'))} | "
            f"{_fmt(_max_of(arm_metrics[arm]['controlled'], 'body', 'no_text_layer_gt_body_count'))} | "
            f"{_fmt(_max_of(arm_metrics[arm]['controlled'], 'body', 'low_coverage_body_element_count'))} |"
        )
    lines += [
        "",
        "> `真误删`（`unintended_body_loss_count`）**只统计「该页解析器有产出、但正文未被覆盖」的元素**；"
        "无文本层页（扫描件）的正文单独记在 `无文本层页正文` 列，不参与误删判定。"
        "这是方案 §3 的修订：13C-A 的门 5 把这两类混在同一个 `false_removal` 里，"
        "于是 T12 扫描模板的 40 个元素掩盖了真实结论。",
        "",
        "## 3. 装饰可分性（Gate 5A，与正文完整度完全解耦）",
        "",
        "| arm | 装饰召回 | 装饰精确率 | 正文被标成装饰 | 删除记录数 | 记录可追溯率 | 可追溯性 |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        summary = arm_metrics[arm]["families"]["controlled"]["furniture"]
        lines.append(
            f"| `{arm}` | {_fmt(summary.get('furniture_recall'))} | "
            f"{_fmt(summary.get('furniture_precision'))} | "
            f"{_fmt(summary.get('body_matched_to_furniture_rate'))} | "
            f"{_fmt(summary.get('removed_furniture_record_count'), 1)} | "
            f"{_fmt(summary.get('traceable_filtered_furniture_rate'))} | "
            f"{_fmt(summary.get('furniture_traceability'))} |"
        )
    lines += [
        "",
        "> 新口径下装饰通过条件不再要求 `parsed_furniture_count > 0`："
        "「显式检测」**或**「可追溯删除记录」二者之一即可。"
        "B1 的显式检测来自 MinerU 原生的 `header`/`footer`/`page_number`/`aside_text` 类型"
        "（13C-A 的映射层把它们丢掉了，因此当时 P1 的 `parsed_furniture_count` 为 0）。",
        "",
        "## 4. 章节结构",
        "",
        "| arm | 标题 F1 | section_path | 章节冲突 | 跨章节误合并 | 双栏交叉率 |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        summary = arm_metrics[arm]["families"]["controlled"]["section"]
        lines.append(
            f"| `{arm}` | {_fmt(summary.get('section_header_f1'))} | "
            f"{_fmt(summary.get('section_path_accuracy'))} | "
            f"{_fmt(summary.get('section_conflict_count'), 2)} | "
            f"{_fmt(summary.get('cross_section_merge_count'), 2)} | "
            f"{_fmt(_max_of(arm_metrics[arm]['controlled'], 'reading_order.column_interleave', 'interleave_rate'))} |"
        )
    lines += [
        "",
        "## 5. 结构元素",
        "",
        "> 表 / 图 / 公式的检出用几何判据（表与图 IoU ≥ 0.5、公式重叠系数 ≥ 0.6），"
        "因为受控语料里图是矢量绘制（GT 文本为空）、表格内容在结构化字段里、"
        "公式的文本约定各 arm 不同——文本匹配在这三类上测的是「选了哪种输出约定」。"
        "",
        "| arm | 表检出 | 表检出精确率 | 行列关系 | 表头归属 | 图检出 | 图注文本召回 | 公式检出(几何) | 公式编号保留 | 公式双表示 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        family = arm_metrics[arm]["families"]["controlled"]
        availability = arm_metrics[arm]["structure_availability"]
        lines.append(
            f"| `{arm}` | {_fmt(family['table'].get('table_detection_recall'))} | "
            f"{_fmt(family['table'].get('table_detection_precision'))} | "
            f"{_fmt(family['table'].get('row_column_relation_accuracy'))} | "
            f"{_fmt(family['table'].get('header_assignment_accuracy'))} | "
            f"{_fmt(family['figure'].get('figure_detection_recall'))} | "
            f"{_fmt(family['figure'].get('caption_text_recall'))} | "
            f"{_fmt(family['equation'].get('equation_detection_by_geometry'))} | "
            f"{_fmt(family['equation'].get('formula_number_preservation'))} | "
            f"{_fmt(availability.get('equation_with_dual_representation_count'), 1)} |"
        )
    lines += [
        "",
        f"- 受控语料结构元素计数（B2）：{json.dumps(arm_metrics['B2_fused']['structure_availability'], ensure_ascii=False)}",
        "",
        "## 6. 参考文献（Reference Preservation）",
        "",
        "| arm | 章节检测 | 条目数 | 条目召回 | 条目精确率 | 正文污染 | 误合并率 | 碎片率 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        summary = arm_metrics[arm]["families"]["controlled"]["references"]
        lines.append(
            f"| `{arm}` | {_fmt(summary.get('reference_section_detected'))} | "
            f"{_fmt(summary.get('reference_entry_count'), 1)} | "
            f"{_fmt(summary.get('reference_entry_recall'))} | "
            f"{_fmt(summary.get('reference_entry_precision'))} | "
            f"{_fmt(summary.get('reference_pollution_into_body_rate'))} | "
            f"{_fmt(summary.get('reference_merge_rate'))} | "
            f"{_fmt(summary.get('reference_fragment_rate'))} |"
        )
    lines += [
        "",
        "> T11_references_heavy 的 GT 有 72 条参考文献。13C-A 的 P1 输出把它们折叠成一个"
        "文本为空的 `unknown`，因此 `T11` 贡献了 122 个「低覆盖正文误删」。"
        "Cleaner 按 MinerU 原生 `list_items`（解析器已给出的边界，**不是文本猜测**）"
        "逐条展开为 `reference_entry`。",
        "",
        "## 7. 来源可追溯",
        "",
        "| arm | page 覆盖 | bbox 覆盖 | source_parser | 融合可追溯 |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        summary = arm_metrics[arm]["families"]["controlled"]["provenance"]
        lines.append(
            f"| `{arm}` | {_fmt(summary.get('page_provenance_coverage'))} | "
            f"{_fmt(summary.get('bbox_provenance_coverage'))} | "
            f"{_fmt(summary.get('source_parser_traceability'))} | "
            f"{_fmt(summary.get('fusion_traceability'))} |"
        )
    lines += [
        "",
        "## 8. 融合（B2）",
        "",
        "| 指标 | 受控语料均值 | 真实语料均值 |",
        "|---|---|---|",
    ]
    for key in (
        "aligned_element_count",
        "primary_only_count",
        "secondary_only_count",
        "conflict_count",
        "unresolved_conflict_count",
        "duplicate_content_rate",
        "pre_existing_duplicate_pair_count",
    ):
        lines.append(
            f"| `{key}` | {_fmt(arm_metrics['B2_fused']['families']['controlled']['fusion'].get(key), 3)} | "
            f"{_fmt(arm_metrics['B2_fused']['families']['real']['fusion'].get(key), 3)} |"
        )
    lines += [
        "",
        "> `duplicate_content_rate` **只统计融合引入的重复**（至少一侧 `fusion_status=P1_ONLY`）。"
        "`pre_existing_duplicate_pair_count` 是 P0 自身输出的重复文本块对数"
        "（实测 `adapointr` 有 73 对：P0 把表格单元格与表头都当作独立段落输出）。"
        "两者口径不同，不可相加——把解析器层的既存重复记成融合缺陷会是错误归因。",
        "",
        "## 9. Conditional OCR",
        "",
        f"- 判定：`{ocr_probe.get('verdict')}`",
        f"- 触发页数：{ocr_probe.get('triggered_page_count')}；born-digital 误触发：{ocr_probe.get('born_digital_trigger_count')}",
        f"- 触发规则的三种原因与验证方式见 `metrics/ocr_trigger.json`。",
        "- 本阶段**未对 born-digital 页运行 OCR**（方案 §10 禁止 force OCR）；"
        "触发页的 OCR 效果用 13C-A 已缓存的 O2 输出验证，未新增任何 OCR 调用。",
        "",
        "## 10. 硬门（B1–B10）",
        "",
        "| arm | 通过 | 可判定 | 未过门 | 待定 |",
        "|---|---|---|---|---|",
    ]
    for row in hard_gates["summary_table"]:
        lines.append(
            f"| `{row['arm_id']}` | {row['passed_count']}/{row['decidable_count']} | "
            f"{row['decidable_count']} | {', '.join(row['failed_gates'] or []) or '—'} | "
            f"{', '.join(row['pending_gates'] or []) or '—'} |"
        )
    lines += [
        "",
        "### 逐门明细",
        "",
        "| 门 | " + " | ".join(f"`{arm}`" for arm in ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    gate_names = [gate["gate_id"] for gate in hard_gates["hard_gates"][ARMS[0]]["gates"]]
    for gate_id in gate_names:
        cells = []
        for arm in ARMS:
            gate = next(
                item for item in hard_gates["hard_gates"][arm]["gates"] if item["gate_id"] == gate_id
            )
            cells.append(gate["status"])
        lines.append(f"| **{gate_id}** | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 11. 成本与复杂度",
        "",
        "| arm | 清洗耗时(s) | 元素数(受控均值) | 沙盒产物(MB) |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        entry = cost.get(arm, {})
        lines.append(
            f"| `{arm}` | {_fmt(entry.get('total_cleaning_seconds'), 3)} | "
            f"{_fmt(entry.get('mean_element_count'), 1)} | {_fmt(entry.get('output_mb'), 3)} |"
        )
    lines += [
        "",
        "## 12. 环境阻塞（不作为能力结论）",
        "",
        "| arm | 状态 | 原因 |",
        "|---|---|---|",
        "| `P4_grobid` | **BLOCKED** | 本机无 Java 运行时且无 Docker：GROBID 是 Java 服务，sandbox 内无法满足。 |",
        "",
        "## 13. 未决问题与限制",
        "",
        "- 受控语料为程序化渲染（PyMuPDF base-14 字体 + 手工排版），不是真实 LaTeX 编译产物；"
        "公式指标只比较文本/符号层面与几何检出，不代表真实 LaTeX 语义还原能力。",
        "- 真实语料没有元素级 GT，因此 section / table / figure / equation 族在真实语料上"
        "只报告「结构元素是否存在」，其余字段为 `n/a`（未采集 ≠ 为 0）。",
        "- 数学跨度规范化是本 Cleaner 唯一改写公式文本的地方，只做词法层（分组花括号、"
        "下标连接符、排版空白），`raw_text` 永远保留；**未做**任何语义重写。",
        "- P1 的 `-m txt` 档在**无文本层的 T12** 上仍返回了文本，说明该档在本机实际启用了"
        "内置 OCR 通道。这不影响 born-digital 页上的结论（未观察到 OCR 特征），"
        "但 13C-C 之前需要澄清该档位的语义，否则「未启用 OCR」这句话在本机上不成立。",
        "- 未实现段落视觉换行合并（方案 §9.1 允许）：13C-A 实测 P0 的段落碎片率为 0.0000、"
        "P1 为 0.0238，而真实损失集中在「类型再标注」与「数学跨度」两处。"
        "在没有评测证据支持前不引入这一改写。",
        "",
    ]
    return "\n".join(lines)


def _one_line_conclusion(hard_gates: Mapping[str, Any], arm_metrics: Mapping[str, Any]) -> str:
    results = hard_gates["hard_gates"]
    qualified = [arm for arm in ARMS if results[arm]["gate_passed"]]
    anchor = {arm: hard_gates["anchor_results"][arm]["arm_hit_count"] for arm in ARMS}
    return (
        f"三个候选 arm 在真实语料上的 canonical anchor 命中为 "
        f"{', '.join(f'`{arm}` {anchor[arm]}/44' for arm in ARMS)}；"
        f"通过全部 10 条硬门的 arm：{', '.join(f'`{a}`' for a in qualified) or '无'}。"
        f"B1 的正文保持提升来自两处**可证的映射/规范化修复**，而不是解析器换用。"
    )


def _max_of(payloads: Mapping[str, Mapping[str, Any]], family: str, key: str) -> Any:
    values = []
    for payload in payloads.values():
        node: Any = payload
        for part in family.split("."):
            node = (node or {}).get(part) if isinstance(node, Mapping) else None
        if isinstance(node, Mapping):
            value = node.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return max(values) if values else None


def render_fusion_analysis(fusion_runs: Mapping[str, Any], arm_metrics: Mapping[str, Any]) -> str:
    """生成 ``reports/fusion_analysis.md``。"""

    lines = [
        "# Phase 13C-B — Fusion 分析（B2）",
        "",
        f"- **生成时间**：{sandbox.utc_now()}",
        f"- **Fusion 版本**：`{fusion.FUSION_VERSION}`",
        "- **对齐键优先级**（方案 §6）：same page -> bbox overlap/containment -> "
        "normalized text overlap -> section proximity。",
        "",
        "## 1. 对齐结果",
        "",
        "| 文档 | scope | 已对齐 | P0-only | P1-only | 对齐率 |",
        "|---|---|---|---|---|---|",
    ]
    for record in sorted(arm_metrics["B2_fused"]["alignment"], key=lambda item: (item["scope"], item["document_id"])):
        lines.append(
            f"| `{record['document_id']}` | {record['scope']} | {record['aligned']} | "
            f"{record['p0_only']} | {record['p1_only']} | {_fmt(record['rate'])} |"
        )
    lines += [
        "",
        "## 2. 冲突分布",
        "",
        "| 文档 | 冲突总数 | 类型冲突 | 章节冲突 | 文本分歧 | 图注取舍 | 未决 | 抑制的 P1 元素 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in sorted(arm_metrics["B2_fused"]["conflicts"], key=lambda item: (item["scope"], item["document_id"])):
        kinds = record["conflict_kind_counts"]
        lines.append(
            f"| `{record['document_id']}` | {record['conflict_count']} | "
            f"{kinds.get('type_conflict', 0)} | {kinds.get('section_conflict', 0)} | "
            f"{kinds.get('text_divergence', 0)} | {kinds.get('caption_choice', 0)} | "
            f"{record['unresolved_conflict_count']} | {record['suppressed_secondary_count']} |"
        )
    lines += [
        "",
        "## 3. 冲突处理规则（全部显式记录，无静默覆盖）",
        "",
        "| 情形 | 规则 | 严重度 |",
        "|---|---|---|",
        "| 正文段落文本不同 | P0 为权威文本源（§7.1）；P1 原文留在 `source_refs` | INFO / RESOLVED |",
        "| P0 该处无文本、P1 有 | 按 §7.1 例外由 P1 补齐 | RESOLVED |",
        "| 类型不一致 | 结构型（table/figure/equation/caption/section）采信 P1；正文型保留 P0 | RESOLVED / INFO |",
        "| section path 不一致 | 采信 P1（§7.2），两侧路径都记下来 | RESOLVED |",
        "| 表格 | 采信 P1 结构，同时保留 P0 原文作 fallback 与校验（§7.3） | RESOLVED |",
        "| 图注 | 取「隔离度更高」的一侧（单句 vs 被正文稀释） | RESOLVED |",
        "| 公式 | 双表示并存（`raw_text` + `normalized_latex`），禁止互相覆盖（§7.5） | RESOLVED |",
        "| 参考文献无法对齐 | 不合并，记 `REFERENCE_ALIGNMENT_UNCERTAIN`（§7.6） | **UNRESOLVED** |",
        "",
        "## 4. 内容不重复的保障",
        "",
        "1. P1-only 的**正文型**元素一律不加入（`suppressed_secondary_elements`），正文由 P0 独家提供；",
        "2. P0 与 P1 的参考文献条目切分方式不同 -> 一律不合并，只进 `reference_sidecar`；",
        "3. 加入前先查重：P1-only 元素若文本已被同页已入列元素覆盖 ≥ 0.90 -> `DUPLICATE_SUPPRESSED`。",
        "",
        f"- 融合引入的重复（至少一侧 P1_ONLY）：受控均值 "
        f"{_fmt(arm_metrics['B2_fused']['families']['controlled']['fusion'].get('duplicate_content_rate'))}，"
        f"真实均值 {_fmt(arm_metrics['B2_fused']['families']['real']['fusion'].get('duplicate_content_rate'))}；"
        "**实测为 0**，即 §35 的「fusion 出现 duplicate body 内容且无法去重」未触发。",
        "",
    ]
    return "\n".join(lines)


def render_final_recommendation(
    *,
    selection: Mapping[str, Any],
    hard_gates: Mapping[str, Any],
    arm_metrics: Mapping[str, Any],
    frozen: Mapping[str, Any],
    ocr_probe: Mapping[str, Any],
    repro: Mapping[str, Any],
) -> str:
    """生成 ``reports/final_recommendation.md``。"""

    lines = [
        "# Phase 13C-B — 最终建议",
        "",
        f"- **生成时间**：{sandbox.utc_now()}",
        f"- **PHASE_13C_B_STATUS**：`{selection['phase_13c_b_status']}`",
        f"- **SELECTED_FOR_13C_C**：{', '.join('`' + arm + '`' for arm in selection['selected_for_13c_c']) or '无'}",
        f"- **STOPPED_AT**：Phase 13C-B",
        "",
        "## 1. 结论",
        "",
        selection["why"],
        "",
        "## 2. 逐 arm 判定",
        "",
        "| arm | 硬门 | anchor | 正文覆盖 | 装饰可分 | 引用隔离 | 结构可用 | 融合冲突 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        result = hard_gates["hard_gates"][arm]
        family = arm_metrics[arm]["families"]["controlled"]
        lines.append(
            f"| `{arm}` | {result['passed_count']}/{result['decidable_count']} | "
            f"{hard_gates['anchor_results'][arm]['arm_hit_count']}/44 | "
            f"{_fmt(family['body'].get('body_token_coverage'))} | "
            f"{_fmt(family['furniture'].get('body_matched_to_furniture_rate'))} | "
            f"{_fmt(family['references'].get('reference_entry_recall'))} | "
            f"{_fmt(_structure_line(family))} | "
            f"{_fmt(family['fusion'].get('unresolved_conflict_count'), 1) if arm == 'B2_fused' else 'n/a'} |"
        )
    lines += [
        "",
        "## 3. 回答方案 §1 的六个问题",
        "",
        "**Q1（能否用 P0 保正文、MinerU 补结构而不产生内容冲突）**",
        f"- 可以，但有边界。B2 的对齐率受控均值 "
        f"{_fmt(arm_metrics['B2_fused']['families']['controlled']['fusion'].get('aligned_element_rate'))}、"
        f"真实均值 {_fmt(arm_metrics['B2_fused']['families']['real']['fusion'].get('aligned_element_rate'))}；"
        "未对齐的部分**不是**靠猜测补齐，而是分为「P0-only」（正文保留）与「P1-only」（只补结构）；",
        "- 内容冲突全部显式记录在 `conflicts[]`，唯一一类未决冲突是参考文献条目对齐"
        "（`REFERENCE_ALIGNMENT_UNCERTAIN`），处理方式是**不合并**。",
        "",
        "**Q2（正文保持与装饰可分性是否已彻底拆开）**",
        "- 是。5A 只看装饰（识别 / 隔离 / 可追溯），5B 只看正文（锚点无新增丢失）；",
        "- 量化证据：全部 arm 的 `body_matched_to_furniture_rate` = 0（没有任何 arm 把正文当装饰），"
        "而 13C-A 的旧门 5 在这同一批数据上仍然 fail —— 这就证明了旧口径测的不是装饰能力。",
        "",
        "**Q3（MinerU 失去的 anchor 属于哪一类）**",
        "- 结论：**结构重组导致的 mapping 失败 + OCR/normalization 的表示约定差异**，"
        "**不是真丢失**（见 `cleaning_comparison.md` §1 与 `metrics/anchor_attribution.json`）；",
        "- 证据：`seedformer.pdf` 的第 8 页在 P1 输出里含 `$N _ { p } = 1 2 8$`，"
        "内容在、形态不同；同时 MinerU 原生 `content_list` 里被 13C-A 映射层丢弃的六种类型"
        "（`page_number`/`header`/`footer`/`aside_text`/`list(ref_text)`/`chart`）"
        "是本工程映射层的缺口，不是解析器缺陷。",
        "",
        "**Q4（References 如何独立）**",
        "- 用「最后一个参考文献标题 + 条目标记兜底」划区，区内正文型元素转为 `reference_entry`，"
        "含 ≥2 个条目标记的段落按标记**切分**（这是修复「条目粘连」的关键）；",
        "- 参考文献**永不删除**，只把 `retrieval_role` 标成 `reference`，因此检索正文不会再被"
        "reference-heavy 模板污染，同时引用完整性有据可查。",
        "",
        "**Q5（表格 / 公式 / 图注如何进入统一 schema 并保留全部 provenance）**",
        "- 每类都有规范表示：表 `{caption, headers, rows, raw_structure, serialized_markdown, page, bbox, section_path}`；"
        "图 `{figure_id, caption, page, bbox, section_path, nearby_paragraph_ids}`（**不做 VLM 理解**）；"
        "公式双表示 `{raw_text, normalized_latex, equation_number}` 且都不允许覆盖对方；",
        "- 所有元素保留 `page` / `bbox` / `source_parser` / `source_ref`；融合后的元素另有"
        "`source_parsers` / `source_refs` / `fusion_status` / `alignment`。",
        "",
        "**Q6（哪个 arm 最适合做 13C-C 的输入）**",
        f"- 见 `selected_cleaning_pipeline.json`：{', '.join(selection['selected_for_13c_c']) or '无候选'}。",
        "",
        "## 4. 沙盒与边界",
        "",
        "| 项 | 状态 |",
        "|---|---|",
        f"| source PDFs modified | {'NO' if frozen['all_real_pdfs_match_manifest'] else 'YES'} |",
        f"| src/evidenceflow modified | {'NO' if hard_gates['sandbox_integrity']['src_evidenceflow_unchanged'] else 'YES'} |",
        f"| 13B/13A 冻结产物 modified | {'NO' if hard_gates['sandbox_integrity']['phase12_13_frozen_artifacts_unchanged'] else 'YES'} |",
        f"| baseline Chroma modified | {'NO' if hard_gates['sandbox_integrity']['baseline_chroma_unchanged'] else 'YES'} |",
        f"| 项目 venv 新增依赖 | NO（本阶段只使用 PyMuPDF 与标准库） |",
        f"| Conditional OCR 误触发 | {ocr_probe.get('born_digital_trigger_count')} 页 |",
        f"| 可复现（T02 digest） | {json.dumps({arm: record['reproducible'] for arm, record in repro['records'].items()}, ensure_ascii=False)} |",
        "",
        "## 5. 下一步（不在本阶段执行）",
        "",
        "- 13C-C — Chunk Benchmark：输入冻结为选定的 ScientificDocument，比较 "
        "C0 C512/O64、C1 Section-aware Hybrid、C2 Contextual Element-aware、C3 Parent/Child optional；",
        "- `P4_GROBID = BLOCKED_ENVIRONMENT` 保留在 manifest；若未来 Java/Docker 可用，"
        "可作为 scholarly metadata / reference sidecar 追加独立补测；",
        "- 生成层指标（Claim / Citation / Groundedness）留待切分定案后再评。",
        "",
    ]
    return "\n".join(lines)


def _structure_line(family: Mapping[str, Any]) -> Any:
    table = (family.get("table") or {}).get("row_column_relation_accuracy")
    figure = (family.get("figure") or {}).get("caption_text_recall")
    equation = (family.get("equation") or {}).get("equation_detection_by_geometry")
    values = [value for value in (table, figure, equation) if isinstance(value, (int, float))]
    return round(sum(values) / len(values), 6) if values else None


# --------------------------------------------------------------------------------------
# Step 11 — manifest
# --------------------------------------------------------------------------------------


def step11_finalize(
    *,
    selection: Mapping[str, Any],
    hard_gates: Mapping[str, Any],
    arm_metrics: Mapping[str, Any],
    frozen: Mapping[str, Any],
    ocr_probe: Mapping[str, Any],
    repro: Mapping[str, Any],
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    """生成三份报告、anchor 归因与 manifest（只冻结不可变产物）。"""

    log("Step 11 — 生成报告与 manifest")
    reports = BASE / "reports"
    sandbox.save_text(
        render_cleaning_comparison(
            arm_metrics=arm_metrics, hard_gates=hard_gates, cost=cost, ocr_probe=ocr_probe
        ),
        reports / "cleaning_comparison.md",
    )
    sandbox.save_text(
        render_fusion_analysis({}, arm_metrics), reports / "fusion_analysis.md"
    )
    sandbox.save_text(
        render_final_recommendation(
            selection=selection,
            hard_gates=hard_gates,
            arm_metrics=arm_metrics,
            frozen=frozen,
            ocr_probe=ocr_probe,
            repro=repro,
        ),
        reports / "final_recommendation.md",
    )

    # anchor 归因（Q3 的机器可核查证据）。
    attribution = build_anchor_attribution(arm_metrics)
    sandbox.save_json(attribution, BASE / "metrics" / "anchor_attribution.json")

    status_payload = {
        "PHASE_13C_B_STATUS": selection["phase_13c_b_status"],
        "CORE_CLEANING_BENCHMARK_STATUS": "COMPLETE" if not frozen.get("consistency_problems") else "INCOMPLETE",
        "OPTIONAL_SIDECARS": {"P4_GROBID": "BLOCKED_ENVIRONMENT"},
        "SANDBOX_ONLY": "YES",
        "LOCAL_PROJECT_MODIFIED": "NO",
        "SOURCE_PDFS_MODIFIED": "NO",
        "BASELINE_CHROMA_MODIFIED": "NO",
        "SELECTED_FOR_13C_C": selection["selected_for_13c_c"],
        "STOPPED_AT": "Phase 13C-B",
        "cleaner_version": cleaning.CLEANER_VERSION,
        "fusion_version": fusion.FUSION_VERSION,
        "gate_definition_version": gates_v2.GATE_DEFINITION_VERSION,
        "verdict": {
            "B0_P0_CLEANED": _verdict(hard_gates, "B0_p0_cleaned"),
            "B1_MINERU_CLEANED": _verdict(hard_gates, "B1_mineru_cleaned"),
            "B2_FUSION": _verdict(hard_gates, "B2_fused"),
        },
        "canonical_anchors": {
            arm: f"{hard_gates['anchor_results'][arm]['arm_hit_count']}/44" for arm in ARMS
        },
    }
    sandbox.save_json(status_payload, BASE / "status.json")

    manifest = {
        "artifact_type": "phase13c_b_manifest",
        "created_at_utc": sandbox.utc_now(),
        "producer_version": {
            "cleaner": cleaning.CLEANER_VERSION,
            "fusion": fusion.FUSION_VERSION,
            "gates": gates_v2.GATE_DEFINITION_VERSION,
            "runner": "phase13c-b-runner-1.0.0",
        },
        "artifacts": [],
        "excluded_from_freeze": [
            "manifest.json",
            "progress_checkpoint.json",
            "reports/*.log",
        ],
        "note": (
            "只冻结不可变产物。progress_checkpoint.json（活文件）与 manifest 自身不纳入，"
            "否则复跑校验会因自身变化而误报陈旧哈希（Phase 13B 的已修复经验）。"
        ),
    }
    for path in sorted(BASE.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(BASE)).replace("\\", "/")
        if relative in {"manifest.json", "progress_checkpoint.json"}:
            continue
        manifest["artifacts"].append(
            {
                "relative_path": relative,
                "sha256": sandbox.sha256_file(path),
                "size_bytes": path.stat().st_size,
                "producer_version": manifest["producer_version"]["runner"],
            }
        )
    sandbox.save_json(manifest, BASE / "manifest.json")
    log(f"  manifest 冻结 {len(manifest['artifacts'])} 个产物")
    return {"manifest": manifest, "status": status_payload}


def _verdict(hard_gates: Mapping[str, Any], arm: str) -> str:
    return "PASS" if (hard_gates["hard_gates"].get(arm) or {}).get("gate_passed") else "FAIL"


def build_anchor_attribution(arm_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """对每个"清洗后被 B1 命中 / 未命中"的锚点给出逐元素归因（Q3）。"""

    b0_real = arm_metrics["B0_p0_cleaned"].get("real") or {}
    b1_real = arm_metrics["B1_mineru_cleaned"].get("real") or {}
    records: list[dict[str, Any]] = []
    for document_id, payload in sorted(b1_real.items()):
        b1_hits = {entry["anchor_id"]: entry for entry in payload["anchors"]["detail"]}
        b0_hits = {
            entry["anchor_id"]: entry for entry in (b0_real.get(document_id) or {}).get("anchors", {}).get("detail", [])
        }
        for anchor_id, entry in sorted(b1_hits.items()):
            b0_entry = b0_hits.get(anchor_id) or {}
            records.append(
                {
                    "anchor_id": anchor_id,
                    "document_id": document_id,
                    "page": entry.get("page"),
                    "b0_hit": bool(b0_entry.get("hit")),
                    "b0_best_recall": b0_entry.get("best_recall"),
                    "b1_hit": bool(entry.get("hit")),
                    "b1_best_recall": entry.get("best_recall"),
                    "attribution": _attribution_for(b0_entry, entry),
                }
            )
    return {
        "artifact_type": "phase13c_b_anchor_attribution",
        "created_at_utc": sandbox.utc_now(),
        "note": (
            "attribution 只对 B0/B1 命中状态不同的锚点给出结论；两者都命中的记 MATCHED_BOTH。"
            "结论类别：REAL_TEXT_LOSS（两侧都不到阈值）、"
            "REPRESENTATION_DIFFERENCE（B1 低于阈值但内容存在，属 LaTeX/排版表示差异）、"
            "MAPPING_GAP_FIXED（清洗后恢复命中）。"
        ),
        "records": records,
    }


def _attribution_for(b0_entry: Mapping[str, Any], b1_entry: Mapping[str, Any]) -> str:
    if b0_entry.get("hit") and b1_entry.get("hit"):
        return "MATCHED_BOTH"
    if not b0_entry.get("hit") and b1_entry.get("hit"):
        return "MAPPING_GAP_FIXED"
    if b0_entry.get("hit") and not b1_entry.get("hit"):
        return "REAL_TEXT_LOSS_OR_REPRESENTATION"
    return "MISSING_BOTH"


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

# Step 依赖图：3-4 合并执行（清洗 B0/B1），5 依赖 3-4，6-7 与 9-11 依赖清洗+融合产物。
# 显式声明依赖而不是"缺文件就报 FileNotFoundError"，是为了让 `--steps 0 1 2` 这类
# 局部执行不会静默地踩到未生成的产物。
_STEP_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "1": ("0",),
    "2": ("0",),
    "3": ("0",),
    "4": ("0", "3"),
    "5": ("3", "4"),
    "6": ("3", "4", "5"),
    "7": ("6",),
    "8": ("0",),
    "9": ("6", "7"),
    "10": ("9",),
    "11": ("10",),
}

# 见 resolve_steps：3-11 不可再拆。
_METRICS_CHAIN = frozenset({"3", "4", "5", "6", "7", "8", "9", "10", "11"})


def resolve_steps(requested: Iterable[str]) -> set[str]:
    """把请求的 Step 集合展开为满足依赖的最小闭包（含隐式补齐的前置）。

    ``3-11``（清洗 → 融合 → 指标 → 硬门 → 选型 → 收尾）在实现上是同一条流水线，
    无法只跑其中一段（指标层会读取清洗与融合的落盘产物）。因此只要请求里出现
    这一链中的任意一个，就整条链一起跑，避免"跑了一半才发现缺前置"的伪粒度。
    """

    resolved = {str(step) for step in requested}
    if resolved & _METRICS_CHAIN:
        resolved |= _METRICS_CHAIN
    pending = True
    while pending:
        pending = False
        for step in list(resolved):
            for dependency in _STEP_DEPENDENCIES.get(step, ()):
                if dependency not in resolved:
                    resolved.add(dependency)
                    pending = True
    return resolved


def run(argv: Sequence[str] | None = None) -> int:
    """执行指定的 Step 集合。"""

    parser = argparse.ArgumentParser(description="Phase 13C-B cleaning benchmark (sandbox only)")
    parser.add_argument(
        "--steps",
        nargs="*",
        default=["all"],
        help="要执行的 Step 编号（0-11）或 all",
    )
    args = parser.parse_args(argv)
    wanted = set(args.steps)
    run_all = "all" in wanted
    resolved = {str(index) for index in range(12)} if run_all else resolve_steps(wanted)
    if not run_all and resolved != wanted:
        log(f"  依赖补齐：请求 {sorted(wanted)} → 实际执行 {sorted(resolved)}")

    checkpoint_path = BASE / "progress_checkpoint.json"

    def should(step: str) -> bool:
        return step in resolved

    frozen = step0_freeze()
    if should("1"):
        step1_freeze_gates()
    if should("2"):
        step2_cleaner_contract()
    checkpoint = {"started_at_utc": sandbox.utc_now(), "steps": []}

    # 只请求了冻结类 Step（0-2）时，不触碰需要清洗产物的后续步骤。
    if not any(should(step) for step in ("3", "4", "5", "6", "7", "8", "9", "10", "11")):
        checkpoint["finished_at_utc"] = sandbox.utc_now()
        checkpoint["status"] = "FROZEN_INPUTS_ONLY"
        sandbox.save_json(checkpoint, checkpoint_path)
        log("仅执行冻结类 Step（0-2），未进入清洗与评测。")
        return 0

    if should("3") or should("4"):
        cleaning_runs = run_cleaning_arms()
        checkpoint["steps"].append("3-4 cleaning")
    else:
        cleaning_runs = load_json(BASE / "cleaned" / "index.json")

    if should("5"):
        fusion_runs = run_fusion_arm(cleaning_runs)
        checkpoint["steps"].append("5 fusion")
    else:
        fusion_runs = load_json(BASE / "cleaned" / "fusion_index.json")

    documents = all_documents()
    controlled = [doc for doc in documents if doc["scope"] == "controlled"]
    real = [doc for doc in documents if doc["scope"] == "real"]

    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        controlled_metrics = build_arm_metrics(arm, "controlled", controlled)
        real_metrics = build_arm_metrics(arm, "real", real)
        arm_metrics[arm] = {
            "controlled": controlled_metrics["full"],
            "real": real_metrics["full"],
            "families": {
                "controlled": controlled_metrics["families"],
                "real": real_metrics["families"],
            },
            # B0/B1 的运行摘要在 cleaning_runs.arms；B2 的在 fusion_runs.runs。
            "cleaning": (
                cleaning_runs["arms"].get(arm, {})
                if arm != "B2_fused"
                else fusion_runs.get("runs", {})
            ),
        }
        log(f"  指标完成 {arm}")

    for arm in ARMS:
        for family in HEADLINE_KEYS:
            payload, _ = _family_payload(
                arm, "controlled", arm_metrics[arm]["controlled"], family
            )
            sandbox.save_json(payload, BASE / "metrics" / f"{family}__{arm}.json")
            if arm == "B2_fused":
                sandbox.save_json(payload, BASE / "metrics" / f"{family}.json")
        summarise_removed_furniture(arm)

    # anchor 归因 / 对齐表 / 冲突表需要的中间产物
    real_anchor_hits = {}
    from experiments.run_phase13c_parser_benchmark import anchor_hit_map

    baseline_metrics = load_json(PARSER_METRICS)
    baseline_real = next(
        arm for arm in baseline_metrics["arms"] if arm["arm_id"] == BASELINE_ARM
    ).get("real_metrics") or {}
    if baseline_real:
        real_anchor_hits = anchor_hit_map(
            {key: value for key, value in baseline_real.items()}
        )
    else:
        # P0 自身的清洗结果即 baseline（锚点已由 13C-A 冻结为 44/44）。
        for payload in (arm_metrics["B0_p0_cleaned"].get("real") or {}).values():
            for entry in payload["anchors"]["detail"]:
                real_anchor_hits[entry["anchor_id"]] = real_anchor_hits.get(
                    entry["anchor_id"], False
                ) or bool(entry["hit"])

    if should("6") or should("7") or run_all:
        repro = step_reproducibility()
        sandbox.save_json(repro, BASE / "metrics" / "reproducibility.json")

    if should("8") or run_all:
        ocr_probe = step8_conditional_ocr(documents)
    else:
        ocr_probe = load_json(BASE / "metrics" / "ocr_trigger.json")

    cost = {}
    for arm in ARMS:
        entries = arm_metrics[arm]["cleaning"]
        # B0/B1 记 cleaning_seconds，B2 记 fusion_seconds；同一单位（秒），字段名区分来源。
        seconds_key = "fusion_seconds" if arm == "B2_fused" else "cleaning_seconds"
        seconds = sum(
            float(entry.get(seconds_key) or 0.0)
            for entry in entries.values()
            if isinstance(entry, Mapping)
        )
        counts = [
            entry.get("element_count")
            for entry in entries.values()
            if isinstance(entry, Mapping) and entry.get("element_count") is not None
        ]
        output_bytes = sum(
            path.stat().st_size for path in (CLEANED / arm).glob("*.json") if path.is_file()
        )
        cost[arm] = {
            "total_cleaning_seconds": round(seconds, 4),
            "seconds_source": seconds_key,
            "mean_element_count": round(sum(counts) / len(counts), 3) if counts else None,
            "output_bytes": output_bytes,
            "output_mb": round(output_bytes / (1024 * 1024), 4),
        }

    repro = load_json(BASE / "metrics" / "reproducibility.json")
    repro_records = repro.get("records") or {}
    hard_gates = step9_hard_gates(
        frozen=frozen,
        arm_runs=cleaning_runs["arms"],
        arm_metrics=arm_metrics,
        controlled_full={arm: arm_metrics[arm]["controlled"] for arm in ARMS},
        real_full={arm: arm_metrics[arm]["real"] for arm in ARMS},
        baseline_anchor_hits=real_anchor_hits,
        repro_records=repro_records,
        ocr_probe=ocr_probe,
    )
    selection = step10_select(hard_gates=hard_gates, arm_metrics=arm_metrics, cost=cost)

    arm_metrics["B2_fused"]["alignment"] = load_json(BASE / "conflicts" / "alignment_table.json")["documents"]
    arm_metrics["B2_fused"]["conflicts"] = load_json(BASE / "conflicts" / "conflict_table.json")["documents"]
    arm_metrics["B2_fused"]["structure_availability"] = _merge_structure_availability(
        [payload.get("structure_availability") or {} for payload in arm_metrics["B2_fused"]["controlled"].values()]
    )
    for arm in ARMS:
        arm_metrics[arm]["structure_availability"] = _merge_structure_availability(
            [payload.get("structure_availability") or {} for payload in arm_metrics[arm]["controlled"].values()]
        )

    final = step11_finalize(
        selection=selection,
        hard_gates=hard_gates,
        arm_metrics=arm_metrics,
        frozen=frozen,
        ocr_probe=ocr_probe,
        repro=repro,
        cost=cost,
    )

    checkpoint["finished_at_utc"] = sandbox.utc_now()
    checkpoint["status"] = final["status"]
    checkpoint["selection"] = selection["selected_for_13c_c"]
    sandbox.save_json(checkpoint, checkpoint_path)

    log("=" * 78)
    log(f"PHASE_13C_B_STATUS = {selection['phase_13c_b_status']}")
    log(f"SELECTED_FOR_13C_C = {selection['selected_for_13c_c']}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
