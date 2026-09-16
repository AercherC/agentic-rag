"""Phase 13C-A — Scientific Paper Ingestion Parser Benchmark（Sandbox-Only）执行入口。

对应方案 §33 的 ``Phase 13C-A — Parser Benchmark``。本脚本只做"第一轮"：

    PDF -> parser -> 统一 ScientificDocument（然后 STOP，不做 Chunk）

## 强制性边界（方案 §4）

- 不修改 ``src/evidenceflow/**``、``PROJECT_SPEC.md``、依赖文件、``.env``、现有 venv、
  canonical qrels、Gold、现有 Chroma 目录、Phase 12/13 冻结产物。
- 源 PDF 只读：入场记录 SHA-256，退场复核。
- 所有写入都落在 sandbox 根或 ``evals/evaluation_v3/phase13c_scientific_ingestion_benchmark``。

## 用法

    python -m experiments.run_phase13c_parser_benchmark --scope controlled --arms P0_current P2_docling
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from phase13c import (  # noqa: E402
    adapters,
    align,
    corpus,
    mappers,
    metrics,
    sandbox,
    schema,
    selfcheck,
)

BASE = sandbox.ARTIFACT_ROOT
SANDBOX = sandbox.SANDBOX_ROOT
SOURCE_DOCS = PROJECT_ROOT / "data/source_docs"
MODELS_DIR = SANDBOX / "models"
CORPUS_MANIFEST = PROJECT_ROOT / "evals/research_ingestion/academic_corpus_manifest_v1.json"
QUESTIONS_SOURCE = (
    PROJECT_ROOT
    / "evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json"
)
PHASE13B_REPORT = (
    PROJECT_ROOT / "evals/evaluation_v3/phase13b_controlled_rebuild_v1/report.md"
)

# 方案 §28 指定的历史失败 case（必须在报告中单列）。
FOCUS_FAILURES = ("ARR-007", "ARR-037", "ARR-040")

# 方案 §14 Parser Selection Gate 的判定项（详见 gate_report）。
GATE_MAX_FURNITURE_FALSE_REMOVAL = 0.02


# --------------------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------------------


def log(message: str) -> None:
    """打印带时间戳的进度行。"""

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def executable(env_name: str, name: str) -> Path | None:
    """返回沙盒 venv 中某个控制台脚本的完整路径。"""

    candidate = SANDBOX / "envs" / env_name / "Scripts" / f"{name}.exe"
    return candidate if candidate.exists() else None


# --------------------------------------------------------------------------------------
# Step 0 — 沙盒边界与冻结身份校验
# --------------------------------------------------------------------------------------


def step0_sandbox_guard() -> dict[str, Any]:
    """建立沙盒边界证据：输入哈希、工程文件快照、上一阶段冻结产物身份。"""

    log("Step 0 — sandbox 边界与冻结身份校验")
    sandbox.project_write_guard()

    papers = load_json(CORPUS_MANIFEST)["papers"]
    input_records = {}
    for paper in papers:
        pdf = SOURCE_DOCS / paper["file_name"]
        stat = pdf.stat()
        record = {
            "path": str(pdf),
            "sha256": sandbox.sha256_file(pdf),
            "size_bytes": stat.st_size,
            "manifest_sha256": paper.get("sha256"),
            "manifest_document_id": paper.get("document_id"),
            "has_two_column": paper.get("has_two_column"),
            "has_tables": paper.get("has_tables"),
            "has_figures": paper.get("has_figures"),
            "has_equations": paper.get("has_equations"),
            "page_count": paper.get("page_count"),
        }
        record["matches_manifest"] = record["sha256"] == record["manifest_sha256"]
        input_records[paper["file_name"]] = record

    frozen_projects = sandbox.snapshot_frozen_project_files()
    source_tree_before = sandbox.snapshot_project_source_tree()

    # 13B 冻结产物：只读核对，证明本轮没有改动上一阶段结论所依赖的文件。
    prior_artifacts = {}
    for relative in (
        "phase13b_controlled_rebuild_v1/report.md",
        "phase13b_controlled_rebuild_v1/structured_elements.json",
        "phase13b_controlled_rebuild_v1/index_manifest.json",
        "phase13_kb_chunk_audit_v1/report.md",
        "standard_ir_v1/qrels.json",
    ):
        path = BASE.parent / relative
        prior_artifacts[relative] = {
            "exists": path.exists(),
            "sha256": sandbox.sha256_file(path) if path.exists() else None,
        }

    payload = {
        "artifact_type": "phase13c_input_hashes",
        "created_at_utc": sandbox.utc_now(),
        "sandbox_root": str(SANDBOX),
        "artifact_root": str(BASE),
        "real_pdf_count": len(input_records),
        "real_pdfs": input_records,
        "all_real_pdfs_match_manifest": all(
            record["matches_manifest"] for record in input_records.values()
        ),
        "frozen_project_files": frozen_projects,
        "prior_phase_artifacts": prior_artifacts,
        "environment": sandbox.environment_fingerprint(),
    }
    sandbox.save_json(input_records, SANDBOX / "inputs" / "real_pdf_hashes.json")
    sandbox.save_json(payload, BASE / "input_hashes.json")
    sandbox.save_json(
        {"source_tree": source_tree_before, "taken_at_utc": sandbox.utc_now()},
        SANDBOX / "inputs" / "source_tree_before.json",
    )
    return payload


# --------------------------------------------------------------------------------------
# Step 1 — 受控模板语料
# --------------------------------------------------------------------------------------


def step1_controlled_corpus() -> dict[str, Any]:
    """生成 12 类模板的受控语料与元素级 Ground Truth。"""

    log("Step 1 — 生成受控模板语料（12 模板）")
    built = corpus.build_controlled_corpus(out_dir=SANDBOX / "synthetic")
    corpus.export_ground_truth(built, BASE / "ground_truth" / "controlled_template_ground_truth.json")

    summary = {
        "artifact_type": "phase13c_controlled_corpus_manifest",
        "created_at_utc": built["created_at_utc"],
        "core_content_version": built["core_content_version"],
        "note": built["note"],
        "template_count": built["template_count"],
        "templates": [
            {
                "template_id": document["template_id"],
                "label": document["label"],
                "role": document["role"],
                "columns": document["columns"],
                "page_size": document["page_size"],
                "rasterized": document["rasterized"],
                "page_count": document["page_count"],
                "element_count": document["element_count"],
                "type_counts": schema.type_counts(document["elements"]),
                "pdf_sha256": document["sha256"],
                "pdf_path": document["pdf_path"],
                "content_key_count": built["ground_truth"][document["template_id"]]["content_key_count"],
            }
            for document in built["templates"]
        ],
    }
    sandbox.save_json(summary, BASE / "template_corpus" / "controlled_corpus_manifest.json")
    return {"built": built, "summary": summary}


# --------------------------------------------------------------------------------------
# Step 2 — 语料有效性自证（解析器无关：直接校验 PDF 文本层）
# --------------------------------------------------------------------------------------


def step2_corpus_self_check(built: dict[str, Any], threshold: float = 0.95) -> dict[str, Any]:
    """校验语料本身是否排得对（不经过任何 parser）。

    首版实现拿 P0 当探针，会把 P0 的结构性局限（无表格结构、页码归入 furniture、
    相邻段落被合并）误判成"语料排版有缺陷"，从而阻塞评测。本版改为与解析器无关的
    PDF 文本层自证，三条判据见 :mod:`experiments.phase13c.selfcheck`：

    1. GT 声称的文本确实出现在 GT 声称的那一页；
    2. 该文本确实落在 GT 声称的 bbox 内；
    3. 该页没有 GT 未声明的幽灵文字。
    """

    log("Step 2 — 受控语料有效性自证（PDF 文本层，解析器无关）")
    payload = selfcheck.check_corpus(built, threshold=threshold)
    sandbox.save_json(payload, BASE / "metrics" / "corpus_self_check.json")
    for line in selfcheck.format_summary(payload):
        log(line)
    if payload["all_passed"]:
        graded = payload["graded_template_count"]
        log(f"  通过：{graded}/{graded} 个有文本层的模板全部满足 page/bbox 召回与无幽灵文字")
    else:
        log(f"  警告：以下模板未通过语料自证（需检查排版引擎或 GT 登记）：{payload['failing_templates']}")
    return payload


# --------------------------------------------------------------------------------------
# Step 3 — arm × 文档矩阵执行
# --------------------------------------------------------------------------------------


def _native_dir(arm_id: str, ocr_mode: str, doc_key: str) -> Path:
    """某个 (arm, ocr_mode, 文档) 的原生输出目录。"""

    return SANDBOX / "parsers" / arm_id / ocr_mode / doc_key


def _schema_path(arm_id: str, ocr_mode: str, doc_key: str) -> Path:
    """某个 (arm, ocr_mode, 文档) 的统一 schema 结果路径。"""

    return SANDBOX / "structured" / arm_id / ocr_mode / f"{doc_key}.json"


def _native_has_files(out_dir: Path) -> bool:
    """原生目录里是否有 parser 写出的文件（空目录与不存在目录都返回 ``False``）。"""

    if not out_dir.exists():
        return False
    return any(path.is_file() for path in out_dir.rglob("*"))


def _reusable_native(out_dir: Path, result_path: Path) -> tuple[bool, str]:
    """判断 ``--reuse-native`` 能否复用这批原生产物，返回 ``(可否复用, 原因)``。

    只判"目录里有文件"是不够的：原生目录在**启动解析之前**就被创建，解析过程中也会持续
    写入，因此一次被中断的解析同样会留下"有文件"的目录——复用它会让映射层读到不完整的
    输入，而产物看起来完全正常。

    因此判据加一条：**必须已存在统一 schema 结果**。后者只可能在"某个前序运行成功完成了
    解析 + 映射"之后出现，从而证明这批原生产物当时是完整的（``arm_run.status`` 为
    ``REMAPPED`` 或 ``OK`` 都算，因为它们都经过了完整映射）。
    代价是"原生完整但映射失败过"的情形不再被复用——那是安全方向：宁可重跑。
    """

    if not _native_has_files(out_dir):
        return False, "原生目录没有 parser 产物（该文档从未成功解析过）"
    if not result_path.exists():
        return False, "缺少已完成的映射结果，无法证明原生目录完整（可能被中断）"
    return True, "复用已有原生产物"


def _previous_parse_seconds(result_path: Path) -> float | None:
    """读回上一轮解析实测的耗时（重映射时复用）。

    为什么必须读回来：§30 的成本指标从 ``arm_run.elapsed_seconds`` 取值。
    重映射没有重新解析，自然没有新的耗时；若记为 ``None``，成本表会被清空，
    而这台机器上真实的解析成本数据本来已经采到了。复用＋标注来源，
    比清空或在重映射时编造一个数字都更诚实。
    """

    if not result_path.exists():
        return None
    try:
        payload = load_json(result_path)
    except Exception:
        return None
    arm_run = payload.get("arm_run") or {}
    value = arm_run.get("elapsed_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def _page_sizes(pdf_path: Path) -> dict[int, tuple[float, float]]:
    """读取 PDF 每页的 (宽, 高)，供归一化坐标换算。"""

    import pymupdf

    sizes: dict[int, tuple[float, float]] = {}
    with pymupdf.open(pdf_path) as pdf:
        for index in range(pdf.page_count):
            rect = pdf[index].rect
            sizes[index + 1] = (float(rect.width), float(rect.height))
    return sizes


def _build_command(arm: adapters.ArmSpec, pdf_path: Path, out_dir: Path, ocr_mode: str) -> list[str]:
    """构造某个 arm 的原生 CLI 命令。

    设计说明：**一律使用各工具的官方命令行**，而不是它们的 Python API。原因是 CLI 是
    各项目最稳定的公开契约（跨小版本不变），而内部 API 在 13C 这种"多工具并列比较"
    的场景里极易因版本差异而失败，从而把环境问题误报成工具能力问题。
    """

    if arm.arm_id == "P2_docling":
        exe = executable("docling", "docling")
        if exe is None:
            raise FileNotFoundError("docling console script not found")
        command = [
            str(exe),
            "convert",
            "--to",
            "json",
            "--output",
            str(out_dir),
            "--artifacts-path",
            str(MODELS_DIR),
            "--device",
            "cpu",
        ]
        command += {"none": ["--no-ocr"], "auto": ["--ocr"], "force": ["--force-ocr"]}[ocr_mode]
        if ocr_mode != "none":
            # 受控语料与真实论文都是英文。Docling 的 RapidOCR 默认识别语言是中文
            # （_RAPIDOCR_DEFAULT_LANGUAGE = "ch"），不显式指定会让 O1/O2 的对比
            # 因为语言错配而失真——那是配置问题，不是 parser 能力差异。
            command += ["--ocr-lang", "english"]
        command.append(str(pdf_path))
        return command

    if arm.arm_id == "P3_marker":
        exe = executable("marker", "marker_single")
        if exe is None:
            raise FileNotFoundError("marker_single console script not found")
        command = [
            str(exe),
            str(pdf_path),
            "--output_dir",
            str(out_dir),
            "--output_format",
            "json",
            # CPU 上使用 fast 模式；balanced 需要 VLM 布局模型 + 全页 OCR，非本机可承受。
            "--mode",
            "fast",
            "--disable_image_extraction",
        ]
        if ocr_mode == "none":
            command.append("--disable_ocr")
        elif ocr_mode == "force":
            command.append("--force_ocr")
        return command

    if arm.arm_id == "P1_mineru":
        exe = executable("mineru", "mineru")
        if exe is None:
            raise FileNotFoundError("mineru console script not found")
        # 必须显式使用 pipeline 后端：默认的 hybrid-engine 需要额外 VLM。
        command = [
            str(exe),
            "-p",
            str(pdf_path),
            "-o",
            str(out_dir),
            "-b",
            "pipeline",
            "-l",
            "en",
        ]
        command += {"none": ["-m", "txt"], "auto": ["-m", "auto"], "force": ["-m", "ocr"]}[ocr_mode]
        return command

    if arm.arm_id == "O2_ocr":
        return [
            str(sandbox_python_path("ocr")),
            "-m",
            "phase13c.runners.ocr_arm",
            "--pdf",
            str(pdf_path),
            "--out",
            str(out_dir / "ocr_result.json"),
            "--ocr-mode",
            "force",
        ]

    raise ValueError(f"No command template for arm {arm.arm_id}")


def sandbox_python_path(env_name: str) -> Path:
    """沙盒 venv 解释器路径（不存在时抛错，由调用方转成 ARM_FAILED）。"""

    path = adapters.sandbox_python(env_name)
    if path is None:
        raise FileNotFoundError(f"sandbox venv '{env_name}' is missing")
    return path


def _load_native_document(
    arm: adapters.ArmSpec, pdf_path: Path, out_dir: Path, *, document_id: str, layout_class: str | None
) -> dict[str, Any]:
    """按 arm 读取原生输出并映射为统一 schema。"""

    sha = sandbox.sha256_file(pdf_path)
    stem = pdf_path.stem

    if arm.arm_id == "P2_docling":
        candidate = out_dir / f"{stem}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"docling output not found: {candidate}")
        return mappers.docling_json_to_schema(
            load_json(candidate),
            document_id=document_id,
            source_pdf=pdf_path.name,
            source_sha256=sha,
            layout_class=layout_class,
        )

    if arm.arm_id == "P3_marker":
        candidates = sorted(out_dir.glob(f"{stem}/*.json")) + sorted(out_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"marker output not found under {out_dir}")
        return mappers.marker_json_to_schema(
            load_json(candidates[0]),
            page_sizes=_page_sizes(pdf_path),
            document_id=document_id,
            source_pdf=pdf_path.name,
            source_sha256=sha,
            layout_class=layout_class,
        )

    if arm.arm_id == "P1_mineru":
        candidates = sorted(out_dir.rglob("*_content_list.json"))
        if not candidates:
            raise FileNotFoundError(f"mineru content_list not found under {out_dir}")
        return mappers.mineru_content_list_to_schema(
            load_json(candidates[0]),
            page_sizes=_page_sizes(pdf_path),
            document_id=document_id,
            source_pdf=pdf_path.name,
            source_sha256=sha,
            layout_class=layout_class,
        )

    if arm.arm_id == "O2_ocr":
        payload = load_json(out_dir / "ocr_result.json")
        return _ocr_payload_to_schema(
            payload, pdf_path=pdf_path, document_id=document_id, sha=sha, layout_class=layout_class
        )

    raise ValueError(f"No native loader for arm {arm.arm_id}")


def _ocr_payload_to_schema(
    payload: dict[str, Any],
    *,
    pdf_path: Path,
    document_id: str,
    sha: str,
    layout_class: str | None,
) -> dict[str, Any]:
    """把 OCR 臂的逐页文本块结果映射为 schema（无类型判定，全部为 paragraph）。"""

    elements: list[dict[str, Any]] = []
    for block in payload.get("blocks", []):
        elements.append(
            schema.make_element(
                document_id=document_id,
                element_id=schema.element_id_for("ocr", document_id, block.get("page"), len(elements)),
                type="paragraph",
                raw_text=str(block.get("text") or ""),
                source_parser="O2_ocr",
                page=block.get("page"),
                bbox=block.get("bbox"),
                reading_order=len(elements),
                section_path=[],
                source_ref=f"rapidocr:{block.get('page')}:{len(elements)}",
            )
        )
    return schema.make_document(
        document_id=document_id,
        source_pdf=pdf_path.name,
        source_sha256=sha,
        parser="O2_ocr",
        page_count=int(payload.get("page_count") or 0),
        elements=elements,
        layout_class=layout_class,
        extra={
            "bbox_source": "pdf_points_top_left(rapidocr)",
            "native_format": "rapidocr page blocks",
            "ocr_pages": payload.get("ocr_pages"),
        },
    )


def run_single_arm_document(
    arm: adapters.ArmSpec,
    *,
    pdf_path: Path,
    doc_key: str,
    ocr_mode: str,
    document_id: str,
    layout_class: str | None,
    timeout: int,
    force: bool,
    reuse_native: bool = False,
) -> dict[str, Any]:
    """执行 (arm, ocr_mode, 文档) 并返回结构化结果记录。

    ``reuse_native``：优先复用**已有的原生产物**，只重做「原生产物 → 统一 schema」这一层
    映射，不调用 parser CLI；原生目录里没有真实产物时才执行解析。存在理由：mapper / 指标的
    修复并不改变 parser 的原生产物，重跑解析纯粹是浪费（受控矩阵一轮近 50 分钟）。

    关键纪律：无论走哪条分支，都会在 ``arm_run.status`` 里如实写明本次是
    ``REMAPPED``（复用）还是 ``OK``（真的执行了解析），并在 ``runs[key]`` 携带
    ``remapped`` 标志——"本轮没有再跑解析"必须能从产物里读出来。
    复用还假设原生产物来自同一工具构建；``tool_version`` 逐 arm 记录在案可供核对。

    注意：可复现性检查（门 6）**必须**走真实重解析，不能用本开关，否则摘要比较退化为
    「拿旧产物和自己比」，会伪造出 PASS。
    """

    result_path = _schema_path(arm.arm_id, ocr_mode, doc_key)
    if result_path.exists() and not force and not reuse_native:
        return {
            "status": "CACHED",
            "document_path": str(result_path),
            "elapsed_seconds": None,
            "peak_ram_mb": None,
            "native_dir": str(_native_dir(arm.arm_id, ocr_mode, doc_key)),
        }

    if arm.kind == "blocked":
        return {
            "status": "BLOCKED",
            "reason": arm.blocked_reason,
            "document_path": None,
            "elapsed_seconds": None,
            "peak_ram_mb": None,
            "native_dir": None,
        }

    availability = adapters.arm_availability(arm)
    if not availability["available"]:
        return {
            "status": "BLOCKED",
            "reason": availability["reason"],
            "document_path": None,
            "elapsed_seconds": None,
            "peak_ram_mb": None,
            "native_dir": None,
        }

    # 复用分支：只对"外部 CLI 产出原生产物"的 arm 有意义。
    # in-process 的 arm（P0_current）没有原生产物可复用，且重跑本身极廉价，走正常路径。
    if reuse_native and arm.kind != "inprocess":
        out_dir = _native_dir(arm.arm_id, ocr_mode, doc_key)
        reusable, reason = _reusable_native(out_dir, result_path)
        if reusable:
            previous_seconds = _previous_parse_seconds(result_path)
            try:
                document = _load_native_document(
                    arm, pdf_path, out_dir, document_id=document_id, layout_class=layout_class
                )
            except Exception as exc:
                return {
                    "status": "MAPPING_FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:],
                    "document_path": None,
                    "elapsed_seconds": None,
                    "peak_ram_mb": None,
                    "native_dir": str(out_dir),
                }
            document.setdefault("extra_arm_meta", {})
            document["arm_run"] = {
                "status": "REMAPPED",
                "remapped": True,
                "elapsed_seconds": previous_seconds,
                "elapsed_seconds_source": "上一轮解析实测（本次仅重做映射，未重新解析）",
                "native_dir": str(out_dir),
            }
            sandbox.save_json(document, result_path)
            return {
                "status": "OK",
                "remapped": True,
                "document_path": str(result_path),
                "elapsed_seconds": previous_seconds,
                "peak_ram_mb": None,
                "native_dir": str(out_dir),
                "element_count": len(document["elements"]),
            }
        # 原生产物不存在（该文档从未解析过）→ 如实走下面的正常解析路径。

    if arm.kind == "inprocess":
        import time as _time

        started = _time.perf_counter()
        document = adapters.parse_with_current(
            pdf_path, document_id_override=document_id, layout_class=layout_class
        )
        elapsed = _time.perf_counter() - started
        sandbox.save_json(document, result_path)
        return {
            "status": "OK",
            "document_path": str(result_path),
            "elapsed_seconds": round(elapsed, 3),
            "peak_ram_mb": None,
            "native_dir": None,
            "element_count": len(document["elements"]),
        }

    out_dir = _native_dir(arm.arm_id, ocr_mode, doc_key)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        command = _build_command(arm, pdf_path, out_dir, ocr_mode)
    except Exception as exc:
        return {
            "status": "ARM_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "document_path": None,
            "elapsed_seconds": None,
            "peak_ram_mb": None,
            "native_dir": str(out_dir),
        }

    log_path = SANDBOX / "logs" / f"{arm.arm_id}_{ocr_mode}_{doc_key}.log"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=adapters.arm_subprocess_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        log_path.write_text(
            f"COMMAND: {' '.join(command)}\nTIMEOUT after {timeout}s\n--- STDOUT ---\n{stdout[-8000:]}\n--- STDERR ---\n{stderr[-8000:]}\n",
            encoding="utf-8",
        )
        return {
            "status": "TIMEOUT",
            "reason": f"exceeded {timeout}s budget",
            "document_path": None,
            "elapsed_seconds": round(elapsed, 3),
            "peak_ram_mb": None,
            "native_dir": str(out_dir),
            "log_path": str(log_path),
        }

    log_path.write_text(
        f"COMMAND: {' '.join(command)}\nexit={returncode}\n--- STDOUT ---\n{stdout[-8000:]}\n--- STDERR ---\n{stderr[-12000:]}\n",
        encoding="utf-8",
    )

    if returncode != 0:
        return {
            "status": "ARM_FAILED",
            "reason": f"exit code {returncode}",
            "stderr_tail": stderr[-1500:],
            "document_path": None,
            "elapsed_seconds": round(elapsed, 3),
            "peak_ram_mb": None,
            "native_dir": str(out_dir),
            "log_path": str(log_path),
        }

    try:
        document = _load_native_document(
            arm, pdf_path, out_dir, document_id=document_id, layout_class=layout_class
        )
    except Exception as exc:
        return {
            "status": "MAPPING_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
            "document_path": None,
            "elapsed_seconds": round(elapsed, 3),
            "peak_ram_mb": None,
            "native_dir": str(out_dir),
            "log_path": str(log_path),
        }

    document.setdefault("extra_arm_meta", {})
    document["arm_run"] = {
        "status": "OK",
        "elapsed_seconds": round(elapsed, 3),
        "native_dir": str(out_dir),
        "log_path": str(log_path),
        "command": command,
    }
    sandbox.save_json(document, result_path)
    return {
        "status": "OK",
        "document_path": str(result_path),
        "elapsed_seconds": round(elapsed, 3),
        "peak_ram_mb": None,
        "native_dir": str(out_dir),
        "log_path": str(log_path),
        "element_count": len(document["elements"]),
    }


# --------------------------------------------------------------------------------------
# Step 4 — 指标计算
# --------------------------------------------------------------------------------------


def build_real_anchors() -> list[dict[str, Any]]:
    """构造 canonical evidence anchor 列表（方案 §26.2，不依赖 chunk id）。"""

    papers = {
        paper["document_id"]: paper["file_name"]
        for paper in load_json(CORPUS_MANIFEST)["papers"]
    }
    questions = load_json(QUESTIONS_SOURCE)["questions"]
    anchors: list[dict[str, Any]] = []
    for question in questions:
        for group in question.get("required_evidence_groups", []):
            for index, alternative in enumerate(group.get("alternatives", [])):
                locator = alternative.get("locator") or {}
                anchors.append(
                    {
                        "anchor_id": f"EVID-{question['question_id']}-{group['group_id']}-A{index + 1}",
                        "question_id": question["question_id"],
                        "group_id": group["group_id"],
                        "document_id": alternative["document_id"],
                        "file_name": papers.get(alternative["document_id"]),
                        "page": locator.get("page_start"),
                        "page_end": locator.get("page_end"),
                        "text": alternative.get("evidence_anchor") or "",
                        "left_context": locator.get("left_context"),
                        "right_context": locator.get("right_context"),
                    }
                )
    return anchors


def evaluate_controlled_document(
    ground_truth: list[dict[str, Any]],
    document: dict[str, Any],
    *,
    is_two_column: bool,
    regression_captions: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    """对一篇受控语料文档计算全部 parser 指标。"""

    parsed = document["elements"]
    alignment = align.align_elements(ground_truth, parsed)
    payload = {
        "gt_count": len(ground_truth),
        "parsed_count": len(parsed),
        "match_rate": alignment["match_rate"],
        "complete_unit_rate": alignment["complete_unit_rate"],
        "text_fidelity": metrics.text_fidelity_metrics(ground_truth, parsed, alignment),
        "reading_order": metrics.reading_order_metrics(ground_truth, parsed, alignment),
        "section": metrics.section_metrics(ground_truth, parsed, alignment),
        "paragraph_integrity": metrics.paragraph_integrity_metrics(ground_truth, parsed, alignment),
        "furniture": metrics.furniture_metrics(ground_truth, parsed, alignment),
        "figure_caption": metrics.figure_caption_metrics(
            ground_truth, parsed, alignment, regression_figures=regression_captions
        ),
        "table": metrics.table_metrics(ground_truth, parsed, alignment),
        "equation": metrics.equation_metrics(ground_truth, parsed, alignment),
        "references": metrics.reference_metrics(ground_truth, parsed, alignment),
        "provenance": metrics.provenance_metrics(parsed),
        "element_type_counts": schema.type_counts(parsed),
        "is_two_column": is_two_column,
    }
    return payload


def evaluate_real_document(
    document: dict[str, Any],
    anchors: Sequence[dict[str, Any]],
    *,
    ground_truth_text: str | None = None,
) -> dict[str, Any]:
    """对一篇真实论文计算 anchor 召回与 provenance 指标。"""

    parsed = document["elements"]
    page_constrained = [
        element
        for element in parsed
        if element.get("page") is not None
    ]
    payload = {
        "parsed_count": len(parsed),
        "anchors": metrics.anchor_recall_metrics(anchors, parsed),
        "anchors_page_constrained": metrics.anchor_recall_metrics(anchors, page_constrained),
        "provenance": metrics.provenance_metrics(parsed),
        "element_type_counts": schema.type_counts(parsed),
    }
    if ground_truth_text:
        payload["ngram"] = metrics.ngram_coverage_metrics(
            ground_truth_text, " ".join(e.get("normalized_text", "") for e in parsed)
        )
    return payload


# --------------------------------------------------------------------------------------
# Step 5 — Parser Selection Gate（方案 §14）
# --------------------------------------------------------------------------------------


def anchor_hit_map(real_metrics: dict[str, Any]) -> dict[str, bool]:
    """把真实语料逐篇的 anchor 明细压平成 ``anchor_id -> 是否命中``。

    同一 anchor 可能出现在多篇文档的明细里（alternatives 共享 group），键相同即视为
    同一锚点，任一命中即算命中——这与 §26.2「一个 group 的任一 alternative 命中即满足
    该证据组」的语义一致。
    """

    hits: dict[str, bool] = {}
    for per_doc in real_metrics.values():
        for entry in (per_doc.get("anchors") or {}).get("detail", []) or []:
            anchor_id = entry.get("anchor_id")
            if anchor_id:
                hits[anchor_id] = hits.get(anchor_id, False) or bool(entry.get("hit"))
    return hits


def validation_for_arm(arm_record: dict[str, Any]) -> dict[str, Any]:
    """汇总**单个 arm**所有产出的 schema 校验结果（门 10 的判据来源）。

    必须按 arm 独立汇总：把全部 arm 的校验结果并到一起会让任一 arm 的 schema 问题
    在其他 arm 的门 10 上错误地显示为 FAIL。
    """

    problems: list[Any] = []
    graded = 0
    for entry in (arm_record.get("controlled_metrics") or {}).values():
        validation = entry.get("document_validation") or {}
        graded += 1
        if not validation.get("valid"):
            problems.append(validation.get("problem_count"))
    for entry in (arm_record.get("real_metrics") or {}).values():
        validation = entry.get("document_validation") or {}
        graded += 1
        if not validation.get("valid"):
            problems.append(validation.get("problem_count"))
    return {"valid": not problems, "graded_documents": graded, "problems": problems}


def evaluate_gate(
    *,
    controlled_metrics: dict[str, Any],
    anchor_hits: dict[str, bool],
    baseline_anchor_hits: dict[str, bool] | None,
    document_validation: dict[str, Any],
    source_pdfs_unchanged: bool,
    external_llm_declaration: str,
    failure_records: Sequence[dict[str, Any]] = (),
    reproducibility_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对单个 arm 执行方案 §14 的 10 条硬门判定。

    ``anchor_hits`` / ``baseline_anchor_hits``：真实语料上逐 anchor 的命中情况
    （anchor_id -> bool）。门 1 需要与 baseline 做差集来判断"新增丢失"。
    ``document_validation`` 必须是**该 arm 自己**的 schema 校验结果，不能跨 arm 聚合，
    否则任一 arm 的 schema 问题会污染其余 arm 的门 10 判定。
    """

    checks: list[dict[str, Any]] = []

    def add(gate: str, passed: bool | None, evidence: Any, note: str = "") -> None:
        checks.append(
            {"gate": gate, "passed": passed, "evidence": evidence, "note": note}
        )

    # 门 1：当前 10 篇 canonical evidence anchor **无新增丢失**（方案 §14 第 1 条）。
    #
    # 为什么不能写成绝对阈值（如 mean_recall >= 0.995）：canonical anchor 里天然存在一批
    # baseline（P0_current）本身就命中不了的锚点——这正是 Phase 11/12 已记录的既有检索失败
    # （见 §28 单列的 ARR-007 / ARR-037 / ARR-040）。那些属于「既有损失」，不是本次 parser
    # 替换引入的「新增损失」。若用绝对阈值判定，基线自身的既有失败会把所有候选 arm 一并判死，
    # 门 1 就退化成「锚点集是否完美」而不是「换 parser 是否丢了东西」。
    #
    # 正确判据：逐 anchor 与 baseline 做差集——「基线命中 且 本 arm 未命中」= 新增丢失。
    if not baseline_anchor_hits:
        # 没有 baseline 命中集合（未跑 real scope，或 real scope 未产出 anchor 明细）时，
        # 无法判定"是否新增丢失"，按 PENDING 处理而不是伪造 PASS。
        add(
            "canonical_anchor_no_new_loss",
            None,
            {
                "status": "baseline_anchor_hits_unavailable",
                "arm_hit_count": sum(1 for hit in anchor_hits.values() if hit),
            },
            "需要 --scope real/all 且 baseline(P0_current) 产出 anchor 明细后才能判定",
        )
    else:
        baseline_hits = {aid for aid, hit in baseline_anchor_hits.items() if hit}
        arm_hits = {aid for aid, hit in anchor_hits.items() if hit}
        new_loss = sorted(baseline_hits - arm_hits)
        recovered = sorted(arm_hits - baseline_hits)
        add(
            "canonical_anchor_no_new_loss",
            not new_loss,
            {
                "baseline_hit_count": len(baseline_hits),
                "arm_hit_count": len(arm_hits),
                "new_loss_count": len(new_loss),
                "new_loss_anchor_ids": new_loss[:20],
                "recovered_anchor_ids": recovered[:20],
            },
            "新增丢失 = 基线命中但本 arm 未命中的锚点；recovered 为相对基线的新增命中",
        )

    # 门 2：双栏控制模板无系统性 column interleave
    two_col = {
        key: value["reading_order"]["column_interleave"]
        for key, value in controlled_metrics.items()
        if value.get("is_two_column")
    }
    interleaved = [key for key, value in two_col.items() if value["interleave_detected"]]
    add(
        "no_systematic_column_interleave",
        not interleaved,
        {"two_column_templates": list(two_col), "interleaved_templates": interleaved},
    )

    # 门 3：retained element 具备 page provenance
    page_coverages = [
        value["provenance"]["page_provenance_coverage"]
        for value in controlled_metrics.values()
        if value.get("provenance") and value["provenance"]["page_provenance_coverage"] is not None
    ]
    min_page_coverage = min(page_coverages) if page_coverages else None
    add(
        "page_provenance_on_retained_elements",
        min_page_coverage is not None and min_page_coverage >= 0.999,
        {"min_page_provenance_coverage": min_page_coverage},
    )

    # 门 4：table / figure / caption 不被全部降级成 plain text
    table_detections = [
        value["table"]["table_detected"]
        for value in controlled_metrics.values()
        if value["table"]["gt_table_count"] > 0
    ]
    figure_detections = [
        value["figure_caption"]["figure_detection"]
        for value in controlled_metrics.values()
        if value["figure_caption"]["gt_figure_count"] > 0
    ]
    caption_detections = [
        value["figure_caption"]["caption_detection"]
        for value in controlled_metrics.values()
        if value["figure_caption"]["gt_figure_caption_count"] > 0
    ]
    mean_table = metrics.mean(table_detections)
    mean_figure = metrics.mean(figure_detections)
    mean_caption = metrics.mean(caption_detections)
    add(
        "structured_elements_not_fully_downgraded",
        all(
            value is not None and value > 0
            for value in (mean_table, mean_figure, mean_caption)
        ),
        {
            "table_detected_mean": mean_table,
            "figure_detection_mean": mean_figure,
            "caption_detection_mean": mean_caption,
        },
        "任一为 0 或 None 表示该类结构元素被整体降级为纯文本或完全缺失",
    )

    # 门 5：furniture 能被单独标记或可靠过滤
    furniture_counts = [
        value["furniture"]["parsed_furniture_count"] for value in controlled_metrics.values()
    ]
    false_removals = [
        value["furniture"]["furniture_false_removal"]
        for value in controlled_metrics.values()
        if value["furniture"]["furniture_false_removal"] is not None
    ]
    mean_false_removal = metrics.mean(false_removals)
    add(
        "furniture_isolated_or_filtered",
        (max(furniture_counts) if furniture_counts else 0) > 0
        and mean_false_removal is not None
        and mean_false_removal <= GATE_MAX_FURNITURE_FALSE_REMOVAL,
        {
            "max_parsed_furniture_count": max(furniture_counts) if furniture_counts else 0,
            "mean_furniture_false_removal": mean_false_removal,
        },
        "furniture_false_removal = 被丢弃的正文元素占比",
    )

    # 门 6：输出可复现（同一输入重复解析，规范化摘要必须一致）
    #
    # 三态而非两态：``reproducible is None`` 表示"复现性未采集"（例如复现模板未参与本次运行，
    # 或 arm 未产出），此时必须记为 PENDING。若直接 ``bool(None)`` 会把它误判成 FAIL——
    # 那是"没能验证"，不是"验证不通过"。
    if reproducibility_record is None:
        add(
            "reproducible_output",
            None,
            {"checked_by": "step_reproducibility", "status": "not_collected"},
            "见 reproducibility_check.json",
        )
    else:
        verdict = reproducibility_record.get("reproducible")
        add(
            "reproducible_output",
            None if verdict is None else bool(verdict),
            {
                "template_id": reproducibility_record.get("template_id"),
                "ocr_mode": reproducibility_record.get("ocr_mode"),
                "checked": reproducibility_record.get("checked"),
                "reason": reproducibility_record.get("reason"),
                "digest_sha256_prefix": (
                    str(reproducibility_record["digest_first"])[:16]
                    if reproducibility_record.get("digest_first")
                    else None
                ),
            },
            "摘要只覆盖 parser/页数/元素类型·文本·坐标·阅读顺序·章节路径，不含路径与耗时",
        )

    # 门 7：不依赖未记录的外部 LLM
    add(
        "no_undocumented_external_llm",
        True,
        {"declaration": external_llm_declaration},
    )

    # 门 8：失败可定位到具体 PDF / page
    # 判据：每条失败记录都必须带 document / page / kind 三个字段。整篇失败时 page 允许为
    # None（此时"定位到 PDF"已成立），但字段本身必须存在——缺字段意味着失败无法归因。
    required_keys = {"document", "page", "kind"}
    missing_keys_records = [
        record for record in failure_records if not required_keys <= set(record)
    ]
    add(
        "failures_locatable_to_pdf_page",
        not missing_keys_records,
        {
            "failure_record_count": len(failure_records),
            "records_missing_attribution_keys": len(missing_keys_records),
        },
        "字段齐全即表示任何失败都能定位到具体 PDF（必要时到页）",
    )

    # 门 9：不修改 source PDF
    add("source_pdfs_untouched", source_pdfs_unchanged, {"verified_by": "input_hash_recheck"})

    # 门 10：可映射到统一 schema
    add(
        "maps_to_unified_schema",
        bool(document_validation.get("valid")),
        {
            "validation_valid": document_validation.get("valid"),
            "problem_count": len(document_validation.get("problems", [])),
        },
    )

    resolved = [check for check in checks if check["passed"] is not None]
    return {
        "checks": checks,
        "passed_count": sum(1 for check in resolved if check["passed"]),
        "decidable_count": len(resolved),
        "all_gates_decided": all(check["passed"] is not None for check in checks),
        "gate_passed": all(check["passed"] for check in resolved),
    }


# --------------------------------------------------------------------------------------
# §35 指标异常检查 —— 「比例越出 [0, 1]」一定是真实缺陷
# --------------------------------------------------------------------------------------

# 语义上必须是比例/比率的指标路径。越界**不可能是**测量噪声，只可能是实现缺陷：
# 实测曾出现 `section_header_f1 = 1.0474` —— 根因是分母用了 `set(element_id)`，
# 把重复 id 的多个实例折叠成一个，分母被缩小。这类缺陷不该靠人眼在报告里发现，
# 应当由 harness 自己记成一条可归因的失败（§35 STOP-ON-ANOMALY）。
#
# 只列**已知语义**的比例，不做"所有数值都该在 [0,1]"的泛化判断：字符错误率、
# 等价/时长/计数类指标本来就可以 > 1，泛化判断会制造大量假告警并淹没真信号。
RATIO_METRIC_PATHS: tuple[tuple[str, ...], ...] = (
    ("text_fidelity", "paragraph_text_recall"),
    ("text_fidelity", "paragraph_text_precision"),
    ("text_fidelity", "normalized_exact_match"),
    ("reading_order", "block_pair_order_accuracy"),
    ("reading_order", "column_interleave", "interleave_rate"),
    ("section", "section_header_precision"),
    ("section", "section_header_recall"),
    ("section", "section_header_f1"),
    ("section", "section_path_accuracy"),
    ("section", "subsection_level_accuracy"),
    ("paragraph_integrity", "paragraph_boundary_precision"),
    ("paragraph_integrity", "paragraph_boundary_recall"),
    ("paragraph_integrity", "fragmented_paragraph_rate"),
    ("paragraph_integrity", "merged_unrelated_paragraph_rate"),
    ("furniture", "furniture_precision"),
    ("furniture", "furniture_recall"),
    ("furniture", "furniture_false_removal"),
    # 正文误删的覆盖率分层也是比例，必须纳入越界检查。
    # 注意：`body_matched_to_furniture` 是**计数**（均值可以 > 1），刻意不在此列。
    ("furniture", "body_loss_by_coverage", "high_coverage_ratio"),
    ("furniture", "body_loss_by_coverage", "low_coverage_ratio"),
    ("furniture", "body_loss_by_coverage", "medium_coverage_ratio"),
    ("figure_caption", "figure_detection"),
    ("figure_caption", "figure_detection_by_geometry"),
    ("figure_caption", "figure_detection_by_text"),
    ("figure_caption", "figure_detection_precision"),
    ("figure_caption", "caption_detection"),
    ("figure_caption", "caption_text_recall"),
    ("figure_caption", "caption_to_figure_link_accuracy"),
    ("figure_caption", "caption_section_assignment"),
    ("table", "table_detected"),
    ("table", "table_detected_by_geometry"),
    ("table", "table_detected_by_text"),
    ("table", "table_detection_precision"),
    ("table", "caption_correct"),
    ("table", "caption_text_recall"),
    ("table", "rows_correct"),
    ("table", "columns_correct"),
    ("table", "cell_text_accuracy"),
    ("table", "row_column_relation_accuracy"),
    ("table", "header_assignment_accuracy"),
    ("table", "spanning_cell_accuracy"),
    ("equation", "equation_detection_recall"),
    ("equation", "equation_detection_by_geometry"),
    ("equation", "equation_detection_by_text"),
    ("equation", "equation_detection_precision"),
    ("equation", "display_vs_inline_accuracy"),
    ("equation", "formula_number_preservation"),
    ("equation", "normalized_latex_similarity"),
    ("equation", "broken_formula_rate"),
    ("references", "reference_section_detection"),
    ("references", "citation_callout_preservation"),
    ("provenance", "page_provenance_coverage"),
    ("provenance", "bbox_provenance_coverage"),
    ("provenance", "source_ref_coverage"),
    ("anchors", "canonical_anchor_recall"),
    ("anchors_page_constrained", "canonical_anchor_recall"),
    ("ngram", "normalized_ngram_coverage"),
)


def _range_violation(metric: str, value: Any) -> dict[str, Any] | None:
    """值越出 ``[0, 1]`` 时返回违规记录，否则返回 ``None``。

    ``kendall_tau`` 与 ``character_error_rate_mean`` 这类"本来就可以为负或 > 1"的指标
    永远不会出现在待检查清单里，因此这里不需要额外排除。
    """

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < 0 or value > 1:
            return {"metric": metric, "value": round(float(value), 6)}
    return None


# arm 级汇总用的键名。受控汇总是**扁平**的（用比例的短名），真实汇总是
# `mean_*` / `min_*` 前缀，两者都与逐文档的嵌套路径不同，因此需要单独一份清单。
RATIO_SUMMARY_METRICS: tuple[str, ...] = tuple(
    sorted({path[-1] for path in RATIO_METRIC_PATHS})
) + (
    "mean_anchor_recall",
    "min_anchor_recall",
    "mean_anchor_recall_page_constrained",
    "mean_page_provenance",
    "mean_bbox_provenance",
    # 装饰分解的均值键名与叶名不同（前缀 `body_loss_`），无法由上面的叶名推导，必须显式列出。
    "body_loss_high_coverage_ratio",
    "body_loss_medium_coverage_ratio",
    "body_loss_low_coverage_ratio",
)


def metric_range_violations(metrics_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """返回语义上必须是比例、却越出 ``[0, 1]`` 的指标（指标名 + 值）。

    返回空列表既表示"没有越界"，也表示"这些指标都是 ``None``"——两种情况都不构成异常。
    """

    violations: list[dict[str, Any]] = []
    for path in RATIO_METRIC_PATHS:
        node: Any = metrics_payload
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        violation = _range_violation(".".join(path), node)
        if violation:
            violations.append(violation)
    return violations


def collect_metric_anomalies(arm_record: dict[str, Any]) -> list[dict[str, Any]]:
    """扫描单个 arm 的受控/真实指标与两级汇总，生成可归因的异常失败记录。"""

    records: list[dict[str, Any]] = []

    def add(scope: str, document: Any, ocr_mode: Any, violation: dict[str, Any]) -> None:
        records.append(
            {
                "scope": scope,
                "document": document,
                "ocr_mode": ocr_mode,
                "kind": "METRIC_OUT_OF_RANGE",
                "page": None,
                "detail": violation,
            }
        )

    for key, entry in (arm_record.get("controlled_metrics") or {}).items():
        parts = key.split("::")
        ocr_mode = parts[0] if len(parts) > 1 else None
        document = parts[-1]
        for violation in metric_range_violations(entry):
            add("controlled", document, ocr_mode, violation)
    for document, entry in (arm_record.get("real_metrics") or {}).items():
        for violation in metric_range_violations(entry):
            add("real", document, None, violation)

    # arm 级均值也是比例：均值会把单文档的越界抹平，所以必须单独查，否则查不到。
    for summary_key in ("controlled_summary", "real_summary"):
        summary = arm_record.get(summary_key) or {}
        for name in RATIO_SUMMARY_METRICS:
            violation = _range_violation(name, summary.get(name))
            if violation:
                add(summary_key, None, None, violation)
    return records


# --------------------------------------------------------------------------------------
# Step 5b — 可复现性（门 6）
# --------------------------------------------------------------------------------------

# 可复现性校验使用的模板：页数最少且同时含双栏、表格与行间公式，能以最低成本覆盖
# 最容易出现非确定性（布局模型 / OCR / 表格结构推断）的路径。
REPRO_TEMPLATE_ID = "T02_ieee_two_column"


def canonical_schema_digest(document: dict[str, Any]) -> str:
    """对结构化输出做规范化摘要，用于"同一输入是否得到相同解析结果"的判定。

    摘要是刻意"窄"的：只覆盖 parser 名、页数，以及每个元素的类型 / 文本 / 坐标 /
    阅读顺序 / 章节路径。**不含** source_pdf 绝对路径、文件哈希、原生输出目录、耗时、
    峰值内存等环境相关字段——它们因运行环境而异，不代表解析结果不稳定。
    """

    projection = {
        "parser": document.get("parser"),
        "page_count": document.get("page_count"),
        "elements": [
            {
                "element_id": element.get("element_id"),
                "type": element.get("type"),
                "normalized_text": element.get("normalized_text"),
                "page": element.get("page"),
                "bbox": element.get("bbox"),
                "reading_order": element.get("reading_order"),
                "section_path": element.get("section_path"),
            }
            for element in document.get("elements") or []
        ],
    }
    return sandbox.sha256_json(projection)


def step_reproducibility(
    built: dict[str, Any], arms_payload: Sequence[dict[str, Any]], *, timeout: int
) -> dict[str, Any]:
    """门 6：对每个可运行 arm 重复解析同一份文档，比较结构输出的规范化摘要。

    成本控制：只重复**一个**模板，而不是整个矩阵。摘要是窄的（见
    :func:`canonical_schema_digest`），因此"不一致"只可能来自 parser 自身的不确定性
    （随机采样、模型非确定性、并发写入顺序），这正是这条门要抓的东西。
    """

    log("Step 5 — 可复现性（同一输入重复解析，比较规范化摘要）")
    template = next(
        (item for item in built["templates"] if item["template_id"] == REPRO_TEMPLATE_ID),
        built["templates"][0],
    )
    template_id = template["template_id"]
    records: dict[str, Any] = {}

    for arm_record in arms_payload:
        arm_id = arm_record["arm_id"]
        arm = adapters.ARM_INDEX[arm_id]
        if arm_record["status"] == "BLOCKED" or not arm_record.get("runs"):
            records[arm_id] = {"checked": False, "reproducible": None, "reason": "arm 未运行（BLOCKED 或没有产出）"}
            continue
        ocr_mode = (arm_record.get("ocr_modes") or ["none"])[0]
        first = _schema_path(arm_id, ocr_mode, template_id)
        if not first.exists():
            records[arm_id] = {
                "checked": False,
                "template_id": template_id,
                "ocr_mode": ocr_mode,
                "reproducible": None,
                "reason": f"首次运行未产出 {template_id} 的结构化结果",
            }
            continue
        run = run_single_arm_document(
            arm,
            pdf_path=Path(template["pdf_path"]),
            doc_key=f"{template_id}__repro",
            ocr_mode=ocr_mode,
            document_id=f"{template_id}-{arm_id}",
            layout_class=template_id,
            timeout=timeout,
            force=True,
        )
        if run["status"] not in {"OK", "CACHED"}:
            records[arm_id] = {
                "checked": True,
                "template_id": template_id,
                "ocr_mode": ocr_mode,
                "reproducible": False,
                "reason": f"重复运行失败：{run['status']}",
            }
            continue
        digest_first = canonical_schema_digest(load_json(first))
        digest_second = canonical_schema_digest(load_json(Path(run["document_path"])))
        reproducible = digest_first == digest_second
        records[arm_id] = {
            "checked": True,
            "template_id": template_id,
            "ocr_mode": ocr_mode,
            "digest_first": digest_first,
            "digest_second": digest_second,
            "reproducible": reproducible,
            "element_count": run.get("element_count"),
        }
        log(f"  {arm_id}: {'REPRODUCIBLE' if reproducible else 'DIFFERS'}")

    graded = [record for record in records.values() if record.get("reproducible") is not None]
    payload = {
        "artifact_type": "phase13c_reproducibility_check",
        "created_at_utc": sandbox.utc_now(),
        "method": (
            "对每个可运行 arm 在同一份受控模板上重复解析一次，比较结构化输出的规范化摘要"
            "（parser / 页数 / 元素的类型·文本·坐标·阅读顺序·章节路径）。摘要不含路径与耗时，"
            "因此不一致只可能来自 parser 自身的不确定性。"
        ),
        "template_id": template_id,
        "arm_count": len(records),
        "graded_arm_count": len(graded),
        "reproducible_arm_count": sum(1 for record in graded if record["reproducible"]),
        "all_reproducible": bool(graded) and all(record["reproducible"] for record in graded),
        "arms": records,
    }
    sandbox.save_json(payload, BASE / "metrics" / "reproducibility_check.json")
    return payload


# --------------------------------------------------------------------------------------
# Step 6 — 报告
# --------------------------------------------------------------------------------------


def _format_metric(value: Any, digits: int = 4) -> str:
    """格式化指标值：``None`` 明确显示为 ``n/a``。"""

    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def arm_gate_section(arm: dict[str, Any]) -> list[str]:
    """生成报告 §8 中单个 arm 的门判定段落。

    关键约定：**环境阻塞必须与「门未过」区分**。把 `BLOCKED` 写成 `FAIL` 会让读者把
    「本机跑不起来」读成「工具能力不达标」，与 §1 的阻塞声明自相矛盾。
    """

    gate = arm["gate"]
    blocked = arm.get("status") == "BLOCKED"
    lines = [
        (
            f"### `{arm['arm_id']}` — BLOCKED（未评估，非工具能力结论）"
            if blocked
            else f"### `{arm['arm_id']}` — {'PASS' if gate['gate_passed'] else 'FAIL'}"
        ),
        "",
    ]
    if not blocked:
        lines += [f"通过 {gate['passed_count']}/{gate['decidable_count']} 条可判定硬门。", ""]
    lines += ["| 门 | 结果 | 证据 |", "|---|---|---|"]
    for check in gate["checks"]:
        if check["passed"]:
            verdict = "PASS"
        elif check["passed"] is None:
            verdict = "PENDING"
        elif blocked:
            verdict = "BLOCKED"
        else:
            verdict = "FAIL"
        evidence = json.dumps(check["evidence"], ensure_ascii=False)
        lines.append(f"| {check['gate']} | {verdict} | {evidence[:220]} |")
    lines.append("")
    return lines


def summarise_body_loss(arms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把各 arm 的「正文误删」成因分解聚合为报告 §4.1 所需的行与判定依据。

    约定（都对应真实踩过的坑）：

    - 占比与计数必须同处**元素级**空间：占比由同一行的计数现算，不能用指标 JSON 里的
      arm 级逐文档均值——两者在「某文档独占几十个元素」时会给出相差数倍的数，
      并排显示会让读者算出自相矛盾的比例。
    - 未评估的 arm（环境阻塞、无 ``controlled_metrics``）以 ``—`` 占位，
      且不得参与 ``collected`` / ``matched_total``，否则会把「未采集」误判成「已采集且为 0」。
    - 计数单位是受控 run 而非文档：跑多个 OCR 档的 arm 每个模板贡献两次计数，
      故同时回传 ``runs`` 供归一化。
    """

    rows: list[dict[str, Any]] = []
    collected = True
    matched_total = 0.0
    any_low = False
    for arm in arms:
        summary = arm.get("controlled_summary") or {}
        metrics_by_run = arm.get("controlled_metrics") or {}
        if not metrics_by_run:
            rows.append(
                {
                    "arm": arm["arm_id"],
                    "fr": "—",
                    "n": "—",
                    "runs": 0,
                    "hi": "—",
                    "lo": "—",
                    "mis": "—",
                    "focus": "—",
                }
            )
            continue
        unmatched = high = low = 0
        low_by_template: dict[str, int] = {}
        for key, entry in metrics_by_run.items():
            split = (entry.get("furniture") or {}).get("body_loss_by_coverage") or {}
            # 受控指标键形如 "<ocr_mode>::<template_id>"，取模板 ID 做归因
            template = key.split("::", 1)[-1]
            h, m, lo = (int(split.get(k, 0)) for k in ("high", "medium", "low"))
            unmatched += h + m + lo
            high += h
            low += lo
            if lo:
                low_by_template[template] = low_by_template.get(template, 0) + lo
        any_low = any_low or bool(low)
        top = sorted(low_by_template.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        focus = "、".join(f"{name} ×{num}" for name, num in top) or "—"
        run_count = (arm.get("cost") or {}).get("controlled_runs")
        if run_count is None:
            run_count = len(metrics_by_run)
        matched = summary.get("body_matched_to_furniture")
        if matched is None:
            # 字段缺失（旧版产物）与真实为 0 含义不同，必须显式区分，不能默认成 0
            collected = False
        else:
            matched_total += float(matched)
        rows.append(
            {
                "arm": arm["arm_id"],
                "fr": _format_metric(summary.get("furniture_false_removal")),
                "n": unmatched,
                "runs": run_count,
                "hi": _format_metric(high / unmatched if unmatched else None),
                "lo": _format_metric(low / unmatched if unmatched else None),
                "mis": _format_metric(matched),
                "focus": focus,
            }
        )
    return {
        "rows": rows,
        "collected": collected,
        "matched_total": matched_total,
        "has_low_coverage": any_low,
    }


def _body_loss_reading_note(*, collected: bool, matched_total: float, has_low_coverage: bool) -> str:
    """生成 §4.1 的「读法」结论。

    该结论**必须由数据推出来**：``body_matched_to_furniture`` 合计为 0 与未采集到该字段
    是两件含义完全不同的事，前者支持「门 5 无区分力」，后者什么都不支持。
    把结论写成固定文案会让产物在换语料后继续宣称一个不再成立的事实。

    ``matched_total`` 是该指标在各 arm 上的均值之和，用浮点比较判零。
    """

    if not collected:
        note = (
            "> **注意**：本产物未采集 `body_matched_to_furniture`，"
            "因此**无法**判断「正文被标成装饰」是否为 0，不得据此下结论。"
        )
    elif abs(matched_total) < 1e-12:
        note = (
            "> **读法**：`正文被标成装饰` 一列在本轮语料上**全部为 0** —— 没有任何 arm 把正文误标为装饰。"
            "因此硬门 5 在这套语料上对这一维度**没有区分力**；它失败的原因来自前三类，"
            "与「装饰能否可靠过滤」无关。修订该门前应先决定：是要度量「装饰元素能否被单独标记」，"
            "还是要度量「正文内容是否完整」——这是两个不同的量，不应由同一个数承担。"
        )
    else:
        note = (
            f"> **读法**：`正文被标成装饰` 一列合计 **{matched_total:.4f}** > 0，"
            "说明确实存在「文本对得上、但被标成 header/footer」的正文元素。这类元素在 "
            "`furniture_false_removal` 里**不被计为丢失**，因此该指标会低估真实的装饰误标。"
        )
    if has_low_coverage:
        note += (
            " `低覆盖集中处` 一列标出了每 arm 低覆盖（真缺失）数量最多的前三个模板："
            "若多个 arm 集中同一模板，应连同该模板的语料特性（例如是否为扫描页）一起归因，"
            "而不是当成 parser 的通用缺陷。"
        )
    return note


def write_parser_comparison_report(payload: dict[str, Any]) -> Path:
    """生成 ``reports/parser_comparison.md``（方案 §33 要求的产物之一）。"""

    arms = payload["arms"]
    run_mode = payload.get("run_mode") or {}
    lines: list[str] = [
        "# Phase 13C-A — Parser Benchmark 报告（Sandbox-Only）",
        "",
        f"- **生成时间**：{payload['created_at_utc']}",
        f"- **沙盒根**：`{payload['sandbox_root']}`",
        f"- **产物根**：`{payload['artifact_root']}`",
        f"- **状态**：`{payload['status']}`",
        f"- **运行模式**：`{'REUSE_NATIVE' if (payload.get('run_mode') or {}).get('reuse_native') else 'FULL'}`"
        f"（映射复用 {run_mode.get('remapped_run_count', 0)} 次 / 实际解析 {run_mode.get('parsed_run_count', 0)} 次）",
        "",
        "## 0. 一句话结论",
        "",
        payload["headline"],
        "",
        "## 1. arm 可运行性与证据",
        "",
        "| arm | 工具版本 | 可运行 | 说明 |",
        "|---|---|---|---|",
    ]
    for arm in arms:
        lines.append(
            f"| `{arm['arm_id']}` | {arm.get('tool_version') or '-'} | "
            f"{'YES' if arm['availability']['available'] else 'NO'} | "
            f"{arm['availability']['reason']} |"
        )

    # 本机环境上不可实现的 OCR 档位：必须与"工具本身不支持"区分开，否则会被误读为能力结论。
    blocked_ocr = [
        (arm["arm_id"], mode, reason)
        for arm in arms
        for mode, reason in sorted((arm.get("blocked_ocr_modes") or {}).items())
    ]
    if blocked_ocr:
        lines += ["", "**本机不可实现的 OCR 档位**（环境限制，不构成工具能力结论）：", ""]
        for arm_id, mode, reason in blocked_ocr:
            lines.append(f"- `{arm_id}` / `{mode}`：{reason}")

    lines += [
        "",
        "## 2. 受控语料有效性自证",
        "",
        f"- 判定方法：{payload['corpus_self_check']['method']}",
        f"- 通过模板数：{payload['corpus_self_check']['passing_template_count']}"
        f"/{payload['corpus_self_check']['graded_template_count']}"
        f"（阈值 {payload['corpus_self_check']['threshold']}）",
        "",
        "| 模板 | page 召回(最小) | bbox 召回(最小) | bbox 召回(均值) | 幽灵文字比例(最大) | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for check in payload["corpus_self_check"]["checks"]:
        if check["excluded"]:
            lines.append(f"| {check['template_id']} | - | - | - | - | 排除（无文本层） |")
            continue
        lines.append(
            f"| {check['template_id']} | {_format_metric(check['min_page_token_recall'])} | "
            f"{_format_metric(check['min_bbox_token_recall'])} | "
            f"{_format_metric(check['mean_bbox_token_recall'])} | "
            f"{_format_metric(check['max_unclaimed_ratio'])} | "
            f"{'PASS' if check['passes'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"> 自证与解析器无关（直接校验 PDF 文本层），原因：{payload['corpus_self_check'].get('why_not_p0_based', '-')}",
    ]

    lines += [
        "",
        "## 3. 受控语料上的结构指标（各 arm 均值）",
        "",
        "| arm | 段落文本召回 | 段落文本精度 | 精确匹配 | CER | 块对顺序 | Kendall tau | 双栏交叉率 | 标题 F1 | section_path | 段落碎片率 | 段落误合并率 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        summary = arm.get("controlled_summary") or {}
        lines.append(
            "| `{arm}` | {recall} | {prec} | {em} | {cer} | {pair} | {tau} | {inter} | {secf1} | {secpath} | {frag} | {merged} |".format(
                arm=arm["arm_id"],
                recall=_format_metric(summary.get("paragraph_text_recall")),
                prec=_format_metric(summary.get("paragraph_text_precision")),
                em=_format_metric(summary.get("normalized_exact_match")),
                cer=_format_metric(summary.get("character_error_rate_mean")),
                pair=_format_metric(summary.get("block_pair_order_accuracy")),
                tau=_format_metric(summary.get("kendall_tau")),
                inter=_format_metric(summary.get("column_interleave_rate")),
                secf1=_format_metric(summary.get("section_header_f1")),
                secpath=_format_metric(summary.get("section_path_accuracy")),
                frag=_format_metric(summary.get("fragmented_paragraph_rate")),
                merged=_format_metric(summary.get("merged_unrelated_paragraph_rate")),
            )
        )

    lines += [
        "",
        "## 4. 结构元素与装饰（各 arm 均值）",
        "",
        "> **图 / 表检出判据**：受控语料里的图是矢量绘制的（GT 文本为空），表格内容在结构化的 "
        "`table_structure` 里（扁平文本为空或只有片段）。对这类元素做 token 文本匹配**在构造上**"
        "恒为 0——它测不出「有没有检出」，只能测出「有没有一段可比的文字」。因此检出改用以"
        "**同页 bbox IoU ≥ 0.5** 为主判据（实测：真正对应的框 IoU 普遍在 0.80–0.98，不对应的多为 0），"
        "文本对齐作为补充证据（parser 把表格降级成纯文本时仍算检出），两者取并集。"
        "「几何」列的分母只算**有合法 bbox 的 GT 元素**——没有坐标的 GT 本来就无法做几何判定，"
        "把它计入分母等于编造一次失败。",
        "",
        "> **几何精确率**（`表检出精确率` / `图检出精确率`）是召回的必要补充：分母是 parser 输出的、"
        "带合法 bbox 的该类元素数，分子是其中真的对应上一个 GT 元素的数量。缺了它，「把图统统标成表」"
        "这种错误只会在召回上拿满分——实测 Marker 在 `T10_figure_heavy`（GT 有 2 表 5 图）输出了 "
        "6 个 table 块（其中 4 个实际是图），表召回 1.0 而精确率仅 0.33。",
        "",
        "> **公式为什么要拆成三列**（`公式检出` / `公式检出(几何)` / `公式检出(文本)`）："
        "各 parser 对公式的**文本约定**不同，而 GT 用 asciiMath 纯文本。实测 `T09_formula_heavy`"
        "（GT 11 个公式）：Docling 输出同约定文本，文本召回 0.958；**MinerU 输出原始 LaTeX**"
        "（`$$ \\mathrm { L } \\operatorname { e n c } = ...`），与 GT 逐 token 不可比，"
        "文本召回仅 0.182——但它**确实检出了全部 11 个公式**（几何召回 1.000）。"
        "因此文本召回测的是「选了哪种输出约定」，几何召回才是「有没有检出」。"
        "公式的几何判据用**重叠系数**（`inter / min(面积)`，阈值 0.6）而不是 IoU："
        "GT 的公式框是行级框（占满栏宽、含右侧编号），parser 的框紧贴字形，"
        "IoU 会被并集分母压低到 0.24 左右而低于 0.5 阈值。改用重叠系数后实测"
        "对应块最低 0.840、误配块最高 0.412，两簇相距 2 倍。"
        "每次运行的 `<arm>/latex_convention.comparable` 字段会显式标出该 arm 的公式文本"
        "是否与 GT 可比；为 `False` 时不得用文本召回给该 arm 的公式能力下结论。",
        "",
        "| arm | 表检出 | 表检出(几何 IoU) | 表检出(文本) | 表检出精确率 | 表格行列关系 | 表头归属 | 跨列单元格 | 图检出 | 图检出(几何 IoU) | 图检出(文本) | 图检出精确率 | 图注检出 | 图注文本召回 | 图注隔离（ARR-007 类） | 公式检出 | 公式检出(几何) | 公式检出(文本) | 公式检出精确率 | 公式编号保留 | 装饰精确率 | 装饰召回 | 正文误删率 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        summary = arm.get("controlled_summary") or {}
        lines.append(
            "| `{arm}` | {tbl} | {tblg} | {tblt} | {tblp} | {rc} | {ha} | {span} | {fig} | {figg} | {figt} | {figp} | {cap} | {ctr} | {iso} | {eq} | {eqg} | {eqt} | {eqp} | {eqn} | {fp} | {fr} | {fdel} |".format(
                arm=arm["arm_id"],
                tbl=_format_metric(summary.get("table_detected")),
                tblg=_format_metric(summary.get("table_detected_by_geometry")),
                tblt=_format_metric(summary.get("table_detected_by_text")),
                tblp=_format_metric(summary.get("table_detection_precision")),
                rc=_format_metric(summary.get("row_column_relation_accuracy")),
                ha=_format_metric(summary.get("header_assignment_accuracy")),
                span=_format_metric(summary.get("spanning_cell_accuracy")),
                fig=_format_metric(summary.get("figure_detection")),
                figg=_format_metric(summary.get("figure_detection_by_geometry")),
                figt=_format_metric(summary.get("figure_detection_by_text")),
                figp=_format_metric(summary.get("figure_detection_precision")),
                cap=_format_metric(summary.get("caption_detection")),
                ctr=_format_metric(summary.get("caption_text_recall")),
                iso=_format_metric(summary.get("caption_regression_isolated_rate")),
                eq=_format_metric(summary.get("equation_detection_recall")),
                eqg=_format_metric(summary.get("equation_detection_by_geometry")),
                eqt=_format_metric(summary.get("equation_detection_by_text")),
                eqp=_format_metric(summary.get("equation_detection_precision")),
                eqn=_format_metric(summary.get("formula_number_preservation")),
                fp=_format_metric(summary.get("furniture_precision")),
                fr=_format_metric(summary.get("furniture_recall")),
                fdel=_format_metric(summary.get("furniture_false_removal")),
            )
        )

    # 页码缺失但 bbox 可用的几何匹配：既不算失败、也不算已验证，必须显式披露，
    # 否则"几何检出"的可信度被默认成满值。
    unverified_by_arm: dict[str, int] = {}
    for arm in arms:
        total = 0
        for entry in (arm.get("controlled_metrics") or {}).values():
            for section, key in (("figure_caption", "figure_geometry"), ("table", "table_geometry"), ("equation", "equation_geometry")):
                geometry = (entry.get(section) or {}).get(key) or {}
                total += int(geometry.get("page_unverified_match_count") or 0)
        unverified_by_arm[arm["arm_id"]] = total
    if any(unverified_by_arm.values()):
        lines += [
            "",
            "> **页码未验证的几何匹配**（bbox 可用、页码缺失；不计为失败，也不视为已验证）："
            + "，".join(
                f"`{arm_id}` {count}"
                for arm_id, count in unverified_by_arm.items()
                if count
            ),
        ]

    lines += [
        "",
        "### 4.1 「正文误删」的成因分解（决定门 5 该怎么读）",
        "",
        "`furniture_false_removal` 的定义是「GT 正文元素**没有对齐到任何 parser 元素**的比例」。"
        "它**不是**「正文被当成装饰删掉的比例」——只要文本还能对上，正文即使被标成 "
        "header/footer 也**不算**丢失（见下方 `正文被标成装饰(均值)` 列，其底层字段是 "
        "`body_matched_to_furniture`）。"
        "因此这个数值背后可能是三件完全不同的事，必须分开读：",
        "",
        "1. **分块/合并差异**（`高覆盖占比` 列）：token 全部能在文档别处找到，只是没对齐到单个元素；",
        "2. **内容真的缺失**（`低覆盖占比` 列）：文档级 token 覆盖率 < 0.5，没抽出来；",
        "3. 扫描页无文本层（看 `低覆盖集中处` 列落在哪些模板上）。",
        "",
        "> **计数口径（重要）**：占比由**同一行的计数**现算（`高 / 未对齐数`），"
        "因此每一行内部自洽——同一行不会出现「计数算出 0.11、占比却写 0.83」这种矛盾。"
        "两列计数的单位是**受控 run**，不是文档：跑多个 OCR 档的 arm（MinerU、Docling 同时跑 "
        "`none` 与 `auto`）会在每个模板上贡献两次计数，因此计数的**跨 arm 绝对值不可直接比较**，"
        "必须先除以该 arm 的 `受控 run 数`。"
        "注意指标 JSON 里的 arm 级 `body_loss_high_coverage_ratio` / `body_loss_low_coverage_ratio` "
        "是**逐文档均值**，与这里不同：一个文档可以装下几十个低覆盖元素（扫描页模板），"
        "逐文档均值会把它与只有 1 个元素的模板等同看待，而元素级占比会如实反映它的支配地位。"
        "归因「门 5 为何失败」要读元素级（本表），跨 arm 比较严重程度时才需要同时看两者。",
        "",
        "| arm | 正文误删率 | 未对齐数(受控 run 合计) | 受控 run 数 | 高覆盖占比(分块差异) | 低覆盖占比(真缺失) | 正文被标成装饰(均值) | 低覆盖集中处 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # 正文误删的成因分解：逐 arm 聚合受控模板的覆盖率分层，并保留低覆盖落在哪个模板上，
    # 以便把「分块差异」「真实缺失」「扫描页无文本层」三类原因分开归因。
    body_loss = summarise_body_loss(arms)
    for row in body_loss["rows"]:
        lines.append(
            "| `{arm}` | {fr} | {n} | {runs} | {hi} | {lo} | {mis} | {focus} |".format(**row)
        )

    # 「读法」结论必须由上面的数据推出来，不能写死：产物换了语料/换了 arm 组合就失效。
    reading = _body_loss_reading_note(
        collected=body_loss["collected"],
        matched_total=body_loss["matched_total"],
        has_low_coverage=body_loss["has_low_coverage"],
    )
    lines += ["", reading]

    lines += [
        "",
        "## 5. 真实语料 canonical anchor 召回",
        "",
        "| arm | 平均 anchor 召回 | 最低 anchor 召回 | 平均 page provenance |",
        "|---|---|---|---|",
    ]
    for arm in arms:
        real = arm.get("real_summary") or {}
        lines.append(
            "| `{arm}` | {recall} | {worst} | {prov} |".format(
                arm=arm["arm_id"],
                recall=_format_metric(real.get("mean_anchor_recall")),
                worst=_format_metric(real.get("min_anchor_recall")),
                prov=_format_metric(real.get("mean_page_provenance")),
            )
        )

    lines += [
        "",
        "## 6. 历史失败 case 单列（方案 §28）",
        "",
        "| 题号 | 锚点 | 文档 | 页 | anchor 文本 | " + " | ".join(f"`{a['arm_id']}`" for a in arms) + " |",
        "|---|---|---|---|---|" + "---|" * len(arms),
    ]
    for case in payload["focus_failure_cases"]:
        cells = " | ".join(
            ("HIT" if entry["hit"] else "MISS") + f" ({_format_metric(entry['best_recall'], 2)})"
            for entry in case["per_arm"]
        )
        lines.append(
            f"| {case['question_id']} | {case['anchor_id']} | {case['file_name']} | {case['page']} | "
            f"{case['text'][:48]} | {cells} |"
        )

    lines += [
        "",
        "## 7. 成本与工程指标（方案 §30）",
        "",
        "| arm | 受控语料均耗时/页(s) | 真实语料均耗时/页(s) | 峰值内存(MB) | 需要 GPU | 解析页数 |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        cost = arm.get("cost") or {}
        lines.append(
            "| `{arm}` | {c1} | {c2} | {ram} | {gpu} | {pages} |".format(
                arm=arm["arm_id"],
                c1=_format_metric(cost.get("controlled_seconds_per_page"), 3),
                c2=_format_metric(cost.get("real_seconds_per_page"), 3),
                ram=_format_metric(cost.get("peak_ram_mb"), 1),
                gpu=cost.get("gpu_required"),
                pages=cost.get("pages_processed"),
            )
        )

    lines += ["", "## 8. Parser Selection Gate（方案 §14）", ""]
    repro = payload.get("reproducibility") or {}
    if repro:
        lines.append(
            f"> 门 6（可复现性）在模板 `{repro.get('template_id')}` 上重复解析一次后比较规范化摘要："
            f"{repro.get('reproducible_arm_count')}/{repro.get('graded_arm_count')} 个 arm 可复现。"
        )
        lines.append("")
    for arm in arms:
        lines.extend(arm_gate_section(arm))
        lines.append("")

    lines += [
        "## 9. 入选下一轮（方案 §33 要求 STOP 于此）",
        "",
    ]
    for index, entry in enumerate(payload["selected_parsers"], start=1):
        qualifier = "（**有条件**，未过门外）" if entry.get("conditional") else ""
        lines.append(f"{index}. `{entry['arm_id']}`{qualifier} — {entry['why']}")
    if not payload["selected_parsers"]:
        lines.append("（无 arm 通过全部硬门）")

    # 未入选的 arm 与未过的门：选型结论必须可复核，只给"入选了谁"不足以支撑 STOP。
    selected_ids = {entry["arm_id"] for entry in payload["selected_parsers"]}
    excluded = [arm for arm in arms if arm["arm_id"] not in selected_ids]
    if excluded:
        lines += ["", "**未入选 / 未过门的 arm**：", ""]
        for arm in excluded:
            if arm["status"] == "BLOCKED":
                lines.append(f"- `{arm['arm_id']}`：BLOCKED — {arm.get('blocked_reason')}")
                continue
            failed = [
                check["gate"] for check in arm["gate"]["checks"] if check["passed"] is False
            ]
            pending = [
                check["gate"] for check in arm["gate"]["checks"] if check["passed"] is None
            ]
            lines.append(
                f"- `{arm['arm_id']}`：未过门 {failed or '（无）'}"
                + (f"；待定 {pending}" if pending else "")
            )
    lines += [
        "",
        "> **STOP**：本阶段为 Phase 13C-A（PDF → parser → structured representation）。"
        "不进入 13C-B（Cleaning / Chunking），不修改任何生产路径。",
    ]

    lines += ["", "## 10. 未决问题与限制", ""]
    lines.extend(f"- {item}" for item in payload["limitations"])

    report_path = BASE / "reports" / "parser_comparison.md"
    sandbox.save_text("\n".join(lines) + "\n", report_path)
    return report_path


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


def resolve_ocr_modes(arm: adapters.ArmSpec, requested: Sequence[str] | None) -> list[str]:
    """决定该 arm 实际运行哪些 OCR 模式。"""

    if requested:
        return [mode for mode in requested if mode in arm.ocr_modes]
    # 默认跑规范允许的 O0/O1：force（O2）只作为对照，且成本高，默认不跑。
    defaults = [mode for mode in arm.ocr_modes if mode != "force"]
    return defaults or list(arm.ocr_modes)


def main(argv: Sequence[str] | None = None) -> int:
    """执行 Phase 13C-A。"""

    parser = argparse.ArgumentParser(description="Phase 13C-A parser benchmark (sandbox only)")
    parser.add_argument("--scope", default="all", choices=["controlled", "real", "all"])
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--ocr-modes", nargs="*", default=None)
    parser.add_argument("--timeout", type=int, default=1800, help="单个 (arm, 文档) 的秒级预算")
    parser.add_argument("--limit-templates", type=int, default=None)
    parser.add_argument(
        "--template-ids",
        nargs="*",
        default=None,
        help="只运行指定模板 ID（用于 OCR 专项定向探针，例如 T12_scanned_image_only）",
    )
    parser.add_argument("--limit-real", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="忽略缓存重跑（会真正调用 parser）")
    parser.add_argument(
        "--reuse-native",
        action="store_true",
        help=(
            "优先复用已有原生产物、只重做「原生 -> 统一 schema」映射（mapper/指标修复后免重跑解析）；"
            "原生目录没有真实产物时才执行解析。逐文档在 arm_run.status 记录 REMAPPED / OK。"
            "与 --force 互斥。"
        ),
    )
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument(
        "--stop-after",
        default="report",
        choices=["guard", "corpus", "selfcheck", "matrix", "report"],
        help="在指定阶段后停止（供迭代语料/自证时避免重跑昂贵的 parser 矩阵）",
    )
    args = parser.parse_args(argv)

    # --force 要求重新执行 parser，--reuse-native 要求优先复用原生产物：两者语义相反，
    # 同时给出必然导致其中之一被静默忽略。直接拒绝，而不是让调用方以为生效了。
    if args.force and args.reuse_native:
        parser.error("--force 与 --reuse-native 互斥：前者强制重跑解析，后者优先复用已有产物")

    started_all = time.time()
    SANDBOX.mkdir(parents=True, exist_ok=True)
    if args.reuse_native:
        log("模式：--reuse-native（优先复用已有原生产物，只重做映射；无产物时才执行解析）")

    guard = step0_sandbox_guard()
    if args.stop_after == "guard":
        return 0
    corpus_payload = step1_controlled_corpus()
    built = corpus_payload["built"]
    if args.stop_after == "corpus":
        return 0

    self_check = (
        {"all_passed": None, "checks": [], "graded_template_count": 0, "passing_template_count": 0,
         "threshold": None, "method": "skipped"}
        if args.skip_self_check
        else step2_corpus_self_check(built)
    )
    if args.stop_after == "selfcheck":
        return 0

    arm_specs = (
        [adapters.ARM_INDEX[arm_id] for arm_id in args.arms]
        if args.arms
        else list(adapters.ARMS)
    )

    # ---- 真实语料准备 ----
    papers = load_json(CORPUS_MANIFEST)["papers"]
    if args.limit_real:
        papers = papers[: args.limit_real]
    document_id_by_file = {paper["file_name"]: paper["document_id"] for paper in papers}
    anchors = build_real_anchors()
    anchors_by_doc = {}
    for anchor in anchors:
        anchors_by_doc.setdefault(anchor["document_id"], []).append(anchor)

    templates = list(built["templates"])
    if args.limit_templates:
        templates = templates[: args.limit_templates]
    if args.template_ids:
        # OCR 专项（方案 §13）只需要在"有文本层/无文本层"两个极端模板上做定向探针，
        # 例如 O2_ocr force 只跑 T12_scanned_image_only 与一个正常模板作对照，
        # 不需要在 12 个模板上重复执行昂贵的全页 OCR。
        wanted = set(args.template_ids)
        unknown = wanted - {template["template_id"] for template in templates}
        if unknown:
            raise SystemExit(f"--template-ids 含未知模板: {sorted(unknown)}")
        templates = [template for template in templates if template["template_id"] in wanted]

    metrics_dir = sandbox.ensure_dir(BASE / "metrics")
    arms_payload: list[dict[str, Any]] = []
    all_failures: dict[str, Any] = {}

    for arm in arm_specs:
        arm_started = time.time()
        log(f"=== arm {arm.arm_id} ({arm.label}) ===")
        availability = adapters.arm_availability(arm)
        ocr_modes = resolve_ocr_modes(arm, args.ocr_modes)
        # 被请求但环境上无法实现的档位：显式记录原因，而不是静默丢弃。
        # 静默丢弃会让产物看起来像"该 arm 只有这些档位"，把环境限制误读为工具能力边界。
        blocked_requested = adapters.blocked_ocr_modes_for(arm, args.ocr_modes)
        arm_record: dict[str, Any] = {
            "arm_id": arm.arm_id,
            "label": arm.label,
            "role": arm.role,
            "kind": arm.kind,
            "ocr_modes": ocr_modes,
            "blocked_ocr_modes": dict(arm.blocked_ocr_modes),
            "blocked_requested_ocr_modes": blocked_requested,
            "availability": availability,
            "blocked_reason": arm.blocked_reason,
            "tool_version": None,
            "runs": {},
            "controlled_metrics": {},
            "real_metrics": {},
        }
        for mode, reason in blocked_requested.items():
            log(f"  [跳过] {arm.arm_id} 的 {mode} 档不可用：{reason}")

        if not availability["available"]:
            arm_record["status"] = "BLOCKED"
            arm_record["controlled_summary"] = {}
            arm_record["real_summary"] = {}
            arm_record["gate"] = {
                "checks": [
                    {
                        "gate": "availability",
                        "passed": False,
                        "evidence": availability,
                        "note": "arm 在本机不可运行，未进入指标比较",
                    }
                ],
                "passed_count": 0,
                "decidable_count": 1,
                "all_gates_decided": True,
                "gate_passed": False,
            }
            arm_record["cost"] = {}
            arms_payload.append(arm_record)
            log(f"  BLOCKED: {availability['reason']}")
            continue

        arm_record["tool_version"] = detect_tool_version(arm)

        failures: list[dict[str, Any]] = []

        # ---- 受控语料 ----
        if args.scope in {"controlled", "all"}:
            for ocr_mode in ocr_modes:
                for template in templates:
                    template_id = template["template_id"]
                    run = run_single_arm_document(
                        arm,
                        pdf_path=Path(template["pdf_path"]),
                        doc_key=template_id,
                        ocr_mode=ocr_mode,
                        document_id=f"{template_id}-{arm.arm_id}",
                        layout_class=template_id,
                        timeout=args.timeout,
                        force=args.force,
                        reuse_native=args.reuse_native,
                    )
                    key = f"controlled::{ocr_mode}::{template_id}"
                    arm_record["runs"][key] = {
                        "scope": "controlled",
                        "ocr_mode": ocr_mode,
                        "document": template_id,
                        **run,
                    }
                    if run["status"] in {"OK", "CACHED"}:
                        document = load_json(Path(run["document_path"]))
                        validation = schema.validate_document(document)
                        metrics_payload = evaluate_controlled_document(
                            built["ground_truth"][template_id]["elements"],
                            document,
                            is_two_column=template["columns"] == 2,
                            regression_captions=[
                                element
                                for element in built["ground_truth"][template_id]["elements"]
                                if element.get("regression_role")
                            ],
                        )
                        metrics_payload["document_validation"] = {
                            "valid": validation["valid"],
                            "problem_count": len(validation["problems"]),
                        }
                        arm_record["controlled_metrics"][f"{ocr_mode}::{template_id}"] = metrics_payload
                        if not validation["valid"]:
                            failures.append(
                                {
                                    "scope": "controlled",
                                    "document": template_id,
                                    "ocr_mode": ocr_mode,
                                    "kind": "SCHEMA_VALIDATION",
                                    "page": None,
                                    "detail": validation["problems"][:10],
                                }
                            )
                    else:
                        failures.append(
                            {
                                "scope": "controlled",
                                "document": template_id,
                                "ocr_mode": ocr_mode,
                                "kind": run["status"],
                                "page": None,
                                "detail": run.get("reason") or run.get("stderr_tail"),
                            }
                        )
                        log(f"  [{ocr_mode}] {template_id}: {run['status']} — {run.get('reason')}")
                    if run["status"] == "OK":
                        log(
                            f"  [{ocr_mode}] {template_id}: OK "
                            f"({run.get('element_count')} elements, {run['elapsed_seconds']}s)"
                        )

        # ---- 真实语料 ----
        if args.scope in {"real", "all"}:
            ocr_mode = ocr_modes[0]
            for paper in papers:
                file_name = paper["file_name"]
                pdf_path = SOURCE_DOCS / file_name
                run = run_single_arm_document(
                    arm,
                    pdf_path=pdf_path,
                    doc_key=pdf_path.stem,
                    ocr_mode=ocr_mode,
                    document_id=document_id_by_file[file_name],
                    layout_class="real_corpus",
                    timeout=args.timeout,
                    force=args.force,
                    reuse_native=args.reuse_native,
                )
                key = f"real::{ocr_mode}::{file_name}"
                arm_record["runs"][key] = {
                    "scope": "real",
                    "ocr_mode": ocr_mode,
                    "document": file_name,
                    **run,
                }
                if run["status"] in {"OK", "CACHED"}:
                    document = load_json(Path(run["document_path"]))
                    validation = schema.validate_document(document)
                    doc_anchors = anchors_by_doc.get(document_id_by_file[file_name], [])
                    arm_record["real_metrics"][file_name] = evaluate_real_document(
                        document, doc_anchors
                    )
                    arm_record["real_metrics"][file_name]["document_validation"] = {
                        "valid": validation["valid"],
                        "problem_count": len(validation["problems"]),
                    }
                else:
                    failures.append(
                        {
                            "scope": "real",
                            "document": file_name,
                            "ocr_mode": ocr_mode,
                            "kind": run["status"],
                            "page": None,
                            "detail": run.get("reason") or run.get("stderr_tail"),
                        }
                    )
                    log(f"  [real/{ocr_mode}] {file_name}: {run['status']} — {run.get('reason')}")
                if run["status"] == "OK":
                    log(
                        f"  [real/{ocr_mode}] {file_name}: OK "
                        f"({run.get('element_count')} elements, {run['elapsed_seconds']}s)"
                    )

        all_failures[arm.arm_id] = failures
        arm_record["failure_count"] = len(failures)
        arm_record["controlled_summary"] = summarise_controlled(arm_record["controlled_metrics"])
        arm_record["real_summary"] = summarise_real(arm_record["real_metrics"])

        # §35 异常检查：比例越出 [0, 1] 只可能是实现缺陷，必须记成可归因的失败，
        # 而不是留给人眼在报告里发现。arm 级均值也要查（均值同样必须是比例）。
        anomalies = collect_metric_anomalies(arm_record)
        if anomalies:
            failures.extend(anomalies)
            arm_record["failure_count"] = len(failures)
            for record in anomalies[:10]:
                log(
                    f"  [异常] {record['scope']}/{record['document']} "
                    f"{record['detail']['metric']} = {record['detail']['value']}（越出 [0, 1]）"
                )

        arm_record["cost"] = summarise_cost(arm_record, templates, papers)
        arm_record["status"] = (
            "OK" if arm_record["controlled_metrics"] or arm_record["real_metrics"] else "NO_DATA"
        )
        arm_record["elapsed_seconds"] = round(time.time() - arm_started, 1)
        arms_payload.append(arm_record)
        log(f"  arm {arm.arm_id} 完成，用时 {arm_record['elapsed_seconds']}s")

    # ---- Gate ----
    reproducibility = step_reproducibility(built, arms_payload, timeout=args.timeout)
    source_check = sandbox.verify_inputs_unchanged(guard["real_pdfs"])

    # 门 1 的 baseline 必须是 P0_current（当前生产路径）在真实语料上的 anchor 命中集合。
    baseline_record = next(
        (record for record in arms_payload if record["arm_id"] == "P0_current"), None
    )
    baseline_anchor_hits = (
        anchor_hit_map(baseline_record["real_metrics"]) if baseline_record else None
    )

    for arm_record in arms_payload:
        if arm_record["status"] == "BLOCKED":
            continue
        arm_record["gate"] = evaluate_gate(
            controlled_metrics=arm_record["controlled_metrics"],
            anchor_hits=anchor_hit_map(arm_record["real_metrics"]),
            baseline_anchor_hits=baseline_anchor_hits,
            # 门 10 必须按 arm 独立判定，不能跨 arm 聚合。
            document_validation=validation_for_arm(arm_record),
            source_pdfs_unchanged=source_check["unchanged"],
            external_llm_declaration=EXTERNAL_LLM_DECLARATION.get(arm_record["arm_id"], "none"),
            failure_records=all_failures.get(arm_record["arm_id"], []),
            reproducibility_record=reproducibility["arms"].get(arm_record["arm_id"]),
        )

    selected = select_parsers(arms_payload)
    focus_cases = build_focus_failure_cases(anchors, arms_payload)

    source_tree_after = sandbox.snapshot_project_source_tree()
    tree_diff = sandbox.diff_source_tree(
        load_json(SANDBOX / "inputs" / "source_tree_before.json")["source_tree"], source_tree_after
    )

    # 运行模式统计：把"复用映射"与"真的执行了解析"分别计数，写进产物。
    # 没有这个计数，"这一轮到底跑没跑解析"就只能靠推测，无法复核。
    remapped_runs = sum(
        1 for arm in arms_payload for run in arm["runs"].values() if run.get("remapped")
    )
    parsed_runs = sum(
        1
        for arm in arms_payload
        for run in arm["runs"].values()
        if run["status"] == "OK" and not run.get("remapped")
    )

    payload = {
        "created_at_utc": sandbox.utc_now(),
        "artifact_type": "phase13c_a_parser_benchmark",
        "status": "PARTIAL" if any(a["status"] in {"BLOCKED", "NO_DATA"} for a in arms_payload) else "PASS",
        "sandbox_root": str(SANDBOX),
        "artifact_root": str(BASE),
        "scope": args.scope,
        "run_mode": {
            "force": bool(args.force),
            "reuse_native": bool(args.reuse_native),
            "remapped_run_count": remapped_runs,
            "parsed_run_count": parsed_runs,
            "note": (
                "已有原生产物的文档只重做了「原生产物 -> 统一 schema」映射，未重新调用 parser；"
                "其余文档执行了解析。逐文档状态见 runs[*].remapped 与 structured 产物的 "
                "arm_run.status（REMAPPED / OK）；被复用文档的 §30 耗时沿用上一轮解析实测值"
                "（arm_run.elapsed_seconds_source 标注来源）。"
                "门 6 的可复现性检查仍真实重跑了 parser。"
            ),
        },
        "arms": arms_payload,
        "corpus_self_check": self_check,
        "reproducibility": reproducibility,
        "selected_parsers": selected,
        "focus_failure_cases": focus_cases,
        "source_pdf_recheck": source_check,
        "project_source_tree_diff": tree_diff,
        "limitations": LIMITATIONS,
        "headline": build_headline(arms_payload, selected, self_check),
        "total_elapsed_seconds": round(time.time() - started_all, 1),
    }

    sandbox.save_json(
        {
            "artifact_type": "phase13c_parser_metrics",
            "created_at_utc": payload["created_at_utc"],
            "arms": [
                {
                    "arm_id": arm["arm_id"],
                    "status": arm["status"],
                    "tool_version": arm["tool_version"],
                    "controlled_summary": arm["controlled_summary"],
                    "real_summary": arm["real_summary"],
                    "cost": arm["cost"],
                    "controlled_metrics": arm["controlled_metrics"],
                    "real_metrics": arm["real_metrics"],
                }
                for arm in arms_payload
            ],
        },
        metrics_dir / "parser_metrics.json",
    )
    sandbox.save_json(
        {
            "artifact_type": "phase13c_parser_failures",
            "created_at_utc": payload["created_at_utc"],
            "failures": all_failures,
            "note": "每条失败都带 scope / document / ocr_mode / kind / page，便于定位到具体 PDF 与页。",
        },
        metrics_dir / "parser_failures.json",
    )
    sandbox.save_json(
        {
            "artifact_type": "phase13c_selected_parsers",
            "created_at_utc": payload["created_at_utc"],
            "stage": "Phase 13C-A",
            "next_stage": "Phase 13C-B — Cleaning Benchmark（只对入选 arm 运行）",
            "selected": selected,
            "gate_summary": {
                arm["arm_id"]: {
                    "gate_passed": arm["gate"]["gate_passed"],
                    "passed_count": arm["gate"]["passed_count"],
                    "decidable_count": arm["gate"]["decidable_count"],
                    # 必须带 status：只给 gate_passed=false 会让"环境阻塞"与"门未过"在机器可读层面
                    # 无法区分，下游会把未评估的 arm 当作评估后失败。
                    "status": arm["status"],
                }
                for arm in arms_payload
            },
        },
        BASE / "selected_parsers.json",
    )
    sandbox.save_json(payload, metrics_dir / "phase13c_a_execution_result.json")
    report_path = write_parser_comparison_report(payload)

    log(f"报告: {report_path}")
    log(f"状态: {payload['status']}；总用时 {payload['total_elapsed_seconds']}s")
    return 0


# --------------------------------------------------------------------------------------
# 汇总辅助
# --------------------------------------------------------------------------------------

EXTERNAL_LLM_DECLARATION = {
    "P0_current": "无外部模型：纯 PyMuPDF 文本/块抽取 + 本地正则判定。",
    "P1_mineru": "pipeline 后端，本地模型（layout/OCR/公式/表格），不调用任何 LLM API。",
    "P2_docling": "standard pipeline，本地模型（layout heron + TableFormerV2）；未启用 VLM pipeline。",
    "P3_marker": "fast 模式：本地轻量检测器 + 按块 OCR；未使用 LLM 服务（--llm_service 未设置）。",
    "P4_grobid": "未运行（需要 Java 服务）。",
    "O2_ocr": "RapidOCR（本地 ONNX 模型），不调用 LLM。",
}

LIMITATIONS = [
    "受控语料为程序化渲染（PyMuPDF base-14 字体 + 手工排版），不是真实 LaTeX 编译产物；"
    "因此公式指标只比较文本/符号层面的还原，不代表真实 LaTeX 语义还原能力。",
    "受控语料中表格由网格线绘制、单元格为独立文本，真实论文表格常以矢量线 + 跨列合并呈现，"
    "本阶段的 row/column 指标偏向乐观。",
    "扫描件模板（T12）由矢量页栅格化而来（200 dpi），与真实扫描件的噪声/倾斜/装订阴影不同，"
    "因此 OCR 相关指标只体现「有无文本层」这一主要变量。",
    "本机装有 GPU（`nvidia-smi -L` 报告 NVIDIA GeForce RTX 5060 Laptop GPU），但环境中的 torch 是 "
    "CPU-only wheel（2.13.0+cpu），`torch.cuda.is_available()` 为 False，因此所有 arm 均实际运行在 "
    "CPU 上，耗时指标不能外推到 GPU 部署。",
    "各 arm 的成本按「每个文档一个独立进程」计量，因此包含每进程一次的模型加载开销"
    "（MinerU 首次冷启动约 139s，其中大部分是模型加载/初始化），不代表稳定状态下的吞吐。",
    "GROBID 需要 Java 运行时或 Docker 服务，本机两者都不具备，故 P4 未获得任何结构化输出。",
    "Marker 的 OCR 档（auto/force）需要 surya-ocr-2 的 VLM 推理服务：surya 判定本机有 GPU 而选择 "
    "vllm 后端，该后端用 Docker 拉起容器，但本机无 docker；CPU 后备的 llama.cpp 需要 PATH 中的 "
    "llama-server，同样不存在。故 P3 只提供原生文本层档（--disable_ocr）。",
    "Marker 的 balanced 模式需要 VLM 布局模型 + 全页 OCR，本机 CPU 不可行，仅运行 fast 模式。",
    "MinerU 默认后端（hybrid-engine / vlm-engine）需要额外 VLM，本阶段只运行 pipeline 后端。",
    "reading order 的 Kendall tau 与块对准确率只基于「成功匹配」的元素对，因此不惩罚匹配失败"
    "所隐含的顺序错误；匹配失败由 missing / fragment 指标单独承担。",
    "受控语料每个模板仅 3–5 页（一份完整短论文的内容量），页数较少使「页级装饰重复检测」"
    "这类依赖跨页重复的启发式难以充分展开。",
    "图 / 表的检出用「同页 bbox IoU ≥ 0.5」判定。阈值 0.5 依据实测分布选取（对应块 IoU 0.80–0.98、"
    "不对应块多为 0），但 IoU 只比较外接矩形，对「框对而内容错」不敏感——该风险由 "
    "caption_text_recall 与表格单元格指标承担，不在此处重复度量。",
    "**公式的几何检出用「同页重叠系数 ≥ 0.6」而不是 IoU**，因为 GT 的公式框是行级框"
    "（占满栏宽、含右侧编号），parser 的框紧贴字形，两者面积差 2–3 倍，IoU 被并集分母"
    "系统性压低（实测对应块中位仅 0.24，低于 0.5 阈值，会把真实检出判成未检出）。"
    "重叠系数（inter / min(面积)）对框大小约定不敏感：实测对应块最低 0.840、"
    "同页误配块最高 0.412、跨页误配最高 0.253，两簇相距 2 倍，0.6 落在空隙正中。"
    "**两种判据的分数不可跨类型直接比较**（公式 0.6 / 图·表 0.5），"
    "每条命中记录的 criterion 字段标明了所用判据。",
    "公式的文本级指标（normalized_latex_similarity、equation_detection_by_text）"
    "**受输出约定影响，不可用于跨 arm 比较**：GT 为 asciiMath 纯文本，而 MinerU 输出原始 LaTeX。"
    "每次运行的 latex_convention.comparable 字段标明该 arm 的约定是否与 GT 可比；"
    "为 False 时结论须以 equation_detection_by_geometry 为准。"
    "本阶段无法渲染真实 LaTeX，LaTeX 语义还原能力未被度量。",
    "若某个 parser 输出了 bbox 却没有页码，几何匹配照常进行并计入 "
    "`page_unverified_match_count`，而不是被直接判为未检出：页码缺失是 provenance 缺陷，"
    "应由 page_provenance_coverage 承担，两类缺陷不应互相掩盖。",
    "`furniture_false_removal` 的语义是「**GT 正文元素未被任何 parser 元素对齐**的比例」，"
    "包含两种成因：被 parser 当作装饰丢弃，以及被彻底漏掉（抽取失败）。"
    "因此它的值**不能**单独读作「家具误删率」——实测 MinerU 的 `parsed_furniture_count` 为 0"
    "（一个装饰元素都没标），却因 36% 正文元素无法对齐而有 `furniture_false_removal = 0.359`。"
    "门 5 是**两个条件的并列**（能单独标记 且 误删 ≤ 2%），"
    "此时它失败在第一条件上；判定正确，但归因必须读 gate evidence 里的 "
    "`max_parsed_furniture_count` 与 `mean_furniture_false_removal` 两项，不能只看后者。",
    "门 5 的第二个条件（误删 ≤ 2%）在**本语料上对全部已评估 arm 都不可满足**，"
    "但它失败的主因不是「装饰过滤不可靠」：按文档级 token 覆盖率分解后，"
    "P0/P2/P3 的低覆盖元素全部集中在同一处（T12 扫描页无文本层，各 40），"
    "MinerU 另有一处集中来源（T11 参考文献段，每 OCR 档 61），OCR arm 的低覆盖为 0；"
    "而「正文文本对得上、却被标成 header/footer」在全部 arm 上均为 0 次。"
    "即该门在本语料上对被评估对象**缺少区分力**，是给每个候选加同一个失败项。"
    "修订该门前需先决定要度量的是「装饰元素能否被单独标记」还是「正文内容是否完整」。",
    "受控指标的**计数**（`未对齐数` / `低覆盖集中处`）粒度是 run 而非文档："
    "同时跑 `none` 与 `auto` 两个 OCR 档的 arm，会在每个模板上贡献两次计数。"
    "跨 arm 比较计数前必须先除以各 arm 的受控 run 数。"
    "报告 §4.1 的两个占比列由**同一行的计数现算**（元素级空间，行内自洽），"
    "与指标 JSON 里的 arm 级 `body_loss_*_coverage_ratio`（**逐文档均值**）不是同一个量："
    "某文档独占几十个元素时两者可相差数倍，归因要读元素级，跨 arm 比较严重程度时需同时看两者。",
]


def detect_tool_version(arm: adapters.ArmSpec) -> str | None:
    """尽力探测 arm 的工具版本；探测失败返回 ``None`` 而不是抛错。"""

    if arm.arm_id == "P0_current":
        import importlib.metadata

        return f"pymupdf=={importlib.metadata.version('pymupdf')}"
    if arm.env_name is None:
        return None
    python_exe = adapters.sandbox_python(arm.env_name)
    if python_exe is None:
        return None
    package = {"P2_docling": "docling", "P3_marker": "marker-pdf", "P1_mineru": "mineru", "O2_ocr": "rapidocr-onnxruntime"}.get(
        arm.arm_id
    )
    if package is None:
        return None
    try:
        completed = subprocess.run(
            [str(python_exe), "-c", f"import importlib.metadata as m; print(m.version('{package}'))"],
            capture_output=True,
            text=True,
            timeout=60,
            env=adapters.arm_subprocess_env(),
        )
        return f"{package}=={completed.stdout.strip()}" if completed.returncode == 0 else None
    except Exception:
        return None


def summarise_controlled(controlled_metrics: dict[str, Any]) -> dict[str, Any]:
    """把逐模板指标聚合为 arm 级均值（``None`` 不参与平均）。"""

    if not controlled_metrics:
        return {}

    def collect(path: Sequence[str]) -> list[float | None]:
        values = []
        for entry in controlled_metrics.values():
            node: Any = entry
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
                if node is None:
                    break
            values.append(node if isinstance(node, (int, float)) else None)
        return values

    regression_isolated = collect(["figure_caption", "regression_slice", "isolated_count"])
    regression_total = collect(["figure_caption", "regression_slice", "case_count"])
    regression_rates = [
        (isolated / total) if isinstance(isolated, (int, float)) and isinstance(total, (int, float)) and total else None
        for isolated, total in zip(regression_isolated, regression_total)
    ]

    return {
        "template_count": len(controlled_metrics),
        "paragraph_text_recall": metrics.mean(collect(["text_fidelity", "paragraph_text_recall"])),
        "paragraph_text_precision": metrics.mean(collect(["text_fidelity", "paragraph_text_precision"])),
        "normalized_exact_match": metrics.mean(collect(["text_fidelity", "normalized_exact_match"])),
        "character_error_rate_mean": metrics.mean(collect(["text_fidelity", "character_error_rate_mean"])),
        "block_pair_order_accuracy": metrics.mean(collect(["reading_order", "block_pair_order_accuracy"])),
        "kendall_tau": metrics.mean(collect(["reading_order", "kendall_tau"])),
        "column_interleave_rate": metrics.mean(
            collect(["reading_order", "column_interleave", "interleave_rate"])
        ),
        "section_header_f1": metrics.mean(collect(["section", "section_header_f1"])),
        "section_path_accuracy": metrics.mean(collect(["section", "section_path_accuracy"])),
        "subsection_level_accuracy": metrics.mean(collect(["section", "subsection_level_accuracy"])),
        "fragmented_paragraph_rate": metrics.mean(collect(["paragraph_integrity", "fragmented_paragraph_rate"])),
        "merged_unrelated_paragraph_rate": metrics.mean(
            collect(["paragraph_integrity", "merged_unrelated_paragraph_rate"])
        ),
        "furniture_precision": metrics.mean(collect(["furniture", "furniture_precision"])),
        "furniture_recall": metrics.mean(collect(["furniture", "furniture_recall"])),
        "furniture_false_removal": metrics.mean(collect(["furniture", "furniture_false_removal"])),
        # 误删的成因分解：高覆盖率 = 内容都在、只是分块不同；低覆盖率 = 真的缺失。
        # 只报总比例会把"扫描页没有文本层/分块差异"误读成"装饰过滤不可靠"。
        "body_loss_high_coverage_ratio": metrics.mean(
            collect(["furniture", "body_loss_by_coverage", "high_coverage_ratio"])
        ),
        "body_loss_low_coverage_ratio": metrics.mean(
            collect(["furniture", "body_loss_by_coverage", "low_coverage_ratio"])
        ),
        # 中覆盖：内容部分在、部分不在，是"分块差异"与"真实缺失"的混合，单列以免被并进任一侧。
        "body_loss_medium_coverage_ratio": metrics.mean(
            collect(["furniture", "body_loss_by_coverage", "medium_coverage_ratio"])
        ),
        # 「把正文标成装饰」的次数（false_removal 测不到这一类）。恒为 0 说明
        # 本语料上没有任何 arm 犯这个错，即该维度对候选没有区分力。
        "body_matched_to_furniture": metrics.mean(
            collect(["furniture", "body_matched_to_furniture"])
        ),
        "figure_detection": metrics.mean(collect(["figure_caption", "figure_detection"])),
        # 几何（bbox IoU）与文本两条证据链分列：几何判据是主判据，文本判据只在 parser 把
        # 图/表降级成带文字的块时才可能有值。分列后报告才能区分"真的没找到"与
        # "找到了但坐标/页码不可比"（后者是 provenance 缺陷，不是检出缺陷）。
        "figure_detection_by_geometry": metrics.mean(
            collect(["figure_caption", "figure_detection_by_geometry"])
        ),
        "figure_detection_by_text": metrics.mean(
            collect(["figure_caption", "figure_detection_by_text"])
        ),
        # 几何精确率：parser 元素里有多少真的对应一个 GT 元素。没有它，"把图统通标成表"
        # 这类错误只会在召回上拿满分而看不出来（实测 Marker 在 T10 即如此）。
        "figure_detection_precision": metrics.mean(
            collect(["figure_caption", "figure_detection_precision"])
        ),
        "caption_detection": metrics.mean(collect(["figure_caption", "caption_detection"])),
        "caption_text_recall": metrics.mean(collect(["figure_caption", "caption_text_recall"])),
        "caption_to_figure_link_accuracy": metrics.mean(
            collect(["figure_caption", "caption_to_figure_link_accuracy"])
        ),
        "caption_regression_isolated_rate": metrics.mean(regression_rates),
        "table_detected": metrics.mean(collect(["table", "table_detected"])),
        "table_detected_by_geometry": metrics.mean(
            collect(["table", "table_detected_by_geometry"])
        ),
        "table_detected_by_text": metrics.mean(collect(["table", "table_detected_by_text"])),
        "table_detection_precision": metrics.mean(
            collect(["table", "table_detection_precision"])
        ),
        "row_column_relation_accuracy": metrics.mean(collect(["table", "row_column_relation_accuracy"])),
        "header_assignment_accuracy": metrics.mean(collect(["table", "header_assignment_accuracy"])),
        "spanning_cell_accuracy": metrics.mean(collect(["table", "spanning_cell_accuracy"])),
        "table_structure_available": metrics.mean(collect(["table", "structure_available"])),
        "equation_detection_recall": metrics.mean(collect(["equation", "equation_detection_recall"])),
        # 与 figure / table 同构：几何判据是主判据（与文本约定无关），文本判据单列。
        # 公式的约定差异比图/表更严重——GT 用 asciiMath 纯文本，MinerU 输出原始 LaTeX，
        # 逐 token 不可比。不分列就会把"输出约定不同"误读成"公式没检出"。
        "equation_detection_by_geometry": metrics.mean(
            collect(["equation", "equation_detection_by_geometry"])
        ),
        "equation_detection_by_text": metrics.mean(
            collect(["equation", "equation_detection_by_text"])
        ),
        "equation_detection_precision": metrics.mean(
            collect(["equation", "equation_detection_precision"])
        ),
        "formula_number_preservation": metrics.mean(collect(["equation", "formula_number_preservation"])),
        "broken_formula_rate": metrics.mean(collect(["equation", "broken_formula_rate"])),
        "reference_section_detection": metrics.mean(collect(["references", "reference_section_detection"])),
        "citation_callout_preservation": metrics.mean(
            collect(["references", "citation_callout_preservation"])
        ),
        "reference_pollution_into_body_rate": metrics.mean(
            collect(["references", "reference_pollution_into_body_rate"])
        ),
        "page_provenance_coverage": metrics.mean(collect(["provenance", "page_provenance_coverage"])),
        "bbox_provenance_coverage": metrics.mean(collect(["provenance", "bbox_provenance_coverage"])),
    }


def summarise_real(real_metrics: dict[str, Any]) -> dict[str, Any]:
    """聚合真实语料的 anchor 与 provenance 指标。"""

    if not real_metrics:
        return {}
    recalls = [
        entry["anchors"]["canonical_anchor_recall"]
        for entry in real_metrics.values()
        if entry.get("anchors") and entry["anchors"]["canonical_anchor_recall"] is not None
    ]
    page_recalls = [
        entry["anchors_page_constrained"]["canonical_anchor_recall"]
        for entry in real_metrics.values()
        if entry.get("anchors_page_constrained")
    ]
    return {
        "document_count": len(real_metrics),
        "mean_anchor_recall": metrics.mean(recalls),
        "min_anchor_recall": round(min(recalls), 6) if recalls else None,
        "mean_anchor_recall_page_constrained": metrics.mean(page_recalls),
        "mean_page_provenance": metrics.mean(
            [
                entry["provenance"]["page_provenance_coverage"]
                for entry in real_metrics.values()
                if entry.get("provenance")
            ]
        ),
        "mean_bbox_provenance": metrics.mean(
            [
                entry["provenance"]["bbox_provenance_coverage"]
                for entry in real_metrics.values()
                if entry.get("provenance")
            ]
        ),
    }


def summarise_cost(
    arm_record: dict[str, Any],
    templates: Sequence[dict[str, Any]],
    papers: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """汇总 arm 的成本指标。"""

    controlled_seconds: list[float] = []
    controlled_pages = 0
    real_seconds: list[float] = []
    real_pages = 0
    page_by_template = {t["template_id"]: t["page_count"] for t in templates}
    page_by_file = {p["file_name"]: p.get("page_count") or 0 for p in papers}

    for run in arm_record["runs"].values():
        if run["status"] != "OK" or not run.get("elapsed_seconds"):
            continue
        if run["scope"] == "controlled":
            page_count = page_by_template.get(run["document"], 0)
            controlled_seconds.append(run["elapsed_seconds"])
            controlled_pages += page_count
        else:
            page_count = page_by_file.get(run["document"], 0)
            real_seconds.append(run["elapsed_seconds"])
            real_pages += page_count

    return {
        "controlled_runs": len(controlled_seconds),
        "controlled_pages": controlled_pages,
        "controlled_seconds_mean": metrics.mean(controlled_seconds),
        "controlled_seconds_per_page": (
            round(sum(controlled_seconds) / controlled_pages, 4) if controlled_pages else None
        ),
        "real_runs": len(real_seconds),
        "real_pages": real_pages,
        "real_seconds_mean": metrics.mean(real_seconds),
        "real_seconds_per_page": (
            round(sum(real_seconds) / real_pages, 4) if real_pages else None
        ),
        "peak_ram_mb": None,
        "gpu_required": False,
        "device": "cpu",
        "pages_processed": controlled_pages + real_pages,
    }


def select_parsers(arm_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 §14："通过门 + 质量最好" 最多选 2 个进入 13C-B。"""

    passed = [
        arm for arm in arm_records if arm.get("gate", {}).get("gate_passed") and arm["status"] == "OK"
    ]
    if not passed:
        # 没有 arm 通过全部硬门时，仍按"通过门数 + 文本保真度"给出条件性建议，
        # 但必须明确标注这是有条件的，而不是把未过门的 arm 当作合格候选。
        candidates = [arm for arm in arm_records if arm["status"] == "OK"]
        candidates.sort(
            key=lambda arm: (
                -arm["gate"]["passed_count"],
                -(arm["controlled_summary"].get("paragraph_text_recall") or 0),
            )
        )
        return [
            {
                "arm_id": arm["arm_id"],
                "conditional": True,
                "why": (
                    f"通过 {arm['gate']['passed_count']}/{arm['gate']['decidable_count']} 条硬门，"
                    "未过门；仅作为有条件候选列出，需在 13C-B 前先解决未过门项。"
                ),
                "failed_gates": [
                    check["gate"] for check in arm["gate"]["checks"] if check["passed"] is False
                ],
            }
            for arm in candidates[:2]
        ]

    passed.sort(
        key=lambda arm: (
            -(arm["controlled_summary"].get("paragraph_text_recall") or 0),
            -(arm["controlled_summary"].get("section_header_f1") or 0),
            arm["cost"].get("controlled_seconds_per_page") or 1e9,
        )
    )
    return [
        {
            "arm_id": arm["arm_id"],
            "conditional": False,
            "why": (
                f"通过全部 {arm['gate']['decidable_count']} 条硬门；"
                f"段落文本召回 {_format_metric(arm['controlled_summary'].get('paragraph_text_recall'))}，"
                f"标题 F1 {_format_metric(arm['controlled_summary'].get('section_header_f1'))}，"
                f"表格检出 {_format_metric(arm['controlled_summary'].get('table_detected'))}。"
            ),
            "failed_gates": [],
        }
        for arm in passed[:2]
    ]


def build_focus_failure_cases(
    anchors: Sequence[dict[str, Any]], arm_records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """把 ARR-007 / ARR-037 / ARR-040 的锚点在每个 arm 上的命中情况单列。"""

    selected_anchors = [anchor for anchor in anchors if anchor["question_id"] in FOCUS_FAILURES]
    file_to_arm_metrics: dict[str, dict[str, Any]] = {}
    for arm in arm_records:
        for file_name, entry in (arm.get("real_metrics") or {}).items():
            detail = {item["anchor_id"]: item for item in entry["anchors"]["detail"]}
            file_to_arm_metrics.setdefault(file_name, {})[arm["arm_id"]] = detail

    cases: list[dict[str, Any]] = []
    for anchor in selected_anchors:
        file_name = anchor["file_name"]
        per_arm = []
        for arm in arm_records:
            detail = (file_to_arm_metrics.get(file_name) or {}).get(arm["arm_id"])
            if detail is None:
                per_arm.append({"arm_id": arm["arm_id"], "hit": False, "best_recall": 0.0,
                                "note": arm["status"]})
                continue
            record = detail.get(anchor["anchor_id"])
            per_arm.append(
                {
                    "arm_id": arm["arm_id"],
                    "hit": bool(record and record["hit"]),
                    "best_recall": record["best_recall"] if record else 0.0,
                }
            )
        cases.append(
            {
                "question_id": anchor["question_id"],
                "group_id": anchor["group_id"],
                "anchor_id": anchor["anchor_id"],
                "file_name": file_name,
                "page": anchor["page"],
                "text": anchor["text"],
                "per_arm": per_arm,
            }
        )
    return cases


def build_headline(
    arm_records: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    self_check: dict[str, Any],
) -> str:
    """生成报告开头的"一句话结论"。"""

    runnable = [arm for arm in arm_records if arm["status"] == "OK"]
    blocked = [arm for arm in arm_records if arm["status"] == "BLOCKED"]
    if not runnable:
        return "本机无任何 parser arm 可完成受控语料 + 真实语料的完整运行（详见 §1 arm 可运行性）。"
    best = max(runnable, key=lambda arm: arm["controlled_summary"].get("paragraph_text_recall") or 0)
    parts = [
        f"{len(runnable)} 个 arm 完成运行（{len(blocked)} 个被环境阻塞）。",
        "受控语料上段落文本召回最高的是 `{}`（{}），"
        "表格结构可得性、图注隔离与装饰处理是各 arm 的分水岭。".format(
            best["arm_id"], _format_metric(best["controlled_summary"].get("paragraph_text_recall"))
        ),
    ]
    if selected:
        parts.append(
            "进入 13C-B 的候选：" + "、".join(f"`{entry['arm_id']}`" for entry in selected) + "。"
        )
    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
