"""Parser arm 适配器：把各 parser 的输出统一映射到 13C ScientificDocument schema。

## arm 划分（方案 §9）

| arm | 名称 | 运行方式 | 说明 |
|---|---|---|---|
| P0 | Current EvidenceFlow Parser | 进程内 | loader+cleaner+13B 结构化解析器（PyMuPDF blocks） |
| P1 | MinerU | 子进程（独立 venv） | 第一主候选 |
| P2 | Docling | 子进程（独立 venv） | 第二主候选 |
| P3 | Marker | 子进程（独立 venv） | 对照候选 |
| P4 | GROBID | 不可用 | 需要 Java/Docker 服务 |
| O2 | RapidOCR（force OCR 对照） | 子进程（独立 venv） | 只在无文本层/乱码页上提供 OCR 文本 |

**P0 是唯一在项目进程内运行的 arm**，因为它就是当前工程的实现；其余 arm 一律通过子进程
调用各自 venv 的解释器，主 harness 从不 import 这些库。这使"项目现有 venv 未被污染"
成为可机械验证的事实，而不是承诺。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pymupdf

from . import sandbox, schema

PROJECT_ROOT = sandbox.PROJECT_ROOT

# Marker 的 OCR 档（auto/force）需要 surya-ocr-2 的 VLM 推理服务，本机不可用。
# 原因链（全部可机器核查，证据落盘在 logs/P3_marker_auto_*.log）：
#   1. `nvidia-smi -L` 报告存在 GPU（NVIDIA GeForce RTX 5060 Laptop GPU）；
#   2. `surya.inference._has_nvidia_gpu()` 在 torch 看不到 CUDA 时会回退到 nvidia-smi 探测，
#      因此它判定"有 GPU"→ 选择 vllm 后端；
#   3. surya 的 VllmBackend 通过 **Docker** 拉起 vLLM 容器（`vllm.py::_resolve_docker_binary`），
#      本机 `docker` 不在 PATH → `surya.inference.backends.spawn.SpawnError: docker binary not found`；
#   4. CPU 后备后端 llama.cpp 需要 `llama-server` 可执行文件（`llamacpp.py::_resolve_llama_server_binary`），
#      PATH 中同样不存在。
# 结论：这是**环境限制**而不是 arm 质量问题，故显式记录为 BLOCKED 档位，而不是让它反复失败。
_MARKER_VLM_OCR_BLOCKED = (
    "surya-ocr-2 的 OCR 档需要 VLM 推理服务：surya 经 nvidia-smi 回退判定本机有 GPU 而选择 vllm "
    "后端，该后端通过 Docker 拉起 vLLM 容器，但 `docker` 不在 PATH（SpawnError: docker binary "
    "not found）；CPU 后备的 llama.cpp 后端又需要 PATH 中的 `llama-server`，同样不存在。"
)


@dataclass(frozen=True)
class ArmSpec:
    """一个 parser arm 的静态描述。"""

    arm_id: str
    label: str
    kind: str  # inprocess | subprocess | blocked
    env_name: str | None = None
    runner: str | None = None
    ocr_modes: tuple[str, ...] = ("auto",)
    role: str = ""
    blocked_reason: str | None = None
    # 本 arm **不提供**（环境上不可实现）的 OCR 档位：mode -> 可机器核查的原因。
    # 与 ocr_modes 分开记录，是为了让"这一档没跑"与"这一档不存在"在产物里可区分——
    # 静默省略会让下游把环境限制误读成"该工具不支持"。
    blocked_ocr_modes: Mapping[str, str] = field(default_factory=dict)


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        arm_id="P0_current",
        label="Current EvidenceFlow Parser (PyMuPDF blocks + 13B 结构判定)",
        kind="inprocess",
        ocr_modes=("none",),
        role="Baseline：当前工程唯一在生产路径上的抽取方式",
    ),
    ArmSpec(
        arm_id="P1_mineru",
        label="MinerU",
        kind="subprocess",
        env_name="mineru",
        runner="phase13c.runners.mineru_arm",
        # "none" 映射 mineru -m txt（只抽原生文本层，不跑 OCR）。必须与 Docling 的
        # --no-ocr / Marker 的 --disable_ocr 对齐：方案要求三个候选在"原生文本层"这一档
        # 可比；缺了它，--ocr-modes none 会被 resolve_ocr_modes 过滤成空列表，arm 静默零运行。
        ocr_modes=("none", "auto", "force"),
        role="第一主候选（layout / reading order / table HTML / 公式）",
    ),
    ArmSpec(
        arm_id="P2_docling",
        label="Docling",
        kind="subprocess",
        env_name="docling",
        runner="phase13c.runners.docling_arm",
        ocr_modes=("none", "auto", "force"),
        role="第二主候选（DoclingDocument / 层级 / body-furniture / bbox provenance）",
    ),
    ArmSpec(
        arm_id="P3_marker",
        label="Marker",
        kind="subprocess",
        env_name="marker",
        runner="phase13c.runners.marker_arm",
        # "none"（marker 的 --disable_ocr）必须参与比较：它是"原生文本层抽取"这一档的
        # 对照臂，与 Docling 的 --no-ocr 语义对齐；否则各 arm 的默认档位不可比。
        # auto/force 需要 surya-ocr-2 的 VLM 服务，本机不可用（见 _MARKER_VLM_OCR_BLOCKED）。
        ocr_modes=("none",),
        blocked_ocr_modes={
            "auto": _MARKER_VLM_OCR_BLOCKED,
            "force": _MARKER_VLM_OCR_BLOCKED,
        },
        role="对照候选（JSON block type 体系）",
    ),
    ArmSpec(
        arm_id="P4_grobid",
        label="GROBID",
        kind="blocked",
        role="科研论文逻辑结构 sidecar（非唯一 table parser）",
        blocked_reason=(
            "本机无 Java 运行时且无 Docker：GROBID 是 Java 服务（jar/容器），"
            "需要 JVM 与可运行的服务进程；sandbox 内无法满足。"
        ),
    ),
    ArmSpec(
        arm_id="O2_ocr",
        label="RapidOCR（force OCR 对照）",
        kind="subprocess",
        env_name="ocr",
        runner="phase13c.runners.ocr_arm",
        ocr_modes=("force",),
        role="OCR 对照臂：验证「无文本层/乱码页才应启用 OCR」（方案 §12–§13）",
    ),
)

ARM_INDEX = {arm.arm_id: arm for arm in ARMS}


def blocked_ocr_modes_for(arm: ArmSpec, requested: Sequence[str] | None) -> dict[str, str]:
    """返回被请求、但该 arm 在本机**环境上无法实现**的 OCR 档位及其原因。

    只对"用户明确请求过"的档位报告，避免把未请求的档位也写进产物造成噪音；
    ``requested`` 为 None（使用默认档）时返回空字典。
    """

    if not requested:
        return {}
    return {
        mode: arm.blocked_ocr_modes[mode]
        for mode in requested
        if mode in arm.blocked_ocr_modes
    }


# --------------------------------------------------------------------------------------
# 沙盒 venv 探测
# --------------------------------------------------------------------------------------


def sandbox_python(env_name: str) -> Path | None:
    """返回沙盒 venv 的解释器路径；不存在时返回 ``None``。"""

    candidate = sandbox.SANDBOX_ROOT / "envs" / env_name / "Scripts" / "python.exe"
    return candidate if candidate.exists() else None


def arm_availability(arm: ArmSpec) -> dict[str, Any]:
    """判断一个 arm 在当前机器上是否可运行，并给出原因。"""

    if arm.kind == "blocked":
        return {"available": False, "reason": arm.blocked_reason or "declared blocked"}
    if arm.kind == "inprocess":
        return {"available": True, "reason": "runs inside the project process"}
    if not arm.env_name or not arm.runner:
        return {"available": False, "reason": "arm spec incomplete"}
    python_exe = sandbox_python(arm.env_name)
    if python_exe is None:
        return {
            "available": False,
            "reason": f"sandbox venv '{arm.env_name}' is missing (install failed or not attempted)",
        }
    return {"available": True, "reason": f"venv={python_exe}"}


def arm_subprocess_env() -> dict[str, str]:
    """构造子进程环境：HF 走镜像，PYTHONPATH 指向项目（只读复用）。"""

    env = dict(os.environ)
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
    env["HF_HOME"] = str(sandbox.SANDBOX_ROOT / "envs" / "_hf_home")
    # Xet 传输走 cas-server.xethub.hf.co，不经过 HF_ENDPOINT 镜像，在受限网络下会 401。
    # 显式关闭后走标准 CDN（可被镜像代理），这是本环境唯一可行的模型获取路径。
    env["HF_HUB_DISABLE_XET"] = "1"
    # 沙箱内存在本地 HTTP 代理（HTTP(S)_PROXY），它无法连通 hf-mirror.com，
    # 而该域名直连可用。为镜像域设置 NO_PROXY 让 httpx 直连，否则会 handshake timeout。
    env["NO_PROXY"] = "hf-mirror.com,.hf-mirror.com,localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("SSL_CERT_FILE", None)
    existing = env.get("PYTHONPATH", "")
    parts = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "experiments")]
    env["PYTHONPATH"] = os.pathsep.join(parts + ([existing] if existing else []))
    # 让 sandbox 脚本加载的 docling/marker 等不要拉取 telemetry。
    env["DOCLING_DISABLE_TELEMETRY"] = "1"
    env["ANONYMIZED_TELEMETRY"] = "False"
    # Marker/surya 的 fast-layout 检测器默认用 ``hf://datalab-to/surya_layout2``，由
    # ``surya.fast_layout.server`` 内部调用 ``snapshot_download`` 解析。本机 Windows 无法
    # 创建符号链接，HF 缓存只落下目录骨架（blobs 里 134MB 的权重读不到），离线必然失败。
    # 而 ``resolve_model_dir`` 同时接受**普通本地目录**，因此把已实体化的快照目录直接指
    # 过去——这是本环境下唯一稳定的做法（models 目录由沙盒侧预取脚本填充）。
    layout_dir = sandbox.SANDBOX_ROOT / "models" / "surya--layout2"
    if (layout_dir / "config.json").is_file():
        env.setdefault("FAST_LAYOUT_MODEL_CHECKPOINT", str(layout_dir))
        env.setdefault("FAST_ORDER_MODEL_CHECKPOINT", str(layout_dir / "order"))
    return env


def run_subprocess_arm(
    arm: ArmSpec,
    *,
    pdf_path: Path,
    out_path: Path,
    ocr_mode: str = "auto",
    timeout: int = 3600,
) -> dict[str, Any]:
    """运行一个子进程 arm，返回 ``{"ok", "stdout", "stderr", "result_path"}``。"""

    python_exe = sandbox_python(arm.env_name or "")
    if python_exe is None or arm.runner is None:
        return {"ok": False, "error": f"arm {arm.arm_id} unavailable", "stdout": "", "stderr": ""}

    command = [
        str(python_exe),
        "-m",
        arm.runner,
        "--pdf",
        str(pdf_path),
        "--out",
        str(out_path),
        "--ocr-mode",
        ocr_mode,
        "--device",
        "cpu",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=arm_subprocess_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"timeout after {timeout}s",
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    return {
        "ok": completed.returncode == 0 and out_path.exists(),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-6000:],
        "error": None if completed.returncode == 0 else f"exit code {completed.returncode}",
    }


def load_arm_document(path: Path) -> dict[str, Any]:
    """读取子进程 arm 写出的统一 schema 文档。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# P0：当前工程解析器（进程内）
# --------------------------------------------------------------------------------------


def _load_p0_dependencies() -> tuple[Any, Any]:
    """延迟导入 P0 依赖（13A 审计模块 / 13B 重建模块），避免非 P0 路径也付出导入成本。"""

    import run_phase13_kb_chunk_audit as audit_module
    import run_phase13b_controlled_rebuild as rebuild_module

    return audit_module, rebuild_module


def parse_with_current(
    pdf_path: Path,
    *,
    document_id_override: str | None = None,
    layout_class: str | None = None,
) -> dict[str, Any]:
    """P0：用当前工程的 PDF 抽取路径构建 ScientificDocument。

    实现是"生产路径 + 13B 结构判定"的组合：

    1. ``load_document`` + ``clean_document`` —— 生产路径的文本来源（页码边界）；
    2. ``parse_page_blocks`` 的逐 block 规范化文本 —— 与官方文本同坐标系的 block 级表示；
    3. 13A 的 ``detect_headings`` / ``detect_document_furniture`` / ``page_level_furniture_spans``
       与 13B 的 ``build_structured_document`` —— 类型判定、章节归属、装饰标记。

    注意 P0 的**已知结构性限制**（本阶段要测量的对象，不是要修复的对象）：

    - 图像块被显式跳过（``if not block["is_text"]: continue``）-> 不产生 ``figure`` 元素；
    - 没有任何表格结构（只有文本）-> 无法计算行列关系；
    - 公式只有文本形态，无 display/inline 与编号字段；
    - 无页级/块级 OCR 能力。
    """

    pdf_path = Path(pdf_path)
    ingestion_root = pdf_path.parent
    audit_module, rebuild_module = _load_p0_dependencies()

    from evidenceflow.domain.document import DocumentSegment
    from evidenceflow.ingestion import clean_document, load_document

    started = sandbox.utc_now()
    # 生产路径在"整份 PDF 无文本层"时会抛 ValueError（这是设计如此：不允许静默返回空文档）。
    # Parser Benchmark 必须把这种情况当作一个**可测量的结果**（O0 在扫描件上必然失败），
    # 而不是让整个 arm 崩掉，因此在这里捕获并记录下来。
    loader_error: str | None = None
    official = None
    try:
        official = clean_document(load_document(pdf_path, ingestion_root))
    except Exception as exc:  # ValueError: PDF contains no extractable text
        loader_error = f"{type(exc).__name__}: {exc}"

    if official is not None:
        document_id = document_id_override or official.document_id
    else:
        from evidenceflow.domain.ids import build_document_id

        document_id = document_id_override or build_document_id(pdf_path.name)

    canonical_segments: list[DocumentSegment] = []
    with pymupdf.open(pdf_path) as pdf:
        page_count = pdf.page_count
        for page_index in range(page_count):
            blocks = rebuild_module.parse_page_blocks(pdf[page_index])
            text = "".join(block["raw_text"] for block in blocks)
            if text.strip():
                canonical_segments.append(
                    DocumentSegment(content=text, page_number=page_index + 1, metadata={})
                )

    # 无文本层（扫描件）：P0 没有 OCR，只能返回空文档而不是编造内容。
    if not canonical_segments:
        return schema.make_document(
            document_id=document_id,
            source_pdf=pdf_path.name,
            source_sha256=sandbox.sha256_file(pdf_path),
            parser="P0_current",
            page_count=page_count,
            elements=[],
            layout_class=layout_class,
            extra={
                "native_text_available": False,
                "loader_error": loader_error,
                "p0_limitation": "no text layer and no OCR capability -> zero elements",
                "started_at_utc": started,
            },
        )

    headings = audit_module.detect_headings(canonical_segments)
    repeated = audit_module.detect_document_furniture(
        [{"document_id": document_id, "segments": canonical_segments}]
    )[document_id]
    structured = {
        "document_id": document_id,
        "file_name": pdf_path.name,
        "segments": canonical_segments,
    }
    furniture_spans = audit_module.page_level_furniture_spans(structured, repeated)
    parsed = rebuild_module.build_structured_document(
        document_id=document_id,
        file_name=pdf_path.name,
        pdf_path=pdf_path,
        cleaned_segments=canonical_segments,
        heading_records=headings,
        furniture_spans=furniture_spans,
    )

    elements: list[dict[str, Any]] = []
    first_title_seen = False
    for index, item in enumerate(parsed["elements"]):
        is_first_page = item["page"] == 1
        is_document_title = False
        if item["element_type"] == "title" and not first_title_seen:
            first_title_seen = True
            is_document_title = True
        element_type, furniture_subtype = schema.map_13b_type(
            item, is_document_title=is_document_title, is_first_page=is_first_page
        )
        # P0 的 caption 归属是"指向所属标题或最近的 caption"，不是指向 figure/table 对象，
        # 因此这里如实记录它给出的 parent_id，不做任何美化。
        elements.append(
            schema.make_element(
                document_id=document_id,
                element_id=schema.element_id_for("p0", document_id, item["page"], index),
                type=element_type,
                raw_text=item["text"],
                source_parser="P0_current",
                page=item["page"],
                bbox=item["bbox"],
                reading_order=index,
                section_path=item.get("section_path") or [],
                parent_id=item.get("parent_id"),
                caption_of=item.get("parent_id") if element_type == "table" else None,
                table_id=None,
                figure_id=None,
                formula_id=None,
                source_ref=f"block:{item.get('block_index')};subtype:{furniture_subtype or ''}",
                confidence=None,
            )
        )

    return schema.make_document(
        document_id=document_id,
        source_pdf=pdf_path.name,
        source_sha256=sandbox.sha256_file(pdf_path),
        parser="P0_current",
        page_count=page_count,
        elements=elements,
        layout_class=layout_class,
        extra={
            "native_text_available": True,
            "heading_records": len(headings),
            "repeated_furniture_count": len(repeated),
            "canonical_segment_count": len(canonical_segments),
            "official_segment_count": len(official.segments) if official is not None else 0,
            "started_at_utc": started,
            "p0_limitations": [
                "image blocks are skipped: zero figure elements",
                "tables are text-only: no row/column/header structure",
                "equations have no display/inline or numbering field",
                "no OCR capability for pages without a text layer",
            ],
        },
    )


# --------------------------------------------------------------------------------------
# 统一的 arm 执行入口
# --------------------------------------------------------------------------------------


def execute_arm(
    arm: ArmSpec,
    *,
    pdf_path: Path,
    out_path: Path,
    ocr_mode: str = "auto",
    document_id_override: str | None = None,
    layout_class: str | None = None,
    timeout: int = 3600,
) -> dict[str, Any]:
    """执行一个 arm 并返回 ``{"document", "run", "availability"}``。"""

    availability = arm_availability(arm)
    if not availability["available"]:
        return {
            "document": None,
            "availability": availability,
            "run": {
                "status": "BLOCKED",
                "reason": availability["reason"],
                "elapsed_seconds": None,
                "peak_ram_mb": None,
            },
        }

    if arm.kind == "inprocess":
        import time

        clock = time.perf_counter()
        document = parse_with_current(
            pdf_path, document_id_override=document_id_override, layout_class=layout_class
        )
        return {
            "document": document,
            "availability": availability,
            "run": {
                "status": "OK",
                "reason": None,
                "elapsed_seconds": round(time.perf_counter() - clock, 3),
                "peak_ram_mb": None,
            },
        }

    run = run_subprocess_arm(
        arm, pdf_path=pdf_path, out_path=out_path, ocr_mode=ocr_mode, timeout=timeout
    )
    if not run.get("ok"):
        return {
            "document": None,
            "availability": availability,
            "run": {
                "status": "ARM_FAILED",
                "reason": run.get("error") or "unknown",
                "elapsed_seconds": None,
                "peak_ram_mb": None,
                "stderr_tail": run.get("stderr", "")[-2000:],
                "stdout_tail": run.get("stdout", "")[-1000:],
            },
        }

    document = load_arm_document(out_path)
    meta = document.get("arm_meta", {})
    return {
        "document": document,
        "availability": availability,
        "run": {
            "status": document.get("status", "OK"),
            "reason": document.get("error"),
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "peak_ram_mb": meta.get("peak_ram_mb"),
            "stderr_tail": run.get("stderr", "")[-2000:],
        },
    }
