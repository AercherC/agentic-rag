"""rag_e2e_v1 题集工具：确定性校验（validate）与证据映射（map）。PROJECT_SPEC §13.1(a)(b)(c)。

子命令：
- ``validate --dataset PATH [--require-approved] --report OUT.json [--source-root DIR] [--overwrite]``
  对题集做全部确定性检查（不抛错、汇总问题），把 ``ValidationReport`` 写成键排序 JSON。
  answer_keys 规则 ① 所需的页文本由 ``load_document``（未经 clean，与 canonical Gold v3 映射同一文本
  基础）读取 ``--source-root``（默认 ``Settings.source_docs_path``）下的源文档。
  题集带 fingerprint 时一并校验（不一致记为问题）。
- ``map --dataset PATH --out OUT.json [--source-root DIR] [--overwrite]``
  用 ``runtime.resolve_active_collection`` 确定 active collection，以 ``create_if_missing=False`` 只读打开并
  ``iter_all_chunks``；``index_signature`` / ``chunk_count`` / ``document_count`` 取自 lifecycle 状态文件
  active 指针，并用实测 chunk 集合复核（数量不一致即失败）；``chunk_id_hash`` 按 A1 基线口径实测，
  active manifest 记录该值时一并比对；源文档同上加载；复用 ``stable_mapping`` 生成 ``E2EMapping``
  写成键排序 JSON。不加载任何模型、不发任何请求。

输出保护（两个子命令相同，且在读取题集、加载文档、连接 Chroma 之前完成判断）：
- 输出路径解析后位于仓库 ``evals/`` 下 → 一律拒绝（无论文件是否已存在；冻结产物不可写入）；
- 其他输出文件已存在且未传 ``--overwrite`` → 拒绝。

退出码：
- 0：validate 没有任何问题；map 成功写出且至少有 1 条可用映射、无 SOURCE_EXTRACTION_FAILURE；
- 2：validate 发现问题（报告照常写出）；输出保护拒绝（stderr ``REFUSED: <消息>``，什么都没写）；
  map 的映射文件仍照常写出但 usable=0 或存在 SOURCE_EXTRACTION_FAILURE（stderr ``REFUSED: <原因>``，
  便于诊断；此时映射不可用于计分）；
- 1：运行错误（题集无法解析、状态文件不可读、collection 不存在、fingerprint 不一致、
  chunk_count / document_count 与 active 指针不一致等；stderr ``FAILED: <异常类型>: <消息>``）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.domain.document import Document  # noqa: E402
from evidenceflow.eval.e2e.dataset import (  # noqa: E402
    E2EDataset,
    compute_fingerprint,
    validate_dataset,
)
from evidenceflow.eval.e2e.mapping import is_usable, map_dataset  # noqa: E402
from evidenceflow.ingestion.loader import SUPPORTED_FILE_TYPES, load_document  # noqa: E402
from evidenceflow.knowledge import LifecycleState  # noqa: E402
from evidenceflow.runtime import resolve_active_collection  # noqa: E402
from evidenceflow.storage.chroma import ChromaVectorStore  # noqa: E402

# 冻结产物根目录：其下已存在的文件一律不可覆盖。
EVALS_ROOT = ROOT / "evals"
EXIT_OK, EXIT_ERROR, EXIT_ISSUES_OR_REFUSED = 0, 1, 2


class OutputRefused(Exception):
    """输出保护拒绝写入；此时尚未读取任何输入或连接 Chroma。"""


def check_output(path: Path, *, overwrite: bool) -> None:
    """输出保护：`evals/` 下任何路径绝不写入；其他已存在文件需显式 `--overwrite`。

    先做 `evals/` 判断而不是先看文件是否存在：B3b 的冻结写入必须由将来的 `freeze` 子命令完成，
    普通工具即使目标文件尚未创建也不得往冻结目录里写。
    """

    resolved = path.resolve()
    if resolved.is_relative_to(EVALS_ROOT.resolve()):
        raise OutputRefused(f"{path} resolves under evals/ (frozen); refusing even with --overwrite")
    if path.exists() and not overwrite:
        raise OutputRefused(f"{path} already exists; pass --overwrite to replace it")


def write_json(path: Path, payload: dict) -> None:
    """UTF-8、键排序、缩进 2、固定 LF 换行，保证同一输入在任意 OS 上产出同一字节。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" 阻止 Windows 把 \n 翻译成 \r\n；否则 dataset.sha256 会随 OS 漂移。
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def load_source_documents(source_root: Path) -> list[Document]:
    """按相对路径字典序加载 root 下全部受支持的源文档。

    只用 ``load_document``、不做 ``clean_document``：canonical Gold v3（``experiments/stabilize_gold_v3.py``）
    的锚点定位与 ``content_version`` 都基于未清洗的 Loader 输出，校验与映射必须用同一文本基础。
    """

    paths = sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_TYPES)
    return [load_document(p, source_root) for p in paths]


def make_page_text_provider(documents: list[Document]):
    """构造 ``page_text_provider``：与 ``validate_source`` 相同，按页用 ``"\\n"`` 拼接 Segment 原文。"""

    by_id = {d.document_id: d for d in documents}

    def provider(document_id: str, page_start: int | None, page_end: int | None) -> str:
        document = by_id.get(document_id)
        if document is None:
            raise KeyError(f"document {document_id} not found under source root")
        if page_start is None:
            # MD/TXT 无页码：§13.1(c)① 规定取整篇。
            return "\n".join(s.content for s in document.segments)
        return "\n".join(s.content for s in document.segments
                         if s.page_number is not None and page_start <= s.page_number <= page_end)

    return provider


def read_dataset(path: Path) -> E2EDataset:
    """读取题集，不在此处强制校验 fingerprint（由各子命令按语义处理）。"""

    return E2EDataset.model_validate_json(path.read_text(encoding="utf-8"))


def read_manifest_chunk_id_hash(manifest_path: str | None) -> str | None:
    """从 active manifest 读取 `chunk_id_hash`（若记录）；兼容顶层与 `index` 两种位置。

    manifest 只是可选证据：当前 `artifacts/phase15a/active_manifest.json` 没有记录 chunk_id_hash，
    此时返回 None，只把实测值写进映射文件，不自造期望值；只有当 manifest 明确记录时才参与比对。
    `manifest_path` 为相对路径时按仓库根目录解析（与 lifecycle 指针的写法一致）。
    """

    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    index = payload.get("index")
    candidates = [payload.get("chunk_id_hash")]
    if isinstance(index, dict):
        candidates.append(index.get("chunk_id_hash"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def run_validate(args: argparse.Namespace) -> int:
    """validate 子命令；问题写进报告并以退出码 2 表示。

    ``--allow-machine`` 必须显式给出：D3 修订后题集的 `approved_by` 是 `machine`，
    而 `validate_dataset` 的默认语义只接受 `user`。不暴露这个开关，机器验收的题集
    就永远无法通过校验——B3b 冻结会卡在 `not_approved` 上。
    ``--enforce-grounding`` 对应 SPEC §13.1(c′) 的四条接地规则，冻结前必须打开。
    """

    dataset = read_dataset(args.dataset)
    source_root = args.source_root or Path(Settings().source_docs_path)
    provider = make_page_text_provider(load_source_documents(source_root))
    report = validate_dataset(
        dataset,
        provider,
        require_approved=args.require_approved,
        allow_machine=args.allow_machine,
        enforce_grounding=args.enforce_grounding,
    )
    write_json(args.report, report.model_dump(mode="json"))
    print(f"issues={report.issue_count} cases={report.case_count} report={args.report}")
    return EXIT_OK if report.ok else EXIT_ISSUES_OR_REFUSED


def run_map(args: argparse.Namespace) -> int:
    """map 子命令：只读打开 active collection，生成与题集分离的映射文件。"""

    dataset = read_dataset(args.dataset)
    # 题集已带 fingerprint 时必须与内容一致，否则映射会绑定到错误的题集身份。
    fingerprint = compute_fingerprint(dataset)
    if dataset.fingerprint is not None and dataset.fingerprint != fingerprint:
        raise ValueError(f"dataset fingerprint mismatch: recorded {dataset.fingerprint}, computed {fingerprint}")
    settings = Settings()
    collection_name = resolve_active_collection(settings)
    state_path = Path(settings.index_state_path)
    if not state_path.is_file():
        # 没有 lifecycle 状态文件就没有可信的 index_signature，拒绝生成无法绑定索引身份的映射。
        raise FileNotFoundError(f"lifecycle state file {state_path} not found; index_signature unavailable")
    state = LifecycleState.model_validate_json(state_path.read_text(encoding="utf-8"))
    if state.active.collection_name != collection_name:
        raise RuntimeError("resolved collection differs from lifecycle active pointer")
    db_path = Path(settings.vector_db_path)
    if not db_path.is_dir():
        # ChromaVectorStore 会 mkdir 持久化目录；目录不存在时直接失败，保持只读语义。
        raise FileNotFoundError(f"vector_db_path {db_path} does not exist")
    store = ChromaVectorStore(db_path, collection_name, create_if_missing=False)
    chunks = list(store.iter_all_chunks())
    # 身份核对：mapping 的 chunk_count / document_count 必须与 lifecycle active 指针一致。
    # 指针抄写本身不可信；这里用真实读到的 chunk 集合复核，不一致说明指针与索引已漂移。
    if len(chunks) != state.active.chunk_count:
        raise ValueError(f"chunk_count mismatch: active pointer {state.active.chunk_count}, store {len(chunks)}")
    actual_document_count = len({chunk.document_id for chunk in chunks})
    if actual_document_count != state.active.document_count:
        raise ValueError(
            f"document_count mismatch: active pointer {state.active.document_count}, store {actual_document_count}"
        )
    # chunk_id_hash 只有 manifest 明确记录时才作为期望值；否则只记录实测值。
    expected_chunk_id_hash = read_manifest_chunk_id_hash(state.active.manifest_path)
    source_root = args.source_root or Path(settings.source_docs_path)
    documents = load_source_documents(source_root)
    mapping = map_dataset(dataset, documents, chunks, collection_name=collection_name,
                          index_signature=state.active.index_signature,
                          expected_chunk_count=state.active.chunk_count,
                          expected_chunk_id_hash=expected_chunk_id_hash)
    write_json(args.out, mapping.model_dump(mode="json"))
    # 统计可用/不可用记录（证据组 + claim 锚点）：B3b 需要这组数字记录映射率与诊断。
    records = [*mapping.groups, *mapping.claim_anchors]
    usable = sum(1 for record in records if is_usable(record.status))
    source_failures = sum(1 for record in records if record.status == "SOURCE_EXTRACTION_FAILURE")
    unusable = len(records) - usable
    print(f"groups={len(mapping.groups)} claim_anchors={len(mapping.claim_anchors)} usable={usable} "
          f"unusable={unusable} source_failures={source_failures} out={args.out}")
    # 映射文件已写出（便于诊断），但以下两种情况不允许把该映射当作成功产物：全不可用，或
    # 存在源文本抽取失败。退出码 2 表示“有问题/被拒绝”，与 validate 的语义一致。
    if usable == 0:
        print("REFUSED: mapping has no usable record (usable=0)", file=sys.stderr)
        return EXIT_ISSUES_OR_REFUSED
    if source_failures:
        print(f"REFUSED: mapping has {source_failures} SOURCE_EXTRACTION_FAILURE record(s); "
              f"file written for diagnosis", file=sys.stderr)
        return EXIT_ISSUES_OR_REFUSED
    return EXIT_OK


def run_freeze(args: argparse.Namespace) -> int:
    """freeze 子命令：B3b 的唯一写入口，把校验通过的题集冻结到 ``evals/`` 下。

    这是**唯一**绕过 `check_output` 的 ``evals/`` 保护路径，因此保护条件极其严格：
    1. 目标目录必须**不存在**（一旦冻结就不能覆盖；要改只能新开版本目录）；
    2. 题集必须先通过全部确定性校验（`validate_dataset` 的 issues 为 0），否则拒绝落盘；
    3. `dataset.sha256` 必须与**磁盘上那份文件的字节**一致——因此先写 `dataset.json`
       （固定 LF），再读回字节算 sha256，最后写 `dataset.sha256`。顺序反了就会写出一个
       与实际文件不匹配的校验值；
    4. `fingerprint` 必须与内容一致，避免把"内容与指纹不符"的题集冻进 `evals/`。

    产物（与计划 §6 交付物一致）：`dataset.json`、`dataset.sha256`、`README.md`；
    映射文件由 `map` 单独产出后由调用方复制进来（本命令不生成 mapping，避免职责混淆）。
    """

    if args.target.exists():
        raise OutputRefused(f"{args.target} already exists; frozen datasets must not be overwritten")
    dataset = read_dataset(args.dataset)
    # 指纹先行：内容与指纹不符时后面的校验都没有意义。
    fingerprint = compute_fingerprint(dataset)
    if dataset.fingerprint is not None and dataset.fingerprint != fingerprint:
        raise ValueError(
            f"dataset fingerprint mismatch: recorded {dataset.fingerprint}, computed {fingerprint}"
        )
    source_root = args.source_root or Path(Settings().source_docs_path)
    provider = make_page_text_provider(load_source_documents(source_root))
    report = validate_dataset(
        dataset, provider, require_approved=True, allow_machine=True, enforce_grounding=True
    )
    if not report.ok:
        # 只挑前几条打印，避免刷屏；完整问题由调用方先用 `validate` 落到报告里查看。
        preview = "; ".join(f"{issue.rule}:{issue.case_id or '-'}" for issue in report.issues[:5])
        raise OutputRefused(f"dataset failed validation ({report.issue_count} issues): {preview}")

    args.target.mkdir(parents=True)
    dataset_path = args.target / "dataset.json"
    # 与 e2e_draft.write_dataset 同口径：UTF-8、键排序、缩进 2、固定 LF。
    payload = dataset.model_dump(mode="json")
    dataset_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    # sha256 必须对**落盘后的字节**计算；先读回再算，才能保证与文件实际内容一致。
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    (args.target / "dataset.sha256").write_text(f"{digest}  dataset.json\n", encoding="utf-8", newline="\n")
    (args.target / "README.md").write_text(
        _render_freeze_readme(dataset, fingerprint=fingerprint, digest=digest, report=report),
        encoding="utf-8", newline="\n",
    )
    print(
        f"frozen cases={len(dataset.cases)} fingerprint={fingerprint} sha256={digest} "
        f"target={args.target}"
    )
    return EXIT_OK


def _render_freeze_readme(dataset, *, fingerprint: str, digest: str, report) -> str:
    """生成冻结目录的 README：规模、分布、已知局限与校验值（不写时间戳，保证可复算）。"""

    from collections import Counter

    cases = dataset.cases
    kinds = Counter(str(case.unanswerable_kind) for case in cases)
    types = Counter(case.question_type for case in cases)
    languages = Counter(case.query_language for case in cases)
    sources = Counter(case.provenance.source for case in cases)
    near_miss = sum(1 for case in cases if case.unanswerable_kind == "near_miss")
    out_of_corpus = sum(1 for case in cases if case.unanswerable_kind == "out_of_corpus")
    denominator = near_miss + out_of_corpus
    lines = [
        f"# {dataset.dataset_id}",
        "",
        "- 合同：`PROJECT_SPEC.md` §13.1（rag_e2e_v1）",
        f"- 规模：{len(cases)} 题",
        f"- dataset fingerprint：`{fingerprint}`",
        f"- dataset.sha256（LF 字节）：`{digest}`",
        f"- corpus_version：`{dataset.corpus_version}`",
        "",
        "## 分布",
        "",
        "| 维度 | 取值 |",
        "|---|---|",
        f"| expected_status | {dict(sorted(Counter(c.expected_status for c in cases).items()))} |",
        f"| unanswerable_kind | {dict(sorted(kinds.items()))} |",
        f"| question_type | {dict(sorted(types.items()))} |",
        f"| query_language | {dict(sorted(languages.items()))} |",
        f"| provenance.source | {dict(sorted(sources.items()))} |",
        "",
        f"`near_miss` 占两类不可答题：{near_miss}/{denominator} = "
        f"{(near_miss / denominator) if denominator else 0:.2f}（Gate 要求 ≥0.50）",
        "",
        "## 校验",
        "",
        f"冻结前 `validate_dataset(require_approved=True, allow_machine=True, enforce_grounding=True)`："
        f"issues = {report.issue_count}，case_count = {report.case_count}。",
        "",
        "## 已知局限",
        "",
        "1. `near_miss` 的实体×属性共现判定使用 **200 字符邻近窗口**；换窗口会改变候选集合，",
        "   因此该阈值属于冻结口径的一部分，不得在后续阶段修改。",
        "2. 两类不可答题不参与证据映射，因此不进入 `context_all_required` / "
        "`context_claim_recall` 的分母。",
        "3. `out_of_corpus` 题复用 Phase 12 的 `GEN-U01..05`，其 absence_terms 零命中已在",
        "   当前 817-chunk 语料上复核；语料若再变更，必须先重新复核。",
        "4. 难度标签来自候选 query 的 Dense Top20 排名，只作分片标签，不作为淘汰依据。",
        "",
        "## 不要修改本目录",
        "",
        "`evals/` 下全部为冻结产物。要改题集只能新开版本目录，并同步更新 `PROJECT_SPEC.md`。",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="确定性校验题集")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--require-approved", action="store_true")
    validate.add_argument("--allow-machine", action="store_true", dest="allow_machine",
                          help="接受 approved_by='machine'（D3 派生题集的默认形态）")
    validate.add_argument("--enforce-grounding", action="store_true", dest="enforce_grounding",
                          help="执行 SPEC §13.1(c′) 的四条接地规则（冻结前必须打开）")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--source-root", type=Path, default=None)
    validate.add_argument("--overwrite", action="store_true")
    mapping = sub.add_parser("map", help="把题集锚点映射到 active 索引")
    mapping.add_argument("--dataset", type=Path, required=True)
    mapping.add_argument("--out", type=Path, required=True)
    mapping.add_argument("--source-root", type=Path, default=None)
    mapping.add_argument("--overwrite", action="store_true")
    freeze = sub.add_parser("freeze", help="把校验通过的题集冻结到 evals/ 下（B3b 唯一写入口）")
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--target", type=Path, required=True,
                        help="冻结目录，必须不存在，如 evals/rag_e2e_v1")
    freeze.add_argument("--source-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口：先做输出保护，再执行子命令；错误统一转成一行 stderr。"""

    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        # freeze 是唯一允许写 evals/ 的路径，因此**不能**走 check_output 的 evals/ 拒绝逻辑。
        # 它的保护条件是「目标目录必须不存在」+「校验必须通过」，由 run_freeze 自己执行。
        try:
            return run_freeze(args)
        except OutputRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return EXIT_ISSUES_OR_REFUSED
        except Exception as exc:  # noqa: BLE001 - CLI 边界统一转换为退出码 1
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return EXIT_ERROR
    output = args.report if args.command == "validate" else args.out
    try:
        check_output(output, overwrite=args.overwrite)
    except OutputRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_ISSUES_OR_REFUSED
    try:
        return run_validate(args) if args.command == "validate" else run_map(args)
    except Exception as exc:  # noqa: BLE001 - CLI 边界统一转换为退出码 1
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
