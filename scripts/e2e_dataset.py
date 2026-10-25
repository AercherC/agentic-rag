"""rag_e2e_v1 dataset tool: deterministic validation (validate) and evidence mapping (map).

PROJECT_SPEC §13.1(a)(b)(c).

Subcommands:
- ``validate --dataset PATH [--require-approved] --report OUT.json [--source-root DIR] [--overwrite]``
  runs every deterministic check on the dataset (collecting issues instead of raising) and writes a
  ``ValidationReport`` as key-sorted JSON. For answer_keys rule 1, page text is read by
  ``load_document`` (uncleaned, i.e. the same text basis as the canonical Gold v3 mapping) from the
  source documents under ``--source-root`` (default ``Settings.source_docs_path``). When the dataset
  carries a fingerprint, it is checked as well (a mismatch is recorded as an issue).
- ``map --dataset PATH --out OUT.json [--source-root DIR] [--overwrite]``
  resolves the active collection via ``runtime.resolve_active_collection``, opens it read-only with
  ``create_if_missing=False`` and calls ``iter_all_chunks``. ``index_signature`` / ``chunk_count`` /
  ``document_count`` come from the active pointer in the lifecycle state file, and are re-verified
  against the chunk set actually read (any count mismatch fails the run). ``chunk_id_hash`` is
  measured under the A1 baseline convention and, when the active manifest records it, compared as
  well; the source documents are loaded as above; ``stable_mapping`` is reused to build an
  ``E2EMapping``, which is written as key-sorted JSON. No model is loaded and no request is issued.

Output protection (identical for both subcommands, and decided before reading the dataset, loading
documents or connecting to Chroma):
- an output path that resolves under the repository's ``evals/`` is always refused (regardless of
  whether the file exists; frozen artifacts must not be written);
- any other output file that already exists without ``--overwrite`` is refused.

Exit codes:
- 0: validate found no issue; map wrote its output and has at least 1 usable mapping with no
  SOURCE_EXTRACTION_FAILURE;
- 2: validate found issues (the report is still written); output protection refused (stderr
  ``REFUSED: <message>``, nothing written); map still wrote its mapping file but usable=0 or a
  SOURCE_EXTRACTION_FAILURE exists (stderr ``REFUSED: <reason>`` for diagnosis; the mapping cannot
  be used for scoring in that case);
- 1: runtime error (dataset cannot be parsed, state file unreadable, collection missing, fingerprint
  mismatch, chunk_count / document_count disagreeing with the active pointer, etc.; stderr
  ``FAILED: <exception type>: <message>``).
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

# Root directory of frozen artifacts: any file already existing under it must never be overwritten.
EVALS_ROOT = ROOT / "evals"
EXIT_OK, EXIT_ERROR, EXIT_ISSUES_OR_REFUSED = 0, 1, 2


class OutputRefused(Exception):
    """Refuse a write under output protection.

    At this point no input has been read and no Chroma connection has been opened.
    """


def check_output(path: Path, *, overwrite: bool) -> None:
    """Output protection.

    Never write any path under `evals/`; other existing files need explicit `--overwrite`. The
    `evals/` check comes before checking whether the file exists: B3b's frozen write must be done by
    the future `freeze` subcommand, so an ordinary tool must not write into the frozen directory even
    when the target file has not been created yet.
    """

    resolved = path.resolve()
    if resolved.is_relative_to(EVALS_ROOT.resolve()):
        raise OutputRefused(f"{path} resolves under evals/ (frozen); refusing even with --overwrite")
    if path.exists() and not overwrite:
        raise OutputRefused(f"{path} already exists; pass --overwrite to replace it")


def write_json(path: Path, payload: dict) -> None:
    """UTF-8, sorted keys, indent 2, fixed LF newlines, so the same input yields the same bytes on
    any OS.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" stops Windows from translating \n into \r\n; otherwise dataset.sha256 would
    # drift per OS.
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def load_source_documents(source_root: Path) -> list[Document]:
    """Load every supported source document under root, in lexicographic relative-path order.

    Only ``load_document`` is used and ``clean_document`` is not: both the anchor location and the
    ``content_version`` of canonical Gold v3 (``experiments/stabilize_gold_v3.py``) are based on the
    uncleaned Loader output, so validation and mapping must use the same text basis.
    """

    paths = sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_TYPES)
    return [load_document(p, source_root) for p in paths]


def make_page_text_provider(documents: list[Document]):
    """Build the ``page_text_provider``: as in ``validate_source``, join Segment text per page with
    ``"\\n"``.
    """

    by_id = {d.document_id: d for d in documents}

    def provider(document_id: str, page_start: int | None, page_end: int | None) -> str:
        document = by_id.get(document_id)
        if document is None:
            raise KeyError(f"document {document_id} not found under source root")
        if page_start is None:
            # MD/TXT have no page numbers: §13.1(c) rule 1 requires taking the whole document.
            return "\n".join(s.content for s in document.segments)
        return "\n".join(s.content for s in document.segments
                         if s.page_number is not None and page_start <= s.page_number <= page_end)

    return provider


def read_dataset(path: Path) -> E2EDataset:
    """Read the dataset without enforcing the fingerprint here.

    Each subcommand handles the fingerprint according to its own semantics.
    """

    return E2EDataset.model_validate_json(path.read_text(encoding="utf-8"))


def read_manifest_chunk_id_hash(manifest_path: str | None) -> str | None:
    """Read `chunk_id_hash` from the active manifest (if recorded).

    Both the top-level and the `index` placement are accepted. The manifest is only optional
    evidence: the current `artifacts/phase15a/active_manifest.json` does not record chunk_id_hash,
    in which case None is returned and only the measured value is written into the mapping file -
    no expected value is invented; the comparison only happens when the manifest records it
    explicitly. A relative `manifest_path` is resolved against the repository root (matching how the
    lifecycle pointer writes it).
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
    """The validate subcommand; issues go into the report and are signalled with exit code 2.

    ``--allow-machine`` must be given explicitly: after the D3 revision the dataset's `approved_by`
    is `machine`, while `validate_dataset` by default only accepts `user`. Without exposing this
    switch, a machine-accepted dataset could never pass validation - the B3b freeze would be stuck
    on `not_approved`.
    ``--enforce-grounding`` corresponds to the four grounding rules of SPEC §13.1(c′) and must be on
    before freezing.
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
    """The map subcommand: open the active collection read-only and write a separate mapping."""

    dataset = read_dataset(args.dataset)
    # When the dataset already carries a fingerprint it must match the content, otherwise the
    # mapping would bind to the wrong dataset identity.
    fingerprint = compute_fingerprint(dataset)
    if dataset.fingerprint is not None and dataset.fingerprint != fingerprint:
        raise ValueError(f"dataset fingerprint mismatch: recorded {dataset.fingerprint}, computed {fingerprint}")
    settings = Settings()
    collection_name = resolve_active_collection(settings)
    state_path = Path(settings.index_state_path)
    if not state_path.is_file():
        # Without the lifecycle state file there is no trustworthy index_signature, so refuse to
        # produce a mapping that cannot bind an index identity.
        raise FileNotFoundError(f"lifecycle state file {state_path} not found; index_signature unavailable")
    state = LifecycleState.model_validate_json(state_path.read_text(encoding="utf-8"))
    if state.active.collection_name != collection_name:
        raise RuntimeError("resolved collection differs from lifecycle active pointer")
    db_path = Path(settings.vector_db_path)
    if not db_path.is_dir():
        # ChromaVectorStore would mkdir the persistence directory; fail fast when it is missing so
        # the read-only semantics hold.
        raise FileNotFoundError(f"vector_db_path {db_path} does not exist")
    store = ChromaVectorStore(db_path, collection_name, create_if_missing=False)
    chunks = list(store.iter_all_chunks())
    # Identity check: the mapping's chunk_count / document_count must agree with the lifecycle
    # active pointer. The pointer's own transcription is not trustworthy; the chunk set actually
    # read is used to re-verify, and a mismatch means pointer and index have drifted apart.
    if len(chunks) != state.active.chunk_count:
        raise ValueError(f"chunk_count mismatch: active pointer {state.active.chunk_count}, store {len(chunks)}")
    actual_document_count = len({chunk.document_id for chunk in chunks})
    if actual_document_count != state.active.document_count:
        raise ValueError(
            f"document_count mismatch: active pointer {state.active.document_count}, store {actual_document_count}"
        )
    # chunk_id_hash serves as an expected value only when the manifest records it explicitly;
    # otherwise only the measured value is recorded.
    expected_chunk_id_hash = read_manifest_chunk_id_hash(state.active.manifest_path)
    source_root = args.source_root or Path(settings.source_docs_path)
    documents = load_source_documents(source_root)
    mapping = map_dataset(dataset, documents, chunks, collection_name=collection_name,
                          index_signature=state.active.index_signature,
                          expected_chunk_count=state.active.chunk_count,
                          expected_chunk_id_hash=expected_chunk_id_hash)
    write_json(args.out, mapping.model_dump(mode="json"))
    # Count usable/unusable records (evidence groups + claim anchors): B3b needs these numbers to
    # record the mapping rate and diagnostics.
    records = [*mapping.groups, *mapping.claim_anchors]
    usable = sum(1 for record in records if is_usable(record.status))
    source_failures = sum(1 for record in records if record.status == "SOURCE_EXTRACTION_FAILURE")
    unusable = len(records) - usable
    print(f"groups={len(mapping.groups)} claim_anchors={len(mapping.claim_anchors)} usable={usable} "
          f"unusable={unusable} source_failures={source_failures} out={args.out}")
    # The mapping file has been written (for diagnosis), but in the two cases below it must not be
    # treated as a successful artifact: either nothing is usable, or a source-text extraction
    # failure is present. Exit code 2 means "has issues / refused", consistent with validate.
    if usable == 0:
        print("REFUSED: mapping has no usable record (usable=0)", file=sys.stderr)
        return EXIT_ISSUES_OR_REFUSED
    if source_failures:
        print(f"REFUSED: mapping has {source_failures} SOURCE_EXTRACTION_FAILURE record(s); "
              f"file written for diagnosis", file=sys.stderr)
        return EXIT_ISSUES_OR_REFUSED
    return EXIT_OK


def run_freeze(args: argparse.Namespace) -> int:
    """The freeze subcommand: B3b's only write entry point.

    It freezes a validated dataset under ``evals/``. This is the **only** path that bypasses
    `check_output`'s ``evals/`` protection, so its preconditions are extremely strict:
    1. the target directory must **not exist** (once frozen it cannot be overwritten; changes
       require a new version directory);
    2. the dataset must first pass every deterministic check (`validate_dataset` reports 0 issues);
       otherwise nothing is written to disk;
    3. `dataset.sha256` must match the **bytes of the file on disk** - so `dataset.json` is written
       first (fixed LF), then read back to compute the sha256, and only then is `dataset.sha256`
       written. The reverse order would record a checksum that does not match the actual file;
    4. `fingerprint` must match the content, so a dataset whose "content and fingerprint disagree"
       cannot be frozen into `evals/`.

    Artifacts (matching the plan §6 deliverables): `dataset.json`, `dataset.sha256`, `README.md`;
    the mapping file is produced separately by `map` and copied in by the caller (this command does
    not generate a mapping, keeping responsibilities clear).
    """

    if args.target.exists():
        raise OutputRefused(f"{args.target} already exists; frozen datasets must not be overwritten")
    dataset = read_dataset(args.dataset)
    # Fingerprint first: if content and fingerprint disagree, the later checks are meaningless.
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
        # Print only the first few issues to avoid flooding the output; the caller should inspect
        # the full set in the report written by `validate` first.
        preview = "; ".join(f"{issue.rule}:{issue.case_id or '-'}" for issue in report.issues[:5])
        raise OutputRefused(f"dataset failed validation ({report.issue_count} issues): {preview}")

    args.target.mkdir(parents=True)
    dataset_path = args.target / "dataset.json"
    # Same basis as e2e_draft.write_dataset: UTF-8, sorted keys, indent 2, fixed LF.
    payload = dataset.model_dump(mode="json")
    dataset_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    # sha256 must be computed over the **bytes written to disk**; reading back before hashing
    # keeps it consistent with the file's actual content.
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
    """Render the frozen directory's README: scale, distributions, known limitations and checksums.

    No timestamp is written, keeping the file recomputable.
    """

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
    """Command-line arguments."""

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
    """Entry point: apply output protection first, then run the subcommand.

    Errors become a single stderr line.
    """

    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        # freeze is the only path allowed to write evals/, so it **must not** go through
        # check_output's evals/ refusal logic. Its preconditions are "the target directory must not
        # exist" plus "validation must pass", enforced by run_freeze itself.
        try:
            return run_freeze(args)
        except OutputRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return EXIT_ISSUES_OR_REFUSED
        except Exception as exc:  # noqa: BLE001 - the CLI boundary converts uniformly to exit code 1
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
    except Exception as exc:  # noqa: BLE001 - the CLI boundary converts uniformly to exit code 1
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
