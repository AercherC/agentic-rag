"""rag_e2e_v1 失败根因诊断：逐阶段观测（observe）与失败层归因（attribute）。PROJECT_SPEC §13.1(e)。

子命令：
- ``observe --dataset PATH --out OUT.jsonl [--retrieval-only] [--depth N] [--tag TAG] [--overwrite]``
  用线上同一装配（``build_retriever`` / ``build_context_builder`` / ``build_generator``）逐题运行，
  记录每一阶段保留了哪些 chunk：Exact Dense 排名（深度 ``--depth``，默认 100，超过线上 Top20，
  用来回答“候选池加深能否找回证据”）、Reranker 对候选池的完整排序、最终 Top-K、BuiltContext 纳入的
  chunk、回答状态、正文与实际引用。阶段顺序与 ``RAGService.answer`` 相同（Retrieve → Context →
  Generate → Citation），区别只是把中间结果留下来。``--retrieval-only`` 不调用 LLM（零 token 成本），
  只能归因检索与上下文层。``--tag`` 只跑带该 tag 的题（如 ``multi_doc``）；默认跑全部题，
  以便多文档题与单文档题对照。
- ``attribute --dataset PATH --mapping MAPPING.json --observations OBS.jsonl --out REPORT.json
  [--markdown REPORT.md] [--overwrite]``
  纯计算：核对题集 fingerprint、映射与 observation 的索引身份后，按 §13.1(e) 规则逐题归因，并给出逐证据组
  的止步阶段与根因信号（见 ``evidenceflow.eval.e2e.diagnosis``）。不加载模型、不发请求。

映射文件由 ``scripts/e2e_dataset.py map`` 生成，必须与 observe 时的 active 索引一致。

输出保护（与 ``e2e_dataset.py`` 相同，在读取任何输入之前判断）：输出路径位于 ``evals/`` 下 → 拒绝；
其他输出已存在且未传 ``--overwrite`` → 拒绝。

退出码：0 成功；2 输出保护拒绝；1 运行错误（stderr ``FAILED: <异常类型>: <消息>``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.citation import CitationValidationError, assemble_rag_answer  # noqa: E402
from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.eval.e2e.dataset import E2EDataset, EvalCase, compute_fingerprint  # noqa: E402
from evidenceflow.eval.e2e.diagnosis import (  # noqa: E402
    E2EObservation,
    ObservedChunk,
    build_report,
    diagnose_dataset,
    render_markdown,
)
from evidenceflow.eval.e2e.mapping import E2EMapping  # noqa: E402
from evidenceflow.generation import GenerationContractError  # noqa: E402
from evidenceflow.llm import LLMRequestError  # noqa: E402
from evidenceflow.runtime import (  # noqa: E402
    build_context_builder,
    build_generator,
    build_retriever,
    resolve_active_collection,
)

EVALS_ROOT = ROOT / "evals"
EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
# 这三类是生成/引用合同失败，按 §13.1(e) 规则 1 归层；其余异常（检索、配置）直接中止整次运行。
_RECORDED_ERRORS = (LLMRequestError, GenerationContractError, CitationValidationError)


class OutputRefused(Exception):
    """输出保护拒绝写入；此时尚未读取任何输入。"""


def check_output(path: Path, *, overwrite: bool) -> None:
    """``evals/`` 下一律拒绝；已存在且未允许覆盖时拒绝。"""

    if path.resolve().is_relative_to(EVALS_ROOT.resolve()):
        raise OutputRefused(f"{path} resolves under evals/ (frozen); refusing to write")
    if path.exists() and not overwrite:
        raise OutputRefused(f"{path} already exists; pass --overwrite to replace it")


def write_text(path: Path, text: str) -> None:
    """UTF-8、固定 LF，保证同一输入在任意 OS 产出同一字节。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def dump_json(payload: dict) -> str:
    """键排序、缩进 2 的 JSON 文本。"""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def read_dataset(path: Path) -> tuple[E2EDataset, str]:
    """读取题集并返回 (题集, fingerprint)；题集自带 fingerprint 时必须一致。"""

    dataset = E2EDataset.model_validate_json(path.read_text(encoding="utf-8-sig"))
    fingerprint = compute_fingerprint(dataset)
    if dataset.fingerprint is not None and dataset.fingerprint != fingerprint:
        raise ValueError("dataset fingerprint does not match its content")
    return dataset, fingerprint


def observe_case(
    case: EvalCase,
    *,
    retriever,
    context_builder,
    generator,
    depth: int,
    retrieval_only: bool,
    collection_name: str,
    chunk_count: int,
) -> E2EObservation:
    """按 RAGService.answer 的阶段顺序运行一题，并保留每一阶段的结果。

    最终 Top-K 取 Reranker 对整个候选池排序的前 K 项：``LearnedReranker`` 先对全部候选打分排序再截断，
    所以与线上 ``retriever.retrieve(query)`` 返回的 Top-K 逐项相同。
    """

    query = case.query
    pool_k = retriever.candidate_pool_k
    dense = retriever.dense_retriever.retrieve(query, top_k=max(depth, pool_k))
    reranked = retriever.retrieve(query, top_k=retriever.reranker_input_k)
    final = reranked[: retriever.final_top_k]
    context = context_builder.build(final)
    record = {
        "case_id": case.case_id,
        "collection_name": collection_name,
        "chunk_count": chunk_count,
        "dense_ranking": [
            ObservedChunk(
                chunk_id=item.chunk.chunk_id,
                document_id=item.chunk.document_id,
                heading_path=item.chunk.metadata.get("heading_path"),
            )
            for item in dense
        ],
        "candidate_pool_k": pool_k,
        "reranked_pool": [item.chunk.chunk_id for item in reranked],
        "final_top_k": retriever.final_top_k,
        "context_chunk_ids": [entry.chunk_id for entry in context.entries],
        "generation_ran": not retrieval_only,
    }
    if retrieval_only:
        return E2EObservation(**record)
    try:
        generated = generator.generate(query, context)
        answer = assemble_rag_answer(
            query=query, generated_answer=generated, built_context=context, retrieved_chunks=final
        )
    except _RECORDED_ERRORS as exc:
        return E2EObservation(**record, error_type=type(exc).__name__)
    return E2EObservation(
        **record,
        status=answer.status,
        answer=answer.answer,
        source_chunk_ids=[source.chunk_id for source in answer.sources],
    )


def run_observe(args: argparse.Namespace) -> int:
    """装配线上组件（只装配一次），逐题观测并写 JSONL。"""

    if args.depth <= 0:
        raise ValueError("--depth must be a positive integer")
    dataset, _fingerprint = read_dataset(args.dataset)
    cases = [case for case in dataset.cases if args.tag is None or args.tag in case.tags]
    if not cases:
        raise ValueError(f"no cases selected (tag={args.tag!r})")
    settings = Settings()
    collection_name = resolve_active_collection(settings)
    retriever = build_retriever(settings)
    context_builder = build_context_builder(settings)
    generator = None if args.retrieval_only else build_generator(settings)
    chunk_count = retriever.dense_retriever.vector_store.size
    lines = []
    for case in cases:
        observation = observe_case(
            case,
            retriever=retriever,
            context_builder=context_builder,
            generator=generator,
            depth=args.depth,
            retrieval_only=args.retrieval_only,
            collection_name=collection_name,
            chunk_count=chunk_count,
        )
        lines.append(json.dumps(observation.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    write_text(args.out, "\n".join(lines) + "\n")
    print(f"cases={len(cases)} retrieval_only={args.retrieval_only} out={args.out}")
    return EXIT_OK


def read_observations(path: Path) -> list[E2EObservation]:
    """读取 observe 产出的 JSONL；空行跳过，同一题重复出现即报错。"""

    observations = [
        E2EObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    case_ids = [o.case_id for o in observations]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case_id in observations")
    if not observations:
        raise ValueError("no observations")
    return observations


def _single_value(observations: list[E2EObservation], field: str):
    """所有 observation 的某配置字段必须相同（同一次运行）。"""

    values = {getattr(o, field) for o in observations}
    if len(values) != 1:
        raise ValueError(f"observations disagree on {field}: {sorted(map(str, values))}")
    return values.pop()


def run_attribute(args: argparse.Namespace) -> int:
    """核对身份 → 逐题归因 → 写 JSON 报告（与可选的 Markdown 对照表）。"""

    dataset, fingerprint = read_dataset(args.dataset)
    mapping = E2EMapping.model_validate_json(args.mapping.read_text(encoding="utf-8-sig"))
    observations = read_observations(args.observations)
    diagnoses = diagnose_dataset(dataset, mapping, observations, fingerprint)
    report = build_report(
        diagnoses,
        candidate_pool_k=_single_value(observations, "candidate_pool_k"),
        final_top_k=_single_value(observations, "final_top_k"),
        generation_ran=_single_value(observations, "generation_ran"),
    )
    write_text(args.out, dump_json(report))
    if args.markdown is not None:
        write_text(args.markdown, render_markdown(report))
    multi = report["slices"]["multi_doc"]
    print(f"cases={len(diagnoses)} multi_doc={multi['cases']} layers={multi['question_layers']} out={args.out}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    observe = sub.add_parser("observe", help="逐题运行线上链路并记录各阶段结果")
    observe.add_argument("--dataset", type=Path, required=True)
    observe.add_argument("--out", type=Path, required=True)
    observe.add_argument("--retrieval-only", action="store_true", help="不调用 LLM，只观测检索与上下文层")
    observe.add_argument("--depth", type=int, default=100, help="记录的 Dense 排名深度（默认 100）")
    observe.add_argument("--tag", default=None, help="只跑带该 tag 的题，如 multi_doc")
    observe.add_argument("--overwrite", action="store_true")
    attribute = sub.add_parser("attribute", help="按 §13.1(e) 归因并输出诊断报告")
    attribute.add_argument("--dataset", type=Path, required=True)
    attribute.add_argument("--mapping", type=Path, required=True)
    attribute.add_argument("--observations", type=Path, required=True)
    attribute.add_argument("--out", type=Path, required=True)
    attribute.add_argument("--markdown", type=Path, default=None)
    attribute.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口：先做输出保护，再执行；拒绝 → 2，其余异常 → 1。"""

    args = build_parser().parse_args(argv)
    outputs = [args.out] + ([args.markdown] if getattr(args, "markdown", None) is not None else [])
    try:
        for path in outputs:
            check_output(path, overwrite=args.overwrite)
    except OutputRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        return run_observe(args) if args.command == "observe" else run_attribute(args)
    except Exception as exc:  # noqa: BLE001 - CLI 边界统一转换为退出码 1
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
