"""B3a-2' auxiliary script: produce Dense Top-K rankings for already generated questions, for use by ``e2e_draft.py gates --dense``.

Why it is needed: the difficulty labels in ``gates`` (``easy`` / ``medium`` / ``hard``, i.e. whether the
gold chunk lands in Dense Top5 / Top6-20 / outside Top20) depend on the ``{query: [chunk_id, ...]}``
artifact passed to ``--dense``; without it ``plan_quotas`` reports ``difficulty labels missing`` and the
Gate B3 difficulty quotas cannot be met.

Why it must go through ``build_retriever`` instead of assembling an index itself: the ranking semantics of
``--dense`` must match serving candidate retrieval (Exact Dense Top-K, CPU-deterministic). Assembling it by
hand would introduce a second set of semantics and decouple the difficulty labels from serving behavior.

Known overhead: ``build_retriever`` also loads the reranker model (this script only uses dense). Keeping
that call is a reliability-first choice: "the same path as serving" matters more than saving one model load.

This script **only does embedding and vector retrieval and sends no LLM request**, so it has no token cost.

Output protection follows the same rules as ``e2e_draft.py``: any path that resolves under the repository's
``evals/`` is refused (exit code 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.model_device import resolve_model_device  # noqa: E402
from evidenceflow.runtime import build_retriever  # noqa: E402

EVALS_ROOT = ROOT / "evals"
EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


class Refused(Exception):
    """The output guard refused the write; no model has been loaded at that point."""


def check_output(path: Path) -> None:
    """Output guard: refuse any path under ``evals/`` (the frozen directory must not be written by this tool)."""

    if path.resolve().is_relative_to(EVALS_ROOT.resolve()):
        raise Refused(f"{path} resolves under evals/ (frozen); refusing to write")


def write_json(path: Path, payload: dict) -> None:
    """UTF-8, sorted keys, 2-space indent and a fixed LF newline, so the same input yields the same bytes on any OS."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def load_queries(path: Path) -> list[str]:
    """Read queries from the JSONL produced by draft: skip empty lines and candidates with no generated question, deduplicate and keep first-appearance order.

    ``--dry-run`` candidates (whose question is ``<DRY-RUN>``) are not real questions and would only pollute
    the rankings if retrieved, so they are skipped. Keeping first-appearance order makes the output
    recomputable and easy to cross-check against ``items.jsonl``.
    """

    queries: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        query = str(item.get("query") or "").strip()
        if not query or query == "<DRY-RUN>" or query in seen:
            continue
        seen.add(query)
        queries.append(query)
    return queries


def resolve_dense_ranks(queries: list[str], *, top_k: int) -> dict[str, list[str]]:
    """Run Exact Dense Top-K once per query and return ``{query: [chunk_id, ...]}``.

    The injection point ``build_retriever`` is what tests monkeypatch, so no real model is loaded.
    """

    retriever = build_retriever(Settings())
    dense = retriever.dense_retriever
    ranks: dict[str, list[str]] = {}
    for query in queries:
        # DenseRetriever.retrieve performs embed_query + vector retrieval internally, the same path
        # as
        # serving candidate generation.
        results = dense.retrieve(query, top_k=top_k)
        ranks[query] = [result.chunk.chunk_id for result in results]
    return ranks


def run(args: argparse.Namespace) -> int:
    """Read questions -> CPU check -> Dense Top-K -> write the ranking JSON."""

    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer")
    settings = Settings()
    # The numbers must be reproducible: the frozen baselines are all CPU results and GPU
    # floating-point
    # differences can change the ordering, so refuse unless that is explicitly allowed.
    device = resolve_model_device(settings.model_device)
    if device != "cpu" and not args.allow_gpu:
        raise ValueError(f"resolved device is {device!r}; set MODEL_DEVICE=cpu or pass --allow-gpu")

    queries = load_queries(args.queries)
    ranks = resolve_dense_ranks(queries, top_k=args.top_k)
    # The output shape must match what `e2e_draft.py gates --dense` reads: the top level is {query:
    # [chunk_id...]}.
    # Do not wrap it in {"ranks": ...}, otherwise every gates dense.get(query) misses and the
    # difficulty
    # labels silently become None everywhere.
    write_json(args.out, ranks)
    print(f"queries={len(queries)} top_k={args.top_k} out={args.out}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queries", type=Path, required=True, help="draft 产出的 items.jsonl")
    parser.add_argument("--out", type=Path, required=True, help='输出 {query: [chunk_id...]} 排名')
    parser.add_argument("--top-k", type=int, default=20, help="Dense 候选深度（默认 20，与线上一致）")
    parser.add_argument("--allow-gpu", action="store_true", help="允许非 CPU 设备（会破坏可复现性）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: apply the output guard first, then run; ``Refused`` -> 2, any other exception -> 1."""

    args = build_parser().parse_args(argv)
    try:
        check_output(args.out)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        return run(args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001 - the CLI boundary converts everything to exit code 1
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
