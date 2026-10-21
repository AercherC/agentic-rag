"""B3a-2′ 辅助脚本：为已生成的题干产出 Dense Top-K 排名，供 ``e2e_draft.py gates --dense`` 使用。

为什么需要它：``gates`` 的难度标签（``easy`` / ``medium`` / ``hard``，即 gold chunk 落在
Dense Top5 / Top6–20 / Top20 之外）依赖 ``--dense`` 的 ``{query: [chunk_id, ...]}`` 产物；
没有它，``plan_quotas`` 会报 ``difficulty labels missing``，Gate B3 的难度配额无法达标。

为什么必须走 ``build_retriever`` 而不是自己构造索引：``--dense`` 的排名语义必须与线上候选检索
一致（Exact Dense Top-K，CPU 确定性）。自己装配会引入第二套口径，让难度标签与线上行为脱钩。

已知开销：``build_retriever`` 会顺带加载 reranker 模型（本脚本只用 dense）。保持调用它是
可靠性优先——"与线上同路径"比省一次模型加载更重要。

本脚本**只做 embedding 与向量检索，不发任何 LLM 请求**，因此没有 token 成本。

输出保护与 ``e2e_draft.py`` 同口径：任何路径解析后位于仓库 ``evals/`` 下 → 拒绝（退出码 2）。
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
    """输出保护拒绝写入；此时尚未加载任何模型。"""


def check_output(path: Path) -> None:
    """输出保护：``evals/`` 下任何路径一律拒绝（冻结目录不得由本工具写入）。"""

    if path.resolve().is_relative_to(EVALS_ROOT.resolve()):
        raise Refused(f"{path} resolves under evals/ (frozen); refusing to write")


def write_json(path: Path, payload: dict) -> None:
    """UTF-8、键排序、缩进 2、固定 LF 换行，保证同一输入在任意 OS 产出同一字节。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def load_queries(path: Path) -> list[str]:
    """从 draft 产出的 JSONL 读取 query：跳过空行与未生成题干的候选，去重并保持首次出现顺序。

    ``--dry-run`` 的候选（题干为 ``<DRY-RUN>``）不是真实问题，参与检索只会污染排名，因此跳过。
    保持首次出现顺序让输出可复算，也便于与 ``items.jsonl`` 对照。
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
    """对每个 query 跑一次 Exact Dense Top-K，返回 ``{query: [chunk_id, ...]}``。

    依赖注入点 ``build_retriever``：测试用 monkeypatch 替换它，不加载真实模型。
    """

    retriever = build_retriever(Settings())
    dense = retriever.dense_retriever
    ranks: dict[str, list[str]] = {}
    for query in queries:
        # DenseRetriever.retrieve 内部完成 embed_query + 向量检索，与线上候选生成同路径。
        results = dense.retrieve(query, top_k=top_k)
        ranks[query] = [result.chunk.chunk_id for result in results]
    return ranks


def run(args: argparse.Namespace) -> int:
    """读取题干 → CPU 校验 → Dense Top-K → 写排名 JSON。"""

    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer")
    settings = Settings()
    # 数值要可复现：冻结基线都是 CPU 结果，GPU 浮点差异可能改变排序，未显式允许时直接拒绝。
    device = resolve_model_device(settings.model_device)
    if device != "cpu" and not args.allow_gpu:
        raise ValueError(f"resolved device is {device!r}; set MODEL_DEVICE=cpu or pass --allow-gpu")

    queries = load_queries(args.queries)
    ranks = resolve_dense_ranks(queries, top_k=args.top_k)
    # 输出格式必须与 `e2e_draft.py gates --dense` 的读取口径一致：顶层就是 {query: [chunk_id...]}。
    # 不要套一层 {"ranks": ...}，否则 gates 的 dense.get(query) 全部取不到，难度标签会静默全为 None。
    write_json(args.out, ranks)
    print(f"queries={len(queries)} top_k={args.top_k} out={args.out}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queries", type=Path, required=True, help="draft 产出的 items.jsonl")
    parser.add_argument("--out", type=Path, required=True, help='输出 {query: [chunk_id...]} 排名')
    parser.add_argument("--top-k", type=int, default=20, help="Dense 候选深度（默认 20，与线上一致）")
    parser.add_argument("--allow-gpu", action="store_true", help="允许非 CPU 设备（会破坏可复现性）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口：先做输出保护，再执行；``Refused`` → 2，其余异常 → 1。"""

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
    except Exception as exc:  # noqa: BLE001 - CLI 边界统一转换为退出码 1
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
