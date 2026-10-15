# -*- coding: utf-8 -*-
"""A4：通过 Chroma API 删除已确认的孤儿 collection；默认 dry-run，只有 ``--execute`` 才真正删除。

安全约束：
- 允许删除的名字写死在 ``ALLOWED_TO_DELETE``，不接受命令行传入；
- ``FORBIDDEN_TO_DELETE`` 硬编码线上、回滚与历史回滚基线 3 个 collection；
- 动态保护：读取 lifecycle 状态文件，把 active / previous / staging 指针中的 collection 名并入禁止集合；
- 允许列表与禁止集合有交集、或状态文件读不出时拒绝执行（退出码 2），什么都不删；
- 只删除实际存在的名字；删除前把全部 collection 的 name / id / count / metadata 导出到 before JSON，
  ``--execute`` 完成后再导出 after JSON。

用法：

    python scripts/prune_chroma_collections.py            # dry-run：只导出 before 并列出将删除的名字
    python scripts/prune_chroma_collections.py --execute  # 真正删除，并导出 after

退出码：0 = 成功（dry-run 或删除完成）；1 = Chroma 操作失败；2 = 拒绝执行（保护冲突或状态文件不可读）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.knowledge import LifecycleState  # noqa: E402

# A4.2 已确认的孤儿 collection：staging 残留、refmeta 实验、两个结构实验。只能改代码才能变更。
ALLOWED_TO_DELETE: tuple[str, ...] = (
    "ef_research_bge_m3_hybrid_c512_v2_staging",
    "ef_research_bge_m3_c512_o64_refmeta_v1",
    "ef_research_structure_v1",
    "ef_research_structure_v1_elemaware",
)
# 当前 active（v2）、上一代 hybrid（v1）与固定回滚基线（c512_o64_v1），任何情况下都不允许删除。
FORBIDDEN_TO_DELETE: tuple[str, ...] = (
    "ef_research_bge_m3_hybrid_c512_v2",
    "ef_research_bge_m3_hybrid_c512_v1",
    "ef_research_bge_m3_c512_o64_v1",
)
# 硬编码断言：两份写死的名单本身不能有交集（改代码时的第一道防线）。
assert not set(ALLOWED_TO_DELETE) & set(FORBIDDEN_TO_DELETE), "allowlist overlaps hardcoded forbidden list"

DEFAULT_BEFORE = ROOT / "artifacts/stage_a/A4_chroma_before.json"
DEFAULT_AFTER = ROOT / "artifacts/stage_a/A4_chroma_after.json"


class PruneRefused(RuntimeError):
    """执行前的拒绝（退出码 2）：没有删除任何 collection。"""


def lifecycle_protected_names(state_path: Path) -> set[str]:
    """读取状态文件中 active / previous / staging 指针引用的 collection 名；读不出时拒绝。"""

    try:
        state = LifecycleState.model_validate_json(Path(state_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 缺失、损坏、校验失败都不能放行删除
        message = " ".join(str(exc).split())
        raise PruneRefused(f"cannot read lifecycle state {state_path}: {type(exc).__name__}: {message}") from exc
    names = {state.active.collection_name}
    for pointer in (state.previous, state.staging):
        if pointer is not None:
            names.add(pointer.collection_name)
    return names


def forbidden_names(state_path: Path) -> set[str]:
    """禁止集合 = 硬编码禁止名单 ∪ 状态文件指针引用的名字。"""

    return set(FORBIDDEN_TO_DELETE) | lifecycle_protected_names(state_path)


def check_allowlist(allowed: tuple[str, ...], forbidden: set[str]) -> None:
    """允许列表与禁止集合有交集时拒绝：宁可什么都不删，也不能误删线上或回滚索引。"""

    overlap = sorted(set(allowed) & forbidden)
    if overlap:
        raise PruneRefused(f"allowlist contains protected collections: {overlap}; refusing to delete anything")


def export_collections(client: Any) -> list[dict[str, Any]]:
    """导出全部 collection 的 name / id / count / metadata，按名字排序保证输出确定。"""

    rows = []
    for collection in client.list_collections():
        rows.append(
            {
                "name": collection.name,
                "id": str(collection.id),
                "count": collection.count(),
                "metadata": dict(collection.metadata or {}),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """与项目 write_json 一致：UTF-8、不转义中文、键排序。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snapshot(db_path: Path, collections: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "db_path": str(db_path),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "collection_count": len(collections),
        "collections": collections,
    }


def prune(
    db_path: Path,
    state_path: Path,
    *,
    execute: bool,
    before_output: Path,
    after_output: Path,
) -> dict[str, Any]:
    """执行一次清理（或 dry-run），返回摘要；保护冲突时抛 ``PruneRefused``，不删除任何东西。"""

    # 先做全部保护检查，再连接 Chroma：拒绝路径上不会打开存储。
    forbidden = forbidden_names(state_path)
    check_allowlist(ALLOWED_TO_DELETE, forbidden)

    import chromadb

    client = chromadb.PersistentClient(path=str(db_path))
    before = export_collections(client)
    _write_json(before_output, _snapshot(db_path, before, "before"))
    existing = {row["name"] for row in before}
    # 只删除实际存在的名字；不存在的允许名单项只记录，不报错。
    targets = [name for name in ALLOWED_TO_DELETE if name in existing]
    absent = [name for name in ALLOWED_TO_DELETE if name not in existing]
    summary: dict[str, Any] = {
        "mode": "execute" if execute else "dry-run",
        "before_count": len(before),
        "targets": targets,
        "absent_allowed_names": absent,
        "forbidden": sorted(forbidden),
        "deleted": [],
    }
    if not execute:
        summary["would_remain"] = sorted(existing - set(targets))
        return summary
    for name in targets:
        # 删除前再确认一次：防御将来有人改动上面的过滤逻辑。
        if name in forbidden:
            raise PruneRefused(f"refusing to delete protected collection {name!r}")
        client.delete_collection(name)
        summary["deleted"].append(name)
    after = export_collections(client)
    _write_json(after_output, _snapshot(db_path, after, "after"))
    summary["after_count"] = len(after)
    summary["remaining"] = [row["name"] for row in after]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="真正删除（默认 dry-run，只导出并列出将删除的名字）")
    parser.add_argument("--db-path", type=Path, default=None, help="Chroma 持久化目录（默认 Settings.vector_db_path）")
    parser.add_argument("--state-path", type=Path, default=None, help="lifecycle 状态文件（默认 Settings.index_state_path）")
    parser.add_argument("--before-output", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after-output", type=Path, default=DEFAULT_AFTER)
    args = parser.parse_args(argv)

    settings = Settings()
    db_path = args.db_path or Path(settings.vector_db_path)
    state_path = args.state_path or Path(settings.index_state_path)
    try:
        summary = prune(
            db_path,
            state_path,
            execute=args.execute,
            before_output=args.before_output,
            after_output=args.after_output,
        )
    except PruneRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - Chroma 操作失败统一以一行报告
        print(f"FAILED: {type(exc).__name__}: {' '.join(str(exc).split())}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
