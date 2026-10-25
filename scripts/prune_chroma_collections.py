# -*- coding: utf-8 -*-
"""A4: delete the confirmed orphan collections through the Chroma API; dry-run by default,
only ``--execute`` really deletes.

Safety constraints:
- the names that may be deleted are hardcoded in ``ALLOWED_TO_DELETE`` and are not taken
  from the command line;
- ``FORBIDDEN_TO_DELETE`` hardcodes the 3 collections for production, rollback, and the
  historical rollback baseline;
- dynamic protection: read the lifecycle state file and merge the collection names
  referenced by the active / previous / staging pointers into the forbidden set;
- refuse to run (exit code 2) when the allowlist and the forbidden set intersect, or when
  the state file cannot be read; nothing is deleted;
- only delete names that actually exist; before deleting, export the name / id / count /
  metadata of every collection to the before JSON, and after ``--execute`` finishes,
  export the after JSON;
- refuse to run (exit code 2) when the output files this run would write (before; plus
  after under ``--execute``) already exist, so the A4 audit record is never overwritten;
  the check happens before connecting to Chroma. Pass ``--overwrite`` explicitly to
  replace them.

Usage:

    python scripts/prune_chroma_collections.py            # dry-run: only export before and list the names that would be deleted
    python scripts/prune_chroma_collections.py --execute  # really delete, then export after

Exit codes: 0 = success (dry-run, or deletion completed); 1 = a Chroma operation failed;
2 = refused to run (output file already exists and ``--overwrite`` was not passed, a
protection conflict, or the state file is unreadable); in that case Chroma was never
connected and nothing was changed.
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

# A4.2 confirmed orphan collections: staging leftovers, the refmeta experiment, and two structure
# experiments. Changing them requires a code change.
ALLOWED_TO_DELETE: tuple[str, ...] = (
    "ef_research_bge_m3_hybrid_c512_v2_staging",
    "ef_research_bge_m3_c512_o64_refmeta_v1",
    "ef_research_structure_v1",
    "ef_research_structure_v1_elemaware",
)
# The current active index (v2), the previous-generation hybrid index (v1), and the frozen rollback
# baseline (c512_o64_v1); deleting any of them is never allowed.
FORBIDDEN_TO_DELETE: tuple[str, ...] = (
    "ef_research_bge_m3_hybrid_c512_v2",
    "ef_research_bge_m3_hybrid_c512_v1",
    "ef_research_bge_m3_c512_o64_v1",
)
# Hardcoded assertion: the two hardcoded lists themselves must not intersect (the first line of
# defence when editing the code).
assert not set(ALLOWED_TO_DELETE) & set(FORBIDDEN_TO_DELETE), "allowlist overlaps hardcoded forbidden list"

DEFAULT_BEFORE = ROOT / "artifacts/stage_a/A4_chroma_before.json"
DEFAULT_AFTER = ROOT / "artifacts/stage_a/A4_chroma_after.json"


class PruneRefused(RuntimeError):
    """A pre-execution refusal (exit code 2): no collection was deleted."""


def lifecycle_protected_names(state_path: Path) -> set[str]:
    """Read the collection names referenced by the active / previous / staging pointers in the state file; refuse when unreadable."""

    try:
        state = LifecycleState.model_validate_json(Path(state_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a missing, corrupt, or invalid state must never authorise deletion
        message = " ".join(str(exc).split())
        raise PruneRefused(f"cannot read lifecycle state {state_path}: {type(exc).__name__}: {message}") from exc
    names = {state.active.collection_name}
    for pointer in (state.previous, state.staging):
        if pointer is not None:
            names.add(pointer.collection_name)
    return names


def check_outputs_writable(paths: list[Path], *, overwrite: bool) -> None:
    """Refuse when an output file this run would write already exists and overwriting was not explicitly allowed: before/after are non-regenerable audit records."""

    if overwrite:
        return
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise PruneRefused(f"output file(s) already exist: {existing}; pass --overwrite to replace them")


def forbidden_names(state_path: Path) -> set[str]:
    """Forbidden set = hardcoded forbidden list union the names referenced by the state file pointers."""

    return set(FORBIDDEN_TO_DELETE) | lifecycle_protected_names(state_path)


def check_allowlist(allowed: tuple[str, ...], forbidden: set[str]) -> None:
    """Refuse when the allowlist intersects the forbidden set: better to delete nothing than to accidentally delete a production or rollback index."""

    overlap = sorted(set(allowed) & forbidden)
    if overlap:
        raise PruneRefused(f"allowlist contains protected collections: {overlap}; refusing to delete anything")


def export_collections(client: Any) -> list[dict[str, Any]]:
    """Export name / id / count / metadata of every collection, sorted by name to keep the output deterministic."""

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
    """Consistent with the project's write_json: UTF-8, no CJK escaping, sorted keys."""

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
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one cleanup (or dry-run) and return the summary; raise ``PruneRefused`` without deleting anything on a protection conflict."""

    # Run every protection check before connecting to Chroma, so the refusal path never opens the
    # store.
    # The output-overwrite check covers only the files this run writes: dry-run writes before only,
    # --execute also writes after.
    check_outputs_writable([before_output, after_output] if execute else [before_output], overwrite=overwrite)
    forbidden = forbidden_names(state_path)
    check_allowlist(ALLOWED_TO_DELETE, forbidden)

    import chromadb

    client = chromadb.PersistentClient(path=str(db_path))
    before = export_collections(client)
    _write_json(before_output, _snapshot(db_path, before, "before"))
    existing = {row["name"] for row in before}
    # Only delete names that actually exist; allowlist entries that are absent are just recorded,
    # not treated as errors.
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
        # Re-confirm right before deleting: this defends against a future change to the filtering
        # logic above.
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的 before/after 输出文件（默认拒绝，退出码 2）",
    )
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
            overwrite=args.overwrite,
        )
    except PruneRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - Chroma failures are reported uniformly as one line
        print(f"FAILED: {type(exc).__name__}: {' '.join(str(exc).split())}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
