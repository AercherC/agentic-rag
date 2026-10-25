# -*- coding: utf-8 -*-
"""Read-only inventory of the Codex runtime-state directory, classified against an exact
allowlist.

This tool does not delete, move or modify CODEX_HOME. By default it protects sessions,
credentials, config, attachments, SQLite and plugins; only files that clearly live in
regenerable directories such as ``cache``, ``tmp`` or ``visualizations`` become
SAFE_CANDIDATE, and even then only a suggested action is reported per the 7/30 day retention
rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "cleanup" / "codex"
SAFE_DIRS = {
    "cache",
    ".tmp",
    "tmp",
    "visualizations",
    "computer-use-turn-ended",
    ".sandbox",
    ".sandbox-bin",
}
PROTECTED_DIRS = {
    "sessions",
    "archived_sessions",
    "attachments",
    "memories",
    "skills",
    "plugins",
}
REVIEW_DIRS = {
    "vendor_imports",
    "computer-use",
    "process_manager",
    "sqlite",
}
PROTECTED_ROOT_FILES = {
    "auth.json",
    "config.toml",
    "history.jsonl",
    "session_index.jsonl",
    ".codex-global-state.json",
}


def resolve_codex_home(value: str | None = None) -> Path:
    """Resolve the directory in the order CODEX_HOME -> USERPROFILE -> Path.home."""

    raw = value or os.environ.get("CODEX_HOME")
    return Path(raw).expanduser().resolve() if raw else (Path.home() / ".codex").resolve()


def _is_sqlite_name(name: str) -> bool:
    """Recognize every SQLite main database plus its WAL/SHM/backup variants."""

    return bool(re.search(r"\.sqlite(?:\d+)?(?:[-.]|$)", name, re.IGNORECASE))


def classify_path(relative_path: str) -> str:
    """Classify a path relative to CODEX_HOME; protect/review takes precedence over the SAFE
    allowlist."""

    rel = relative_path.replace("\\", "/").strip("/")
    parts = rel.split("/") if rel else []
    name = parts[-1] if parts else ""

    # DENY takes precedence: sessions, sensitive state and plugins must never fall into the
    # age-based cleanup allowlist.
    if any(part in PROTECTED_DIRS for part in parts):
        return "C-PROTECT"
    if len(parts) == 1 and name in PROTECTED_ROOT_FILES:
        return "C-PROTECT"
    if _is_sqlite_name(name):
        # The internal logs_*.sqlite files lack a stable deletion contract, so they are marked
        # REVIEW rather than SAFE.
        return "C-REVIEW" if name.lower().startswith("logs_") else "C-PROTECT"

    # These directories may take part in runtime bookkeeping; this round only reports them and
    # never touches them automatically.
    if any(part in REVIEW_DIRS for part in parts):
        return "C-REVIEW"
    if name == "models_cache.json" or name.startswith("logs_"):
        return "C-REVIEW"

    # Only an exact top-level directory allowlist entry can become a SAFE_CANDIDATE.
    if parts and parts[0] in SAFE_DIRS:
        return "C-SAFE-CANDIDATE"
    return "C-REVIEW"


def retention_action(age_days: float, classification: str) -> str:
    """Map SAFE_CANDIDATE to a retention policy; all other classifications are excluded from the
    age-based decision entirely."""

    if classification != "C-SAFE-CANDIDATE":
        return "DO_NOT_AUTO_DELETE"
    if age_days < 7:
        return "KEEP"
    if age_days <= 30:
        return "DELETE_CANDIDATE"
    return "DELETE_SAFE_CANDIDATE"


def sha256_of(path: Path) -> str:
    """Stream the file digest; file content is never written into project artifacts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(root: Path):
    """Walk the directory without following directory links, so the CODEX_HOME boundary is never
    crossed."""

    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if not (Path(dirpath) / name).is_symlink()]
        for name in filenames:
            yield Path(dirpath) / name


def inventory_codex_home(codex_home: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Produce the runtime-state file inventory, its classifications, size rollups and
    safety-snapshot metadata."""

    now = now or datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    for path in iter_files(codex_home) or ():
        relative = path.relative_to(codex_home).as_posix()
        stat = path.stat()
        age_days = max(0.0, now.timestamp() - stat.st_mtime) / 86400.0
        classification = classify_path(relative)
        records.append(
            {
                "path": relative,
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "type": "symlink" if path.is_symlink() else "file",
                "classification": classification,
                "recommended_action": retention_action(age_days, classification),
                "age_days": round(age_days, 3),
            }
        )
    records.sort(key=lambda item: item["path"])
    return {
        "generated_at": now.isoformat(),
        "codex_home": str(codex_home),
        "file_count": len(records),
        "total_bytes": sum(item["size_bytes"] for item in records),
        "files": records,
    }


def _protected_snapshot(codex_home: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    """Produce a pre-cleanup snapshot that contains no sensitive content."""

    files = inventory["files"]
    protected = [item for item in files if item["classification"] == "C-PROTECT"]
    protected_fingerprints = []
    for item in protected:
        path = codex_home / item["path"]
        protected_fingerprints.append(
            f"{item['path']}|{item['size_bytes']}|{item['mtime']}|{sha256_of(path)}"
        )
    protected_hash = hashlib.sha256(
        ("\n".join(protected_fingerprints) + "\n").encode("utf-8")
    ).hexdigest()

    def file_snapshot(name: str) -> dict[str, Any] | None:
        path = codex_home / name
        if not path.is_file():
            return None
        stat = path.stat()
        return {
            "path": name,
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256_of(path),
        }

    sqlite_items = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "mtime": item["mtime"],
        }
        for item in files
        if _is_sqlite_name(Path(item["path"]).name)
    ]
    safe_count = sum(item["classification"] == "C-SAFE-CANDIDATE" for item in files)
    return {
        "generated_at": inventory["generated_at"],
        "codex_home": str(codex_home),
        "sessions_file_count": sum(item["path"].startswith("sessions/") for item in files),
        "archived_sessions_file_count": sum(
            item["path"].startswith("archived_sessions/") for item in files
        ),
        "history": file_snapshot("history.jsonl"),
        "state_sqlite": sqlite_items,
        "auth": file_snapshot("auth.json"),
        "config": file_snapshot("config.toml"),
        "protected_paths_hash": protected_hash,
        "safe_candidate_count": safe_count,
    }


def write_inventory_artifacts(inventory: dict[str, Any], codex_home: Path, output_dir: Path) -> None:
    """Write the five classification artifacts and the cleanup snapshot; auth/config content is
    never written."""

    output_dir.mkdir(parents=True, exist_ok=True)
    files = inventory["files"]
    safe = [item for item in files if item["classification"] == "C-SAFE-CANDIDATE"]
    review = [item for item in files if item["classification"] == "C-REVIEW"]
    protected = [item for item in files if item["classification"] == "C-PROTECT"]
    (output_dir / "codex_state_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for filename, payload in (
        ("codex_safe_candidates.json", safe),
        ("codex_review_items.json", review),
        ("codex_protected_items.json", protected),
    ):
        (output_dir / filename).write_text(
            json.dumps({"codex_home": str(codex_home), "items": payload}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    by_class: dict[str, list[dict[str, int]]] = {}
    for item in files:
        bucket = by_class.setdefault(item["classification"], [{"count": 0, "bytes": 0}])[0]
        bucket["count"] += 1
        bucket["bytes"] += item["size_bytes"]
    lines = [
        "# Codex Runtime State Inventory",
        "",
        f"- CODEX_HOME: `{codex_home}`",
        f"- files: {inventory['file_count']}",
        f"- bytes: {inventory['total_bytes']:,}",
        "",
        "| classification | files | bytes |",
        "|---|---:|---:|",
    ]
    for classification in sorted(by_class):
        bucket = by_class[classification][0]
        lines.append(f"| {classification} | {bucket['count']} | {bucket['bytes']:,} |")
    lines += [
        "",
        "C-PROTECT and C-REVIEW are report-only. SAFE_CANDIDATE is not an automatic deletion authorization.",
    ]
    (output_dir / "codex_size_by_category.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "codex_cleanup_snapshot.json").write_text(
        json.dumps(_protected_snapshot(codex_home, inventory), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    """Run the read-only Codex runtime-state inventory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-home", default=None, help="覆盖 CODEX_HOME 路径")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    codex_home = resolve_codex_home(args.codex_home)
    inventory = inventory_codex_home(codex_home)
    write_inventory_artifacts(inventory, codex_home, args.output_dir)
    counts: dict[str, int] = {}
    for item in inventory["files"]:
        counts[item["classification"]] = counts.get(item["classification"], 0) + 1
    print(json.dumps({"codex_home": str(codex_home), "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
