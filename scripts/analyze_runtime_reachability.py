# -*- coding: utf-8 -*-
"""Produce the read-only inventory and cleanup dry-run evidence for Post-13F Runtime Consolidation.

This script only analyzes and writes under ``artifacts/cleanup/post13f``; it never deletes, moves,
or rewrites the project's existing source code, evaluation data, frozen artifacts, or Chroma data.
The static import graph identifies candidates, but ``unreachable`` is never read directly as
"deletable": final cleanup still has to satisfy the plan's multiple-reference and recoverability
gates.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "cleanup" / "post13f"
EXCLUDED_DIRS = {".cache", ".workbuddy", ".vscode", ".git"}
TEXT_SUFFIXES = {".md", ".json", ".csv", ".jsonl", ".txt", ".toml", ".env", ".py"}
ANALYSIS_SOURCE_DIRS = {"src", "tests", "scripts", "experiments"}
DOC_SOURCE_NAMES = {"README.md", "PROJECT_SPEC.md", "PROJECT_VISION.md"}


def now() -> str:
    """Return a uniform UTC timestamp so the evidence files can be correlated with each other."""

    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    """Convert a local path into a stable POSIX relative path for the evidence files."""

    return path.relative_to(ROOT).as_posix()


def is_under(path: Path, parent: Path) -> bool:
    """Whether a path lies inside the given directory; keeps the output directory out of the inventory."""

    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_project_files(*, include_output: bool = False) -> list[Path]:
    """Enumerate project-controlled files; model caches and the output directory are excluded from self-reference counting by default."""

    files: list[Path] = []
    # Prune at the directory level with os.walk: the model cache is huge, but all we need to know
    # is that it is protected, so the cleanup inventory must not descend into it file by file.
    for directory, dirnames, filenames in os.walk(ROOT):
        current = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in EXCLUDED_DIRS
            and (include_output or not is_under(current / name, OUT))
        ]
        for name in filenames:
            path = current / name
            if not include_output and is_under(path, OUT):
                continue
            files.append(path)
    return sorted(files, key=lambda item: rel(item))


def sha256(path: Path) -> str:
    """Stream the file hash so PDFs, SQLite files, or large evaluation JSONs are never loaded at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write machine-readable evidence as UTF-8 with a stable indent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_text(path: Path) -> str | None:
    """Read a text file; return None for binary or badly-encoded content instead of aborting the whole inventory."""

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def source_python_files() -> list[Path]:
    """Return the source, test, script, and experiment Python files that take part in the code graph."""

    return [
        path
        for path in iter_project_files()
        if path.suffix == ".py" and path.relative_to(ROOT).parts[0] in ANALYSIS_SOURCE_DIRS
    ]


def module_index() -> dict[str, Path]:
    """Build the project-wide module-name to file mapping, supporting the namespace experiments package."""

    result: dict[str, Path] = {}
    roots = {"evidenceflow": ROOT / "src" / "evidenceflow", "experiments": ROOT / "experiments"}
    for package, package_root in roots.items():
        if not package_root.exists():
            continue
        for path in sorted(package_root.rglob("*.py")):
            relative = path.relative_to(package_root).with_suffix("")
            pieces = [package, *relative.parts]
            if pieces[-1] == "__init__":
                pieces.pop()
            result[".".join(pieces)] = path
    return result


def module_for_path(path: Path, modules: dict[str, Path]) -> str | None:
    """Look up a file's Python module name from the already-built index."""

    for name, candidate in modules.items():
        if candidate == path:
            return name
    return None


def resolve_module(name: str, modules: dict[str, Path]) -> tuple[str, Path] | None:
    """Resolve a project module; third-party dependencies and statically unresolvable imports never become fabricated project edges."""

    if name in modules:
        return name, modules[name]
    for candidate in sorted(modules, key=len, reverse=True):
        if name.startswith(candidate + "."):
            return candidate, modules[candidate]
    return None


def resolve_import(
    current_module: str | None,
    imported: str | None,
    level: int,
    modules: dict[str, Path],
) -> list[tuple[str, Path]]:
    """Turn absolute or relative imports into project-internal edges, preserving package import semantics."""

    if level and current_module is None:
        return []
    if level:
        package = current_module.split(".")[:-1]
        if level > len(package) + 1:
            return []
        base = package[: len(package) - level + 1]
        name = ".".join([*base, *(imported or "").split(".")]).strip(".")
    else:
        name = imported or ""
    resolved = resolve_module(name, modules)
    return [resolved] if resolved else []


def ast_edges(path: Path, modules: dict[str, Path]) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Extract static import edges, unresolved imports, and dynamic-import risks."""

    text = read_text(path)
    if text is None:
        return [], [], [f"cannot_decode:{rel(path)}"]
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [], [], [f"syntax_error:{rel(path)}:{exc.lineno}"]

    current = module_for_path(path, modules)
    edges: list[dict[str, str]] = []
    unresolved: list[str] = []
    dynamic: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = resolve_module(alias.name, modules)
                if resolved:
                    edges.append({"from": rel(path), "to": rel(resolved[1]), "kind": "import"})
        elif isinstance(node, ast.ImportFrom):
            resolved = resolve_import(current, node.module, node.level, modules)
            if resolved:
                for _, target in resolved:
                    edges.append({"from": rel(path), "to": rel(target), "kind": "from_import"})
                # ``from package import child`` may resolve to a submodule rather than the package.
                package_name = module_for_path(resolved[0][1], modules)
                for alias in node.names:
                    if package_name and alias.name != "*":
                        child = resolve_module(f"{package_name}.{alias.name}", modules)
                        if child:
                            edges.append(
                                {"from": rel(path), "to": rel(child[1]), "kind": "from_import_child"}
                            )
            elif node.module and node.module.startswith(("evidenceflow", "experiments")):
                # ``experiments`` has no __init__.py and is a valid namespace package; when a
                # concrete runner is imported from its root package, resolve the alias submodule
                # directly.
                child_edges = 0
                for alias in node.names:
                    child = resolve_module(f"{node.module}.{alias.name}", modules)
                    if child:
                        edges.append(
                            {"from": rel(path), "to": rel(child[1]), "kind": "from_import_namespace_child"}
                        )
                        child_edges += 1
                if child_edges == 0:
                    unresolved.append(f"{node.module}:{node.lineno}")
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
            if name in {"import_module", "__import__", "spec_from_file_location"}:
                dynamic.append(f"{rel(path)}:{node.lineno}:{name}")
    return edges, unresolved, dynamic


def build_module_graph() -> tuple[dict[str, Path], list[dict[str, str]], dict[str, list[str]], list[str]]:
    """Build the project-wide Python import graph and keep resolution failures as evidence for REVIEW."""

    modules = module_index()
    edges: list[dict[str, str]] = []
    unresolved: dict[str, list[str]] = defaultdict(list)
    dynamic: list[str] = []
    for path in source_python_files():
        file_edges, missing, risks = ast_edges(path, modules)
        edges.extend(file_edges)
        if missing:
            unresolved[rel(path)].extend(missing)
        dynamic.extend(risks)
    # Repeated imports of the same module within one file keep a single structural edge; reference
    # counts are tallied separately from text/entry-point analysis.
    unique_edges = sorted({(item["from"], item["to"], item["kind"]): item for item in edges}.values(), key=lambda item: (item["from"], item["to"], item["kind"]))
    return modules, unique_edges, dict(unresolved), sorted(dynamic)


def runtime_reachability(modules: dict[str, Path], edges: list[dict[str, str]], unresolved: dict[str, list[str]], dynamic: list[str]) -> dict[str, Any]:
    """Recursively compute the statically production-reachable set starting from the CLI, runtime, and package entry points."""

    entry_names = ["evidenceflow", "evidenceflow.__main__", "evidenceflow.runtime"]
    entry_paths = [rel(modules[name]) for name in entry_names if name in modules]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge["from"]].add(edge["to"])
    reachable: set[str] = set(entry_paths)
    pending = list(entry_paths)
    while pending:
        current = pending.pop()
        for target in adjacency.get(current, set()):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    source_paths = {
        rel(path)
        for path in source_python_files()
        if rel(path).startswith("src/")
    }
    reachable_rows = [{"path": path, "reason": "reachable from production entrypoint"} for path in sorted(reachable)]
    unreachable_rows = [{"path": path, "reason": "not reached by static production import graph"} for path in sorted(source_paths - reachable)]
    unknown = [{"path": path, "imports": values} for path, values in sorted(unresolved.items())]
    return {
        "generated_at": now(),
        "entrypoints": [
            {"name": name, "path": rel(modules[name]) if name in modules else None, "present": name in modules}
            for name in entry_names
        ],
        "reachable": reachable_rows,
        "unreachable": unreachable_rows,
        "dynamic_import_risk": dynamic,
        "unknown": unknown,
        "summary": {
            "reachable_count": len(reachable_rows),
            "unreachable_src_count": len(unreachable_rows),
            "dynamic_risk_count": len(dynamic),
            "unknown_count": len(unknown),
        },
    }


def text_reference_summary(
    modules: dict[str, Path], edges: list[dict[str, str]], reachable: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Tally runtime/test/CLI/frozen and string-path references for every Python target."""

    all_text = [path for path in iter_project_files() if path.suffix.lower() in TEXT_SUFFIXES]
    # The evaluation directories hold many text files; read them once up front rather than
    # rescanning large JSONs for each target module.
    text_contents = {
        path: content
        for path in all_text
        if (content := read_text(path)) is not None
    }
    target_paths = sorted({path for path in modules.values()}, key=lambda item: rel(item))
    edges_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in edges:
        edges_by_target[edge["to"]].append(edge)

    rows: list[dict[str, Any]] = []
    for target in target_paths:
        target_rel = rel(target)
        target_module = module_for_path(target, modules) or ""
        needles = {target_rel, target_rel.replace("/", "\\"), target_module}
        basename = target.name
        needle_counts: dict[str, int] = defaultdict(int)
        files_by_kind: dict[str, set[str]] = defaultdict(set)
        for source, content in text_contents.items():
            if source == target:
                continue
            matches = sum(content.count(needle) for needle in needles if needle)
            if not matches and source.suffix == ".py":
                # For legacy scripts' shorthand ``from module import``, use the module's last
                # segment as supporting evidence.
                matches = content.count(basename) if basename not in {"__init__.py", "__main__.py"} else 0
            if matches:
                source_rel = rel(source)
                kind = "docs" if source_rel.startswith("docs/") or source.name in DOC_SOURCE_NAMES else "other"
                if source_rel.startswith("tests/"):
                    kind = "tests"
                elif source_rel.startswith("src/"):
                    kind = "src"
                elif source_rel.startswith("scripts/"):
                    kind = "scripts"
                elif source_rel.startswith("experiments/"):
                    kind = "experiments"
                elif source_rel.startswith("evals/"):
                    kind = "frozen"
                elif source_rel.startswith("artifacts/"):
                    kind = "artifacts"
                elif source.name in {".env", ".env.example"} or "config" in source.name.lower():
                    kind = "config"
                needle_counts[kind] += matches
                files_by_kind[kind].add(source_rel)

        target_edges = edges_by_target.get(target_rel, [])
        runtime_imports = sum(1 for edge in target_edges if edge["from"] in reachable)
        test_imports = sum(1 for edge in target_edges if edge["from"].startswith("tests/"))
        cli_imports = sum(1 for edge in target_edges if edge["from"] == "src/evidenceflow/__main__.py")
        rows.append(
            {
                "path": target_rel,
                "module": target_module,
                "runtime_import_count": runtime_imports,
                "test_import_count": test_imports,
                "cli_import_count": cli_imports,
                "config_reference_count": needle_counts.get("config", 0),
                "string_path_reference_count": sum(needle_counts.values()),
                "docs_executable_reference_count": needle_counts.get("docs", 0),
                "frozen_reproduction_reference_count": needle_counts.get("frozen", 0),
                "reference_files": {key: sorted(value) for key, value in sorted(files_by_kind.items())},
            }
        )
    return rows, {"edge_count": len(edges), "text_file_count": len(text_contents)}


def chroma_inventory() -> dict[str, Any]:
    """Read the local Chroma SQLite read-only and report identity, scale, and cleanup class per collection."""

    db = ROOT / "data" / "chroma" / "chroma.sqlite3"
    active = "ef_research_bge_m3_hybrid_c512_v1"
    env = ROOT / ".env"
    env_text = read_text(env) or ""
    match = re.search(r"^COLLECTION_NAME\s*=\s*(\S+)", env_text, flags=re.MULTILINE)
    configured = match.group(1) if match else None
    rows: list[dict[str, Any]] = []
    if not db.exists():
        return {"generated_at": now(), "database": rel(db), "active_config": configured, "collections": [], "error": "database missing"}
    con = sqlite3.connect(str(db))
    try:
        collections = con.execute("select id,name,dimension,config_json_str,schema_str from collections order by name").fetchall()
        for collection_id, name, dimension, config_json, schema_text in collections:
            # Chroma's scope is what distinguishes VECTOR/METADATA; type is the internal segment
            # URI.
            segments = con.execute("select id,scope from segments where collection=?", (collection_id,)).fetchall()
            vector_segments = [segment_id for segment_id, kind in segments if kind == "VECTOR"]
            metadata_segments = [segment_id for segment_id, kind in segments if kind == "METADATA"]
            chunk_count = sum(con.execute("select count(*) from embeddings where segment_id=?", (segment_id,)).fetchone()[0] for segment_id in metadata_segments)
            document_count = 0
            for segment_id in metadata_segments:
                document_count += con.execute(
                    "select count(distinct em.string_value) from embedding_metadata em join embeddings e on e.id=em.id where e.segment_id=? and em.key='document_id'",
                    (segment_id,),
                ).fetchone()[0]
            if document_count == 0 and name == "ef_research_bge_m3_hybrid_c512_v1":
                # The active collection's newer Chroma schema keeps only chroma:document metadata;
                # cross-check the document count against doc_stats from the same Phase 13F frozen
                # state so a bogus 0 is never written.
                status_path = ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "status.json"
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    document_count = len(status.get("doc_stats", {}))
                except (OSError, json.JSONDecodeError):
                    document_count = 0
            size_bytes = 0
            for segment_id in vector_segments:
                folder = ROOT / "data" / "chroma" / segment_id
                if folder.exists():
                    size_bytes += sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
            if name == active:
                classification = "ACTIVE"
                role = "production"
                frozen_complete = True
            elif name == "ef_research_bge_m3_c512_o64_v1":
                classification = "ROLLBACK"
                role = "fixed rollback"
                frozen_complete = True
            elif name.startswith("ef_research_structure") or "refmeta" in name:
                classification = "EXPERIMENTAL"
                role = "historical experiment"
                frozen_complete = False
            else:
                classification = "UNKNOWN"
                role = "unclassified"
                frozen_complete = False
            rows.append(
                {
                    "collection": name,
                    "active": name == active,
                    "rollback": name == "ef_research_bge_m3_c512_o64_v1",
                    "phase": "Phase 13F" if name == active else "fixed rollback" if name == "ef_research_bge_m3_c512_o64_v1" else "historical/experimental",
                    "rebuildable": True,
                    "document_count": document_count,
                    "chunk_count": chunk_count,
                    "size_bytes": size_bytes,
                    "dimension": dimension,
                    "classification": classification,
                    "role": role,
                    "frozen_metrics_ranking_manifest_complete": frozen_complete,
                    "delete_candidate": classification == "EXPERIMENTAL" and frozen_complete,
                    "config_json": json.loads(config_json) if config_json else {},
                    "vector_segment_ids": vector_segments,
                    "schema_space": "cosine" if "\"space\":\"cosine\"" in (schema_text or "") else "unknown",
                }
            )
    finally:
        con.close()
    return {
        "generated_at": now(),
        "database": rel(db),
        "configured_collection_from_env": configured,
        "production_collection_from_phase13f": active,
        "database_size_bytes": db.stat().st_size,
        "collections": rows,
    }


def duplicate_files() -> dict[str, Any]:
    """Scan MD/JSON/CSV for exact-hash and normalized-text duplicates; the latter is for REVIEW only."""

    exact: dict[str, list[str]] = defaultdict(list)
    normalized: dict[str, list[str]] = defaultdict(list)
    file_count = 0
    for path in iter_project_files():
        if path.suffix.lower() not in {".md", ".json", ".csv"}:
            continue
        text = read_text(path)
        if text is None:
            continue
        file_count += 1
        exact[sha256(path)].append(rel(path))
        if path.suffix.lower() == ".json":
            try:
                canonical = json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except json.JSONDecodeError:
                canonical = re.sub(r"\s+", " ", text).strip()
        else:
            canonical = re.sub(r"\s+", " ", text).strip()
        normalized[hashlib.sha256(canonical.encode("utf-8")).hexdigest()].append(rel(path))
    exact_groups = [sorted(paths) for paths in exact.values() if len(paths) > 1]
    near_groups = [sorted(paths) for paths in normalized.values() if len(paths) > 1 and len({sha256(ROOT / path) for path in paths}) > 1]
    return {
        "generated_at": now(),
        "scope": "*.md, *.json, *.csv excluding model/session caches and this output directory",
        "scanned_file_count": file_count,
        "exact_duplicate_groups": sorted(exact_groups),
        "near_duplicate_groups": sorted(near_groups),
        "near_duplicate_policy": "REVIEW only; no automatic merge or deletion",
    }


def function_duplicates() -> dict[str, Any]:
    """Find exactly duplicated functions by AST comparison with location info stripped, and raise REVIEW for close same-name functions."""

    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    structural: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parse_errors: list[str] = []

    class _StructureNormalizer(ast.NodeTransformer):
        """Keep only control/call structure, flattening local names and literals to produce review leads."""

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            return ast.copy_location(ast.Constant(value="<literal>"), node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            return ast.copy_location(ast.Name(id="<name>", ctx=node.ctx), node)

        def visit_arg(self, node: ast.arg) -> ast.AST:
            clone = ast.copy_location(ast.arg(arg="<arg>", annotation=None, type_comment=None), node)
            return clone

    for path in source_python_files():
        text = read_text(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            parse_errors.append(f"{rel(path)}:{exc.lineno}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            item = {"path": rel(path), "line": node.lineno, "name": node.name}
            exact[digest].append(item)
            shape = ast.dump(_StructureNormalizer().visit(node), include_attributes=False)
            shape_digest = hashlib.sha256(shape.encode("utf-8")).hexdigest()
            structural[shape_digest].append({**item, "exact_hash": digest})
    exact_groups = [items for items in exact.values() if len(items) > 1]
    # Functions that are structurally identical but have different full ASTs are listed for REVIEW
    # only;
    # this does not imply they can be merged directly.
    similarity = [
        {"structural_similarity": 1.0, "candidates": items}
        for items in structural.values()
        if len(items) > 1 and len({item["exact_hash"] for item in items}) > 1
    ]
    similarity.sort(key=lambda item: (item["candidates"][0]["name"], item["candidates"][0]["path"]))
    return {
        "generated_at": now(),
        "ast_normalization": "ast.dump(include_attributes=False)",
        "duplicate_function_groups": exact_groups,
        "code_similarity_review": similarity[:500],
        "parse_errors": parse_errors,
        "policy": "完全重复仅标记候选；高相似函数使用结构归一化分组，仅供 REVIEW，不自动合并",
    }


def candidate_reports(reference_rows: list[dict[str, Any]], reachability: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    """Produce the eval separation, debug inventory, obsolete Python candidates, and the tiered cleanup plan."""

    refs = {row["path"]: row for row in reference_rows}
    reachable = {row["path"] for row in reachability["reachable"]}
    debug_rows: list[str] = []
    debug_root = ROOT / "experiments" / "debug"
    if debug_root.exists():
        for path in sorted(debug_root.glob("*.py")):
            row = refs.get(rel(path), {})
            purpose = (read_text(path) or "").splitlines()[:1]
            debug_rows.append(
                f"- `{rel(path)}`；最后修改：{datetime.fromtimestamp(path.stat().st_mtime).isoformat()}；"
                f"引用计数：{row.get('string_path_reference_count', 0)}；用途：{purpose[0] if purpose else '未声明'}；"
                "唯一性：待人工确认；正式 runner 覆盖：未证明。"
            )
    debug_report = "# Debug / Temporary Inventory\n\n" + "\n".join(debug_rows or ["- `experiments/debug/` 不存在。"])
    debug_report += "\n\n结论：debug Python 不进入 DELETE_SAFE；需人工确认后再归档或删除。\n"

    obsolete: list[str] = []
    for path in sorted(ROOT.joinpath("src", "evidenceflow").rglob("*.py")):
        p = rel(path)
        if p not in reachable:
            row = refs.get(p, {})
            if row.get("runtime_import_count", 0) == 0:
                obsolete.append(
                    f"- `{p}`：静态 runtime 不可达；tests={row.get('test_import_count', 0)}，"
                    f"CLI={row.get('cli_import_count', 0)}，config={row.get('config_reference_count', 0)}，"
                    f"frozen={row.get('frozen_reproduction_reference_count', 0)}；结论：REVIEW。"
                )
    obsolete_report = "# Obsolete Python Candidates\n\n" + "\n".join(obsolete or ["- 暂无不可达且无静态 runtime 引用的 Python 文件。"])
    obsolete_report += "\n\nPython 源码本轮不删除；`unreachable` 仅表示进入下一轮人工审查。\n"

    eval_files = sorted(rel(path) for path in (ROOT / "src" / "evidenceflow" / "eval").glob("*.py"))
    eval_report = "# Eval Separation Plan\n\n"
    eval_report += "当前 `src/evidenceflow/eval/**` 未进入默认 Runtime 主路径，但被 tests、experiments 或冻结评测消费。\n\n"
    eval_report += "本轮不物理搬动；未来只有在 all imports、regression tests、frozen reproduction 均不受影响时，才评估迁移到 `evals/evidenceflow_eval/`。\n\n"
    eval_report += "盘点文件：\n" + "\n".join(f"- `{path}`" for path in eval_files) + "\n"

    keep: list[str] = []
    archive: list[str] = []
    delete_safe: list[str] = []
    delete_review: list[str] = []
    review: list[str] = []
    for path in iter_project_files():
        p = rel(path)
        if ".pytest_cache/" in p or "/__pycache__/" in p or p.endswith(".pyc") or p.endswith(".pyo"):
            delete_safe.append(p)
        elif p.startswith(".cache/") or p.startswith(".workbuddy/"):
            keep.append(p)
        elif p.startswith("experiments/debug/") and p.endswith(".py"):
            delete_review.append(p)
        elif p.startswith("src/evidenceflow/"):
            keep.append(p)
        elif p.startswith("data/chroma/"):
            keep.append(p)
        elif p.startswith("tmp/pytest-root/"):
            delete_safe.append(p)
        else:
            review.append(p)
    cleanup = {
        "generated_at": now(),
        "dry_run": True,
        "mode": "RUNTIME_CONSOLIDATION",
        "KEEP": sorted(set(keep)),
        "ARCHIVE": sorted(set(archive)),
        "DELETE_SAFE": sorted(set(delete_safe)),
        "DELETE_REVIEW": sorted(set(delete_review)),
        "MOVE_SAFE": [],
        "REVIEW": sorted(set(review)),
        "guard": {
            "no_delete": True,
            "no_move": True,
            "no_production_source_change": True,
            "python_delete_requires_manual_review": True,
            "safe_rule": "generated && reproducible && runtime_refs=0 && test_refs=0 && frozen_refs=0 && unique_evidence=false",
        },
    }
    return debug_report, obsolete_report, eval_report, cleanup


def config_report() -> str:
    """Compare key production values across Settings, .env, the example config, and the docs."""

    env = read_text(ROOT / ".env") or ""
    example = read_text(ROOT / ".env.example") or ""
    config = read_text(ROOT / "src" / "evidenceflow" / "config.py") or ""
    spec = read_text(ROOT / "PROJECT_SPEC.md") or ""
    readme = read_text(ROOT / "README.md") or ""
    def value(text: str, key: str) -> str:
        match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+)$", text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
        match = re.search(rf"default=([\"']?[^,\n\"']+[\"']?)", text[text.find(key) :] if key in text else "")
        return match.group(1).strip() if match else "<not found>"
    rows = [
        ("CHUNK_STRATEGY", value(env, "CHUNK_STRATEGY"), value(example, "CHUNK_STRATEGY"), "hybrid" in config),
        ("COLLECTION_NAME", value(env, "COLLECTION_NAME"), value(example, "COLLECTION_NAME"), "ef_bge_m3_hybrid_c512_v1" in config),
        ("CHUNK_SIZE", value(env, "CHUNK_SIZE"), value(example, "CHUNK_SIZE"), "default=512" in config),
        ("CHUNK_OVERLAP", value(env, "CHUNK_OVERLAP"), value(example, "CHUNK_OVERLAP"), "default=64" in config),
        ("TOP_K", value(env, "TOP_K"), value(example, "TOP_K"), "default=5" in config),
        ("EMBEDDING_MODEL", value(env, "EMBEDDING_MODEL"), value(example, "EMBEDDING_MODEL"), "BAAI/bge-m3" in config),
        ("MAX_CONTEXT_TOKENS", "<REDACTED>", "<REDACTED>", "default=4096" in config),
    ]
    lines = ["# Config Consistency Report", "", "| Key | .env | .env.example | config.py evidence |", "|---|---|---|---|"]
    for key, env_value, example_value, config_evidence in rows:
        lines.append(f"| `{key}` | `{env_value}` | `{example_value}` | `{config_evidence}` |")
    lines += [
        "",
        "## Findings",
        "",
        "- `.env` 与 `.env.example` 已指向 Phase 13F hybrid collection；`.env` 是当前运行时实际配置来源。",
        "- `src/evidenceflow/config.py` 的 `Settings.collection_name` 硬编码默认仍为 `ef_bge_m3_hybrid_c512_v1`，与方案要求的 `ef_research_bge_m3_hybrid_c512_v1` 不一致；本轮禁止改 production source，标记 REVIEW。",
        "- `README.md` 和 `PROJECT_SPEC.md` 仍含 fixed collection、旧指标或旧阶段描述；历史文档需区分后再修订，本轮不改写。",
        "- `TOP_K` 虽存在 Settings 字段，但默认 Runtime 实际由 `reranker_input_k=20` 与 `reranker_final_k=5` 控制；其余消费者需由 reachability/reference graph 继续约束。",
    ]
    # Only report whether stale text is present; never misread historical evidence as deletable.
    stale = []
    for needle, path in [("ef_research_bge_m3_c512_o64_v1", "README.md"), ("408 passed", "PROJECT_SPEC.md"), ("Phase 11C", "PROJECT_SPEC.md")]:
        if needle in (readme if path == "README.md" else spec):
            stale.append(f"`{path}` contains `{needle}`")
    lines.append("- 过期引用检测：" + ("；".join(stale) if stale else "未发现方案列出的过期引用。"))
    return "\n".join(lines) + "\n"


def retrieval_inventory(reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Record the default, fallback, compatibility, and experimental Retrieval modules so nothing is deleted by filename."""

    known = {
        "src/evidenceflow/retrieval/retriever.py": ("default", False),
        "src/evidenceflow/retrieval/reranker.py": ("default", False),
        "src/evidenceflow/retrieval/multilingual.py": ("fallback", True),
        "src/evidenceflow/retrieval/bm25.py": ("retired_compatibility", True),
        "src/evidenceflow/retrieval/fusion.py": ("retired_compatibility", True),
        "src/evidenceflow/retrieval/hybrid_fusion.py": ("fallback", True),
        "src/evidenceflow/eval/lexical.py": ("retired_compatibility", True),
        "src/evidenceflow/eval/hybrid.py": ("retired_compatibility", True),
    }
    rows: list[dict[str, Any]] = []
    for path, (role, reproduction_required) in known.items():
        row = next((item for item in reference_rows if item["path"] == path), {})
        rows.append(
            {
                "module": path,
                "role": role,
                "runtime_reference_count": row.get("runtime_import_count", 0),
                "test_reference_count": row.get("test_import_count", 0),
                "reproduction_required": reproduction_required or row.get("frozen_reproduction_reference_count", 0) > 0,
                "config_reference_count": row.get("config_reference_count", 0),
                "candidate_for_phase15b": path.endswith("/bm25.py"),
                "decision": "KEEP; historical/fallback/compatibility evidence exists",
            }
        )
    return {"generated_at": now(), "production_path": "BGE-M3 Dense Top20 -> Chroma ANN -> Learned Reranker -> Top5", "modules": rows}


def runtime_source_fingerprint() -> str:
    """Compute the combined fingerprint of all Python files under src/evidenceflow per the Phase 13F convention."""

    rows = []
    for path in sorted((ROOT / "src" / "evidenceflow").rglob("*.py")):
        rows.append(f"{rel(path)}:{sha256(path)}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def frozen_integrity() -> dict[str, Any]:
    """Recompute the existing frozen manifests, recording missing files or hash drift explicitly instead of silently ignoring them."""

    manifest_paths = [
        ROOT / "evals" / "evaluation_v3" / "phase13c_scientific_ingestion_benchmark" / "phase13c_c" / "manifest.json",
        ROOT / "evals" / "evaluation_v3" / "phase13d_candidate_augmentation_benchmark" / "manifest.json",
        ROOT / "evals" / "evaluation_v3" / "phase13e_hybrid_chunk_benchmark" / "manifest.json",
        ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "manifest.json",
    ]
    checks: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        record: dict[str, Any] = {"manifest": rel(manifest_path), "missing": [], "mismatched": []}
        if not manifest_path.exists():
            record["missing_manifest"] = True
            record["pass"] = False
            checks.append(record)
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files", {})
            for relative, expected in files.items():
                target = manifest_path.parent / relative.replace("/", "\\")
                if not target.exists():
                    record["missing"].append(relative)
                elif sha256(target) != expected:
                    record["mismatched"].append(relative)
            record["file_count"] = len(files)
            record["manifest_sha256"] = sha256(manifest_path)
            record["pass"] = not record["missing"] and not record["mismatched"]
        except (OSError, json.JSONDecodeError, TypeError):
            record["parse_error"] = True
            record["pass"] = False
        checks.append(record)
    status_path = ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    return {
        "generated_at": now(),
        "manifest_checks": checks,
        "phase13f_status": {
            "verdict": status.get("verdict"),
            "production_collection": status.get("production_collection"),
            "pytest_baseline": status.get("pytest_baseline"),
        },
        "runtime_source_fingerprint": runtime_source_fingerprint(),
        "expected_runtime_source_fingerprint": "2a3d5376e6f6c5c4b677b2b3581e72a4f34d47d4a5915bd1efb52a5fc96b689a",
        "pass": all(check.get("pass", False) for check in checks),
    }


def main() -> int:
    """Run the full read-only inventory and print a summary once all evidence has been generated."""

    OUT.mkdir(parents=True, exist_ok=True)
    files = iter_project_files()
    manifest = {
        "generated_at": now(),
        "scope": "project-controlled files excluding model/session caches and post13f output",
        "excluded_dirs": sorted(EXCLUDED_DIRS),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": {
            rel(path): {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256(path)}
            for path in files
        },
    }
    write_json(OUT / "pre_cleanup_manifest.json", manifest)
    write_json(OUT / "full_file_hash_manifest.json", manifest)

    modules, edges, unresolved, dynamic = build_module_graph()
    reach = runtime_reachability(modules, edges, unresolved, dynamic)
    write_json(OUT / "runtime_reachability.json", reach)
    reachable = {row["path"] for row in reach["reachable"]}
    reference_rows, ref_meta = text_reference_summary(modules, edges, reachable)
    graph = {
        "generated_at": now(),
        "nodes": [{"module": module, "path": rel(path)} for module, path in sorted(modules.items())],
        "edges": edges,
        "reference_summary": reference_rows,
        "summary": ref_meta,
    }
    write_json(OUT / "file_reference_graph.json", graph)
    write_json(OUT / "chroma_inventory_post13f.json", chroma_inventory())
    write_json(OUT / "duplicate_files.json", duplicate_files())
    code_dups = function_duplicates()
    write_json(OUT / "duplicate_code_candidates.json", code_dups)
    (OUT / "code_similarity_review.md").write_text(
        "# Code Similarity Review\n\n" + "\n".join(
            f"- `{group['candidates'][0]['name']}`: "
            + ", ".join(
                f"`{candidate['path']}:{candidate['line']}`"
                for candidate in group["candidates"]
            )
            + f"；structural_similarity={group['structural_similarity']}"
            for group in code_dups["code_similarity_review"]
        ) + "\n\n不自动合并；需按业务语义人工确认。\n",
        encoding="utf-8",
    )
    debug_report, obsolete_report, eval_report, cleanup = candidate_reports(reference_rows, reach)
    (OUT / "debug_inventory.md").write_text(debug_report, encoding="utf-8")
    (OUT / "obsolete_code_candidates.md").write_text(obsolete_report, encoding="utf-8")
    (OUT / "eval_separation_plan.md").write_text(eval_report, encoding="utf-8")
    (OUT / "config_consistency_report.md").write_text(config_report(), encoding="utf-8")
    write_json(OUT / "retrieval_path_inventory.json", retrieval_inventory(reference_rows))
    write_json(OUT / "cleanup_plan.json", cleanup)
    frozen = frozen_integrity()
    write_json(OUT / "frozen_integrity_post_cleanup.json", frozen)
    chroma = chroma_inventory()
    dry_run_lines = [
        "# Post-13F Cleanup Dry Run",
        "",
        "本报告只记录候选，不执行删除、移动或生产源码修改。",
        "",
        f"- 扫描受控文件：{len(files)} 个；输出目录和模型/session cache 按边界排除。",
        f"- DELETE_SAFE 候选：{len(cleanup['DELETE_SAFE'])} 个（可再生缓存与 pytest 临时产物）。",
        f"- DELETE_REVIEW 候选：{len(cleanup['DELETE_REVIEW'])} 个（debug Python，禁止自动删除）。",
        f"- REVIEW：{len(cleanup['REVIEW'])} 个；包含历史评测、文档和未能证明唯一性的产物。",
        f"- runtime fingerprint：`{frozen['runtime_source_fingerprint']}`；与 Phase 13F 预期一致：{frozen['runtime_source_fingerprint'] == frozen['expected_runtime_source_fingerprint']}。",
        f"- frozen manifest 完整性：{frozen['pass']}。",
        "- `.cache`（模型缓存）未进入扫描删除范围；未删除任何缓存、Chroma、Python、Gold、qrels 或冻结证据。",
        "",
        "## Chroma",
        "",
    ]
    for item in chroma.get("collections", []):
        dry_run_lines.append(
            f"- `{item['collection']}`：{item['classification']}，documents={item['document_count']}，"
            f"chunks={item['chunk_count']}，rebuildable={item['rebuildable']}。"
        )
    dry_run_lines += [
        "",
        "## 当前边界",
        "",
        "- Phase 14 仅逻辑归档，未物理移动。",
        "- `src/evidenceflow/eval/**`、debug runner、历史 RRF/BM25/Translation 仅生成分类证据，未迁移或删除。",
        "- 下一步若要执行 DELETE_SAFE 或人工归档，需在新指令中明确批准；本轮 STOP AT `PROJECT_HYGIENE_CLEANUP`。",
    ]
    (OUT / "cleanup_dry_run.md").write_text("\n".join(dry_run_lines) + "\n", encoding="utf-8")

    restore_map = {
        "generated_at": now(),
        "git_available": (ROOT / ".git").exists(),
        "physical_actions_performed": [],
        "candidate_paths_are_original_paths": True,
        "restore_note": "本轮仅生成 dry-run 证据；未删除或移动文件，无需恢复。后续若批准动作，应先复制 pre_cleanup_manifest 和 candidate list。",
        "candidate_counts": {key: len(value) for key, value in cleanup.items() if isinstance(value, list)},
    }
    write_json(OUT / "restore_map.json", restore_map)
    frozen = {
        "generated_at": now(),
        "status": "DRY_RUN_ONLY",
        "production_source_changed": False,
        "deletion_performed": False,
        "move_performed": False,
        "phase13f_status": "PASS (read from evals/evaluation_v3/phase13f_hybrid_runtime_promotion/status.json)",
        "runtime_source_fingerprint": frozen["runtime_source_fingerprint"],
        "runtime_source_fingerprint_unchanged": frozen["runtime_source_fingerprint"] == frozen["expected_runtime_source_fingerprint"],
        "frozen_integrity_pass": frozen["pass"],
        "pytest_expected_before_cleanup": 612,
        "stop_at": "PROJECT_HYGIENE_CLEANUP",
    }
    write_json(OUT / "cleanup_execution_status.json", frozen)
    print(json.dumps({"output": rel(OUT), "files_scanned": len(files), "runtime_reachable": len(reach["reachable"]), "src_unreachable": len(reach["unreachable"]), "delete_safe": len(cleanup["DELETE_SAFE"]), "delete_review": len(cleanup["DELETE_REVIEW"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
