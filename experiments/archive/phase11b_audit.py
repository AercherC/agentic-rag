#!/usr/bin/env python
"""已冻结的 Phase 11B Production Reachability 历史审计。

方法（不依赖运行时，只读源码）：
  1. 用 ast 解析 src/、tests/、experiments/ 下全部 .py，建立 evidenceflow 内部 import 边；
  2. 从正式入口（__main__ → runtime.build_rag_service）做可达性闭包；
  3. 对每个公共符号统计 src / tests / experiments 三处的外部引用次数；
  4. 按规则给出分类与建议动作。

分类：
  PRODUCTION_REACHABLE  正式入口可达
  FALLBACK_REACHABLE    仅在 fallback / rollback 分支可达
  COMPATIBILITY_ONLY    只被 re-export 兼容层引用
  EXPERIMENT_ONLY       只被 experiments/ 引用
  UNREACHABLE           无任何 import（含 test / experiment）
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
SRC = Path("src/evidenceflow")
TESTS = Path("tests")
EXPERIMENTS = Path("experiments")
OUT = Path("evals/evaluation_v3/phase11b_cleanup_v1")

# 正式 Runtime 入口：CLI 只调用 build_rag_service，不承载业务逻辑。
PRODUCTION_ENTRY_MODULES = {
    "evidenceflow.__main__",
    "evidenceflow.runtime",
}
# Fallback / rollback 分支入口：默认关闭或仅在 RERANKER_ENABLED=false 时装配。
FALLBACK_ENTRY_MODULES = {
    "evidenceflow.retrieval.multilingual",
    "evidenceflow.translation.llm",
    "evidenceflow.retrieval.bm25",
    "evidenceflow.retrieval.hybrid_fusion",
    "evidenceflow.retrieval.fusion",
}
# 仅做 re-export 的兼容层（实现已迁至 retrieval 包）。
COMPATIBILITY_MODULES = {
    "evidenceflow.eval.lexical",
    "evidenceflow.eval.hybrid",
}


def _module_name(path: Path) -> str:
    """把 src 下的文件路径还原为 importable 模块名。"""

    rel = path.relative_to(Path("src")).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _iter_py_files(base: Path):
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(tree: ast.AST) -> set[str]:
    """抽取模块中 import 的全部 evidenceflow.* 目标模块（含 from ... import 的父模块）。"""

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("evidenceflow"):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("evidenceflow"):
                found.add(node.module)
            # from evidenceflow.retrieval import X ⇒ 也可能是子模块
            if node.module:
                for alias in node.names:
                    candidate = f"{node.module}.{alias.name}"
                    found.add(candidate)
    return found


def _exported_symbols(path: Path, tree: ast.AST) -> list[str]:
    """抽取模块对外符号：优先 __all__，否则取顶层 def / class。"""

    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        symbols.extend(ast.literal_eval(node.value))
                    except ValueError:
                        pass
    if symbols:
        return sorted(set(symbols))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
    # 无 __all__ 的 __init__ 只做 re-export，符号意义不大，返回空即可。
    if path.name == "__init__.py":
        return []
    return sorted(set(symbols))


def build_graph() -> tuple[dict[str, set[str]], dict[str, Path], dict[str, ast.Module]]:
    files = _iter_py_files(SRC)
    module_to_path = {_module_name(p): p for p in files}
    edges: dict[str, set[str]] = defaultdict(set)
    trees: dict[str, ast.Module] = {}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        mod = _module_name(path)
        trees[mod] = tree
        for target in _imported_modules(tree):
            # 只保留真实存在的模块（filter 掉 from pkg import Symbol 的误判）
            if target in module_to_path:
                edges[mod].add(target)
    return dict(edges), module_to_path, trees


def reachable_from(edges: dict[str, set[str]], roots: set[str]) -> set[str]:
    seen: set[str] = set()
    stack = [r for r in roots if r in edges or True]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(edges.get(cur, ()))
    return seen


def main() -> int:
    edges, module_to_path, trees = build_graph()
    production = reachable_from(edges, PRODUCTION_ENTRY_MODULES)
    production_plus_fallback = reachable_from(
        edges, PRODUCTION_ENTRY_MODULES | FALLBACK_ENTRY_MODULES
    )

    # 反向引用：谁 import 了这个模块（区分 src / tests / experiments）
    reverse: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"src": set(), "tests": set(), "experiments": set()}
    )
    for base, bucket in ((SRC, "src"), (TESTS, "tests"), (EXPERIMENTS, "experiments")):
        for path in _iter_py_files(base):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if base is SRC:
                importer = _module_name(path)
            else:
                importer = str(path).replace("\\", "/")
            for target in _imported_modules(tree):
                if target in module_to_path:
                    reverse[target][bucket].add(importer)

    # 符号级引用计数：在三处源码文本中统计词边界出现次数（排除定义模块自身）
    corpus: dict[str, list[tuple[str, str]]] = {}
    for bucket, base in (("src", SRC), ("tests", TESTS), ("experiments", EXPERIMENTS)):
        entries = []
        for path in _iter_py_files(base):
            entries.append((str(path).replace("\\", "/"), path.read_text(encoding="utf-8")))
        corpus[bucket] = entries

    def symbol_refs(module: str, symbol: str) -> dict[str, int]:
        own_paths = {str(module_to_path[module]).replace("\\", "/")}
        counts = {}
        for bucket, entries in corpus.items():
            total = 0
            for file_path, text in entries:
                if file_path in own_paths:
                    continue
                total += len(re.findall(rf"\b{re.escape(symbol)}\b", text))
            counts[bucket] = total
        return counts

    records = []
    for module, path in sorted(module_to_path.items()):
        refs = reverse.get(module, {"src": set(), "tests": set(), "experiments": set()})
        src_importers = {m for m in refs["src"] if m != module}
        if module in COMPATIBILITY_MODULES:
            classification = "COMPATIBILITY_ONLY"
        elif module in FALLBACK_ENTRY_MODULES:
            # 静态闭包无法区分条件分支：这些模块只由 runtime.py 在
            # RERANKER_ENABLED=false / hybrid 开关下装配，按人工证据判为 fallback。
            classification = "FALLBACK_REACHABLE"
        elif module in production:
            classification = "PRODUCTION_REACHABLE"
        elif refs["experiments"] and not src_importers:
            classification = "EXPERIMENT_ONLY"
        elif not src_importers and not refs["tests"] and not refs["experiments"]:
            classification = "UNREACHABLE"
        elif path.name == "__init__.py":
            # 包入口按被引用子模块的最强角色归类，避免把 re-export 包误判为死代码。
            classification = "PRODUCTION_REACHABLE" if any(
                child.startswith(module + ".") and child in production
                for child in module_to_path
            ) else "EXPERIMENT_ONLY"
        else:
            classification = "EXPERIMENT_ONLY"

        symbols = []
        for sym in _exported_symbols(path, trees[module]):
            counts = symbol_refs(module, sym)
            symbols.append(
                {
                    "symbol": sym,
                    "refs_src": counts["src"],
                    "refs_tests": counts["tests"],
                    "refs_experiments": counts["experiments"],
                }
            )
        records.append(
            {
                "path": str(path).replace("\\", "/"),
                "module": module,
                "classification": classification,
                "imported_by_src": sorted(src_importers),
                "imported_by_tests": sorted(refs["tests"]),
                "imported_by_experiments": sorted(refs["experiments"]),
                "used_by_runtime": module in production,
                "symbols": symbols,
            }
        )

    manifest = {
        "artifact_type": "phase11b_reachability_manifest",
        "method": "ast static import closure from evidenceflow.__main__ / evidenceflow.runtime",
        "production_entry_modules": sorted(PRODUCTION_ENTRY_MODULES),
        "fallback_entry_modules": sorted(FALLBACK_ENTRY_MODULES),
        "production_reachable_modules": sorted(m for m in production if m in module_to_path),
        "modules": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reachability_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"modules audited: {len(records)}")
    for rec in records:
        print(f"  {rec['classification']:<24} {rec['module']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
