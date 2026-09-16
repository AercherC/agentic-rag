#!/usr/bin/env python
"""已冻结的 Phase 11B unused-import 历史脚本。

规则：
  - 只扫描 src/evidenceflow 下非 __init__ 的模块；__init__.py 的 import 多为 re-export，
    单独按 __all__ 是否包含该符号判断。
  - 名称使用统计包含注解、默认参数、字符串以外的一切 ast.Name / ast.Attribute 用法，
    以及 `from x import y as z` 的别名。
  - 仅输出候选，不自动修改源码。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

SRC = Path("src/evidenceflow")
OUT = Path("evals/evaluation_v3/phase11b_cleanup_v1")


class NameCollector(ast.NodeVisitor):
    """收集模块体内实际被使用的名称（含注解与默认参数）。"""

    def __init__(self) -> None:
        self.used: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.used.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # a.b.c 只统计最左侧根名
        cur = node
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name):
            self.used.add(cur.id)
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        # from x import y as z：z 计入使用；y 只是来源名
        self.used.add(node.asname or node.name)
        self.generic_visit(node)


def analyse(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collector = NameCollector()
    body = list(tree.body)
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            collector.visit(node)  # 登记别名，避免 import 自身被误判
        else:
            collector.visit(node)

    declared_all: set[str] = set()
    for node in body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        declared_all.update(ast.literal_eval(node.value))
                    except ValueError:
                        pass

    findings: list[dict] = []
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound == "*":
                    continue
                if bound in collector.used:
                    continue
                if path.name == "__init__.py" and bound in declared_all:
                    continue  # 显式 re-export，属于兼容层职责
                findings.append(
                    {
                        "path": str(path).replace("\\", "/"),
                        "line": node.lineno,
                        "statement": ast.unparse(node),
                        "unused_name": bound,
                        "is_init_reexport": path.name == "__init__.py",
                    }
                )
    return findings


def main() -> int:
    results: list[dict] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        results.extend(analyse(path))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "unused_imports.json").write_text(
        json.dumps(
            {"artifact_type": "phase11b_unused_imports", "findings": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"unused import candidates: {len(results)}")
    for item in results:
        print(f"  {item['path']}:{item['line']}  {item['unused_name']}  <- {item['statement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
