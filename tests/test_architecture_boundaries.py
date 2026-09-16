"""保护 Runtime、Evaluation 与 Experiment 的单向架构边界。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "src/evidenceflow"
RETIRED_RUNTIME_MODULES = {
    "evidenceflow.translation",
    "evidenceflow.retrieval.bm25",
    "evidenceflow.retrieval.fusion",
    "evidenceflow.retrieval.hybrid_fusion",
    "evidenceflow.retrieval.multilingual",
}


def _imports(path: Path) -> set[str]:
    """读取单个模块的静态 import，不执行可能加载模型的生产代码。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_runtime_modules_do_not_import_retired_retrieval() -> None:
    """正式源码不得重新引入已退休的 Translation、BM25、RRF 或多语链路。"""

    violations: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        if "eval" in path.relative_to(RUNTIME_ROOT).parts:
            continue
        forbidden = {
            module
            for module in _imports(path)
            if any(
                module == retired or module.startswith(retired + ".")
                for retired in RETIRED_RUNTIME_MODULES
            )
        }
        if forbidden:
            violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {sorted(forbidden)}")

    assert violations == []


def test_tests_do_not_import_experiments_debug() -> None:
    """测试只依赖 src；不得直接 import experiments/debug 下的开发者脚本。

    Phase 11C Step 1 已把 debug 层的纯格式化函数上提到 src/evidenceflow/inspection.py，
    这条断言用于防止该依赖再次回流（tests → experiments 属禁止方向）。
    """

    violations: list[str] = []
    for path in sorted((PROJECT_ROOT / "tests").rglob("*.py")):
        for module in _imports(path):
            if module.startswith("experiments.debug"):
                violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {module}")

    assert violations == []


def test_production_components_do_not_define_secondary_settings_factories() -> None:
    """Settings 到生产依赖的映射必须集中在 runtime.py。"""

    definitions: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        if path.name == "runtime.py" or "eval" in path.relative_to(RUNTIME_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "from_settings":
                definitions.append(str(path.relative_to(PROJECT_ROOT)))

    assert definitions == []
