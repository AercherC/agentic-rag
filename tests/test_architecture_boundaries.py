"""Protect the unidirectional architecture boundaries between Runtime, Evaluation and Experiment."""

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
    """Read a single module's static imports without executing production code that may load models."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_runtime_modules_do_not_import_retired_retrieval() -> None:
    """Production source must not reintroduce the retired Translation, BM25, RRF or multilingual chain."""

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
    """Tests depend on src only; they must not import developer scripts under experiments/debug.

    Phase 11C Step 1 already promoted the debug layer's pure formatting functions into
    src/evidenceflow/inspection.py, and this assertion prevents that dependency from flowing back
    (tests -> experiments is a forbidden direction).
    """

    violations: list[str] = []
    for path in sorted((PROJECT_ROOT / "tests").rglob("*.py")):
        for module in _imports(path):
            if module.startswith("experiments.debug"):
                violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {module}")

    assert violations == []


def test_production_components_do_not_define_secondary_settings_factories() -> None:
    """The mapping from Settings to production dependencies must be concentrated in runtime.py."""

    definitions: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        if path.name == "runtime.py" or "eval" in path.relative_to(RUNTIME_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "from_settings":
                definitions.append(str(path.relative_to(PROJECT_ROOT)))

    assert definitions == []


def test_runtime_modules_do_not_import_eval() -> None:
    """Production source outside the eval layer (including tracing.py, domain/trace.py, runtime,
    retrieval, generation, rag) must not statically import ``evidenceflow.eval``: evaluation depends on
    the live chain, and a reverse dependency would form a cycle and leak evaluation logic into tracing.
    """

    violations: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        if "eval" in path.relative_to(RUNTIME_ROOT).parts:
            continue
        forbidden = {
            module
            for module in _imports(path)
            if module == "evidenceflow.eval" or module.startswith("evidenceflow.eval.")
        }
        if forbidden:
            violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {sorted(forbidden)}")

    # Explicitly confirm that the two modules added in B2 fall within the scanned range.
    scanned = {path.relative_to(RUNTIME_ROOT).as_posix() for path in RUNTIME_ROOT.rglob("*.py")}
    assert {"tracing.py", "domain/trace.py"} <= scanned
    assert violations == []
