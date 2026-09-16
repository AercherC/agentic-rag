"""验证移动后的工具仍从项目根解析路径，避免触发真实实验。"""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", [
    "inspect_retrieval", "inspect_vector_store", "chunk_id_verify",
    "audit_research_ingestion", "gen_eval",
])
def test_debug_root_resolves_outside_working_directory(name, monkeypatch, tmp_path):
    """仅执行路径赋值，验证其他工作目录下不会误指向 experiments。"""
    path = ROOT / "experiments/debug" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id in {"ROOT", "PROJECT_ROOT"}
                              for t in node.targets))
    monkeypatch.chdir(tmp_path)
    value = eval(compile(ast.Expression(assignment.value), str(path), "eval"),
                 {"Path": Path, "__file__": str(path)})
    assert value == ROOT
