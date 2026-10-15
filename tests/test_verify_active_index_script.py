"""测试 scripts/verify_active_index.py 的输出行为：默认只打印到 stdout，不写任何文件。

用 monkeypatch 替换检查函数，不打开任何真实 Chroma 存储。
"""

import json

import pytest

import scripts.verify_active_index as module
from evidenceflow.config import Settings

FAKE_RESULT = {"collection_name": "fake", "checks": {"collection_exists": True}, "pass": True}


@pytest.fixture
def fake_inspect(monkeypatch):
    calls = []

    def fake(persist_path, collection_name, *, expected_count=None, expected_strategy="hybrid"):
        calls.append((persist_path, collection_name, expected_count))
        return dict(FAKE_RESULT)

    monkeypatch.setattr(module, "inspect_active_index", fake)
    # 不读取仓库 .env：脚本内部构造的 Settings 改为纯默认值。
    monkeypatch.setattr(module, "Settings", lambda: Settings(_env_file=None))
    return calls


def _snapshot(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*"))


def test_default_writes_no_file_and_prints_json(tmp_path, monkeypatch, capsys, fake_inspect) -> None:
    tracked = module.ROOT / "artifacts" / "cleanup" / "active_index_smoke.json"
    tracked_before = tracked.read_bytes() if tracked.exists() else None
    monkeypatch.chdir(tmp_path)
    before = _snapshot(tmp_path)

    code = module.main(["--path", str(tmp_path / "chroma"), "--collection", "fake", "--expected-count", "3"])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == FAKE_RESULT
    assert fake_inspect == [(tmp_path / "chroma", "fake", 3)]
    assert _snapshot(tmp_path) == before  # 当前目录下没有新文件
    assert (tracked.read_bytes() if tracked.exists() else None) == tracked_before  # 被跟踪的历史产物未被改写


def test_explicit_output_writes_file(tmp_path, capsys, fake_inspect) -> None:
    output = tmp_path / "nested" / "smoke.json"

    assert module.main(["--path", str(tmp_path), "--collection", "fake", "--output", str(output)]) == 0

    assert json.loads(output.read_text(encoding="utf-8")) == FAKE_RESULT
    assert json.loads(capsys.readouterr().out) == FAKE_RESULT


def test_failed_check_returns_2(tmp_path, monkeypatch, capsys, fake_inspect) -> None:
    monkeypatch.setattr(module, "inspect_active_index", lambda *a, **k: {"checks": {"x": False}, "pass": False})

    assert module.main(["--path", str(tmp_path), "--collection", "fake"]) == 2
