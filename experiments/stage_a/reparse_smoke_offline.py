"""离线重解析：读取冒烟保存的原始回复，重新执行 v3 解析、修复流程与 Citation 校验。

修改解析器之后无需再调用 API 就能回归验证。做法：用保存的回复构造 ReplayClient，驱动真实的
``Generator``（与线上同一份解析与修复逻辑）；再用保存的 allowed_labels 做 Citation 校验。
输出统计格式与冒烟的 ``summary`` 一致。若新解析逻辑需要比已保存回复更多的调用，
该题记为 ``ReplayExhaustedError``（需要在线重跑），而不是臆造回复。

不联网，不读取 API Key。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evidenceflow.citation import (  # noqa: E402
    check_citation_rules,
    classify_citation_labels,
)
from evidenceflow.domain import BuiltContext, ContextEntry, SourceReference  # noqa: E402
from evidenceflow.generation import Generator  # noqa: E402
from experiments.stage_a.online_common import (  # noqa: E402
    failure_class,
    read_jsonl,
    summarize_smoke,
)

JSONL = ROOT / "artifacts/stage_a/A2_generation_smoke.jsonl"
OUTPUT = ROOT / "artifacts/stage_a/A2_generation_smoke_reparsed.json"


class ReplayExhaustedError(RuntimeError):
    """新解析逻辑需要比已保存回复更多的 LLM 调用，无法离线复现。"""


class ReplayClient:
    """按顺序回放已保存的原始回复，并复现 usage（仅用于离线统计）。"""

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = [c for c in calls if "raw" in c]
        self._index = 0
        self.last_usage: dict | None = None

    def generate(self, messages) -> str:  # pragma: no cover - v3 不走该路径
        raise AssertionError("Generator v3 must use generate_json")

    def generate_json(self, messages, *, schema_name: str, json_schema: dict) -> str:
        if self._index >= len(self._calls):
            raise ReplayExhaustedError("replay needs more LLM calls than were saved")
        call = self._calls[self._index]
        self._index += 1
        self.last_usage = call.get("usage")
        return call["raw"]


def make_context(labels: list[str]) -> BuiltContext:
    """按保存的 allowed_labels 合成最小 Context：Generator 只用到“是否有条目”和 context_text。"""

    labels = labels or ["[S1]"]
    entries = [
        ContextEntry(
            source_label=label,
            retrieval_rank=index,
            chunk_id=f"replay-{index}",
            content="replay",
            source_reference=SourceReference(
                source_label=label, document_id="replay", file_name="replay.md",
                page_start=None, page_end=None, chunk_id=f"replay-{index}",
            ),
            token_count=1,
        )
        for index, label in enumerate(labels, start=1)
    ]
    return BuiltContext(
        context_text="replay", entries=entries, total_tokens=len(entries),
        candidate_count=len(entries), included_count=len(entries), skipped_count=0,
    )


def reparse_record(record: dict[str, Any]) -> dict[str, Any]:
    """对一条冒烟记录重新执行解析与 Citation 校验，返回与冒烟同格式的精简记录。"""

    calls = record.get("calls") or []
    base = {
        "question_id": record["question_id"],
        "rate_limit_events": record.get("rate_limit_events", []),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "original_status": record.get("status"),
        "original_failure_class": record.get("failure_class"),
    }
    if not any("raw" in c for c in calls):
        # 当时就没有拿到任何回复（服务错误等），无法离线重解析：原样保留分类。
        return {
            **base, "status": None, "exception_type": record.get("exception_type"),
            "failure_class": record.get("failure_class"), "diagnostics": record.get("diagnostics") or {},
            "reparsable": False,
        }
    generator = Generator(ReplayClient(calls))
    error: Exception | None = None
    status: str | None = None
    try:
        generated = generator.generate(record["query"], make_context(record.get("allowed_labels", [])))
        parsed = classify_citation_labels(generated.answer_text, set(record.get("allowed_labels", [])))
        check_citation_rules(generated.status, parsed)
        status = generated.status.value
    except Exception as exc:
        error = exc
    return {
        **base,
        "status": status,
        "exception_type": type(error).__name__ if error else None,
        "failure_class": failure_class(error),
        "diagnostics": generator.last_diagnostics,
        "reparsable": True,
    }


def reparse_all(records: list[dict[str, Any]]) -> dict[str, Any]:
    """重解析全部记录并给出与原结果不一致的题目清单。"""

    reparsed = [reparse_record(r) for r in records]
    changed = {
        r["question_id"]: {
            "status": [r["original_status"], r["status"]],
            "failure_class": [r["original_failure_class"], r["failure_class"]],
        }
        for r in reparsed
        if r["reparsable"]
        and (r["original_status"] != r["status"] or r["original_failure_class"] != r["failure_class"])
    }
    return {
        "summary": summarize_smoke(reparsed),
        "changed_vs_original": changed,
        "not_reparsable": [r["question_id"] for r in reparsed if not r["reparsable"]],
        "records": reparsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=JSONL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    records = read_jsonl(args.jsonl)
    if not records:
        print(f"ERROR: no records in {args.jsonl}", file=sys.stderr)
        return 2
    payload = reparse_all(records)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "changed": payload["changed_vs_original"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
