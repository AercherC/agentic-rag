"""A2.0：探测 LLM 服务对三种结构化输出方式的支持情况，结果决定 STRUCTURED_OUTPUT_MODE。

三种方式：json_schema（strict）、json_object、prompt_only。使用与线上完全相同的 Prompt
（``Generator._build_messages``）和客户端代码路径。只记录 base_url 的 host 与 model 名称，
不输出或保存 API Key。

在线行为（由 ``experiments/stage_a/online_common.py`` 提供）：
- ``--min-interval-seconds``：两次请求之间的最小间隔，默认 6 秒；
- 403 / 配置类错误：立即停止整个脚本；429：等待后重试，最多 3 次，并逐次记录；
- 断点续跑：每完成一项就追加写入 JSONL，重跑时跳过已完成项；
- 保存每次调用的原始回复与 reasoning_content 是否存在。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.domain import (  # noqa: E402
    BuiltContext,
    ContextEntry,
    SourceReference,
    generator_output_json_schema,
)
from evidenceflow.generation import (  # noqa: E402
    GenerationContractError,
    Generator,
    parse_generator_output,
)
from evidenceflow.llm import STRUCTURED_OUTPUT_MODES, OpenAICompatibleLLMClient  # noqa: E402
from evidenceflow.llm.diagnostics import serialize_llm_failure  # noqa: E402
from experiments.stage_a.online_common import (  # noqa: E402
    FatalServiceError,
    PolicyLLMClient,
    RequestPolicy,
    append_jsonl,
    read_jsonl,
)

OUTPUT = ROOT / "artifacts/stage_a/A2_probe.json"
JSONL = ROOT / "artifacts/stage_a/A2_probe.jsonl"

# 三个 RAG 风格的小请求：可回答（英文）、可回答（中文多来源）、证据不足。
REQUESTS = [
    {
        "name": "answerable_en",
        "entries": ["PoinTr represents a point cloud as an unordered set of point proxies and feeds them as a sequence to a Transformer encoder-decoder."],
        "question": "How does PoinTr represent a point cloud for Transformer processing?",
    },
    {
        "name": "answerable_zh_multi",
        "entries": ["BGE-M3 是一个支持多语言的稠密检索模型，向量维度为 1024。", "bge-reranker-base 是一个交叉编码器，用于对候选段落重新打分。"],
        "question": "这个系统用什么模型做稠密检索，向量是多少维？",
    },
    {
        "name": "insufficient",
        "entries": ["The paper evaluates on the ShapeNet-55 benchmark with Chamfer Distance."],
        "question": "What learning rate schedule does the paper use?",
    },
]

_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n?.*?\n?[ \t]*```\Z", re.DOTALL)


def build_probe_messages(item: dict[str, Any]):
    """用线上 Generator 的 Prompt 构造探测消息，保证探测的就是实际要用的请求。"""

    entries = []
    blocks = []
    for index, content in enumerate(item["entries"], start=1):
        label = f"[S{index}]"
        source = SourceReference(
            source_label=label, document_id=f"d{index}", file_name=f"probe{index}.md",
            page_start=None, page_end=None, chunk_id=f"c{index}",
        )
        entries.append(
            ContextEntry(
                source_label=label, retrieval_rank=index, chunk_id=f"c{index}",
                content=content, source_reference=source, token_count=10,
            )
        )
        blocks.append(f"{label}\nFile: probe{index}.md\nContent:\n{content}")
    context = BuiltContext(
        context_text="\n\n".join(blocks), entries=entries, total_tokens=10 * len(entries),
        candidate_count=len(entries), included_count=len(entries), skipped_count=0,
    )
    return Generator._build_messages(item["question"], context)


def analyze_raw(raw: str) -> dict[str, Any]:
    """从原始回复提取探测记录项（纯函数，不改写原文）。"""

    text = raw.strip()
    has_open = text.startswith("<think>") or "<think>" in text
    closed = "</think>" in text
    json_ok_raw = True
    try:
        json.loads(text)
    except ValueError:
        json_ok_raw = False
    body = re.sub(r"\A<think>.*?</think>", "", text, count=1, flags=re.DOTALL).strip()
    error: str | None = None
    schema_ok = False
    try:
        parse_generator_output(raw)
        schema_ok = True
    except GenerationContractError as exc:
        error = str(exc)
    return {
        "content_length": len(raw),
        "json_loads_ok_raw": json_ok_raw,
        "schema_valid_after_cleanup": schema_ok,
        "parse_error": error,
        "has_think_open": has_open,
        "think_closed": closed if has_open else None,
        "wrapped_in_code_fence": bool(_FENCE.match(body)),
    }


def aggregate_probe(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按优先级 json_schema > json_object > prompt_only 选择模式（全部请求成功且通过校验才算可用）。"""

    by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in STRUCTURED_OUTPUT_MODES}
    for record in records:
        if record.get("completed"):
            by_mode[record["mode"]].append(record)

    def usable(mode: str) -> bool:
        rows = by_mode[mode]
        return len(rows) == len(REQUESTS) and all(
            r.get("error") is None and r["analysis"]["schema_valid_after_cleanup"] for r in rows
        )

    usable_map = {m: usable(m) for m in STRUCTURED_OUTPUT_MODES}
    selected = next((m for m in STRUCTURED_OUTPUT_MODES if usable_map[m]), None)
    return {"modes": by_mode, "usable": usable_map, "selected_mode": selected}


def run_probe(
    make_client: Callable[[str], Any],
    policy: RequestPolicy,
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    """逐项探测并追加写入 JSONL；返回全部记录。403 时抛出 FatalServiceError（已完成项已落盘）。"""

    records = read_jsonl(jsonl_path)
    done = {(r["mode"], r["request"]) for r in records if r.get("completed")}
    schema = generator_output_json_schema()
    for mode in STRUCTURED_OUTPUT_MODES:
        client = PolicyLLMClient(make_client(mode), policy)
        for item in REQUESTS:
            if (mode, item["name"]) in done:
                continue
            client.begin()
            messages = build_probe_messages(item)
            record: dict[str, Any] = {"mode": mode, "request": item["name"], "error": None}
            try:
                raw = client.generate_json(messages, schema_name="generator_output", json_schema=schema)
            except FatalServiceError:
                raise
            except Exception as exc:
                record["error"] = serialize_llm_failure(exc)
                # 限流耗尽不是“模式不可用”的证据：不标记完成，下次重跑会重试这一项。
                record["completed"] = not getattr(exc, "rate_limit_exhausted", False)
                record["rate_limit_exhausted"] = not record["completed"]
            else:
                call = client.calls[-1]
                record.update(
                    completed=True,
                    analysis=analyze_raw(raw),
                    raw=raw,
                    usage=call.get("usage"),
                    usage_present=call.get("usage") is not None,
                    reasoning_content_present=call.get("reasoning_content_present"),
                    reasoning_content_non_empty=call.get("reasoning_content_non_empty"),
                    finish_reason=call.get("finish_reason"),
                )
            record["elapsed_seconds"] = client.calls[-1]["elapsed_seconds"] if client.calls else None
            record["rate_limit_events"] = client.question_rate_limit_events
            append_jsonl(jsonl_path, record)
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-interval-seconds", type=float, default=6.0)
    parser.add_argument("--jsonl", type=Path, default=JSONL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    settings = Settings()
    if not settings.llm_api_key:
        print("ERROR: LLM_API_KEY is not configured", file=sys.stderr)
        return 2
    policy = RequestPolicy(min_interval_seconds=args.min_interval_seconds)

    def make_client(mode: str) -> OpenAICompatibleLLMClient:
        # max_attempts=1：429 的等待与重试统一由 RequestPolicy 负责，避免客户端内部立即重试。
        return OpenAICompatibleLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            max_attempts=1,
            structured_output_mode=mode,
        )

    try:
        records = run_probe(make_client, policy, args.jsonl)
    except FatalServiceError as exc:
        print(f"STOP: {exc}. Completed items are kept in {args.jsonl}.", file=sys.stderr)
        return 3
    summary = aggregate_probe(records)
    payload = {
        "base_url_host": urlparse(settings.llm_base_url).netloc,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "min_interval_seconds": args.min_interval_seconds,
        "rate_limit_events": policy.rate_limit_events,
        **summary,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_mode": summary["selected_mode"], "usable": summary["usable"]}, ensure_ascii=False))
    return 0 if summary["selected_mode"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
