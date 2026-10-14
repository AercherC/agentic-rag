"""A2.3：用 Phase 12 数据集的 25 个 query 走线上完整链路，统计格式、状态与错误（不评估答案质量）。

链路与 ``runtime.build_rag_service`` 相同（Exact Dense20 → Reranker → Top5 → Context →
Generator v3 → Citation）。冒烟不是数值 Gate，允许使用 GPU（必须传 ``--allow-gpu``）；
query 使用数据集的 ``query`` 字段（线上真实输入），与 B0' 使用的 ``english_query`` 不是同一个字段，
因此不与 B0' 比较，只记录每题 Top5 的 chunk_id 供阶段 B 核对 Context。

在线行为见 ``online_common.py``：最小请求间隔、403 立即停止、429 等待重试（单独统计）、
JSONL 断点续跑、保存每次调用（含修复调用）的原始回复与 reasoning_content 诊断。
不覆盖 Phase 12 的任何文件。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.generation import GENERATOR_PROMPT_VERSION, Generator  # noqa: E402
from evidenceflow.llm import STRUCTURED_OUTPUT_MODE  # noqa: E402
from evidenceflow.model_device import resolve_model_device  # noqa: E402
from experiments.stage_a.online_common import (  # noqa: E402
    FatalServiceError,
    PolicyLLMClient,
    ResumeIdentityError,
    check_resume_identity,
    identity_fields,
    RequestPolicy,
    append_jsonl,
    extract_allowed_labels,
    failure_class,
    pending,
    read_jsonl,
    rewrite_jsonl,
    summarize_smoke,
)

DATASET = ROOT / "evals/evaluation_v3/phase12_rag_quality_v1/generation_eval_dataset.json"
JSONL = ROOT / "artifacts/stage_a/A2_generation_smoke.jsonl"
OUTPUT = ROOT / "artifacts/stage_a/A2_generation_smoke.json"
# 这两类是服务/限流问题而非生成合同问题：--retry-llm-errors 时清掉重跑。
_RETRYABLE_CLASSES = {"llm_request_error", "rate_limit_exhausted"}


class RecordingRetriever:
    """包装正式检索器，只记录最近一次 Top5 的 chunk_id（按顺序），不改变检索行为。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_ids: list[str] = []

    def retrieve(self, query: str) -> list:
        self.last_ids = []
        results = self._inner.retrieve(query)
        self.last_ids = [r.chunk.chunk_id for r in results]
        return results


def process_question(
    item: dict[str, Any],
    *,
    service: Any,
    generator: Any,
    llm: PolicyLLMClient,
    recorder: RecordingRetriever,
    device: str,
    identity: dict[str, str],
) -> dict[str, Any]:
    """处理一题并返回可直接落盘的记录。403 的 FatalServiceError 不在此捕获，向上终止脚本。"""

    llm.begin()
    recorder.last_ids = []
    started = time.perf_counter()
    answer = None
    error: Exception | None = None
    try:
        answer = service.answer(item["query"])
    except FatalServiceError:
        raise
    except Exception as exc:  # 记录并继续下一题：格式失败、Citation 失败、服务错误等
        error = exc
    elapsed = round(time.perf_counter() - started, 3)
    calls = llm.calls
    return {
        **identity,
        "question_id": item["question_id"],
        "query": item["query"],
        "answerability": item.get("answerability"),
        "device": device,
        "top5_chunk_ids": list(recorder.last_ids),
        "status": answer.status.value if answer is not None else None,
        "answer": answer.answer if answer is not None else None,
        "source_labels": [s.source_label for s in answer.sources] if answer is not None else [],
        "exception_type": type(error).__name__ if error is not None else None,
        "exception_message": str(error)[:300] if error is not None else None,
        "failure_class": failure_class(error),
        "diagnostics": generator.last_diagnostics,
        "allowed_labels": extract_allowed_labels(calls[0]["messages"]) if calls else [],
        "rate_limit_events": llm.question_rate_limit_events,
        "calls": calls,
        "elapsed_seconds": elapsed,
    }


def run_smoke(
    items: list[dict[str, Any]],
    *,
    service: Any,
    generator: Any,
    llm: PolicyLLMClient,
    recorder: RecordingRetriever,
    device: str,
    jsonl_path: Path,
    identity: dict[str, str],
    retry_llm_errors: bool = False,
) -> list[dict[str, Any]]:
    """逐题运行并即时追加写入；已完成的题在重跑时被跳过。"""

    records = read_jsonl(jsonl_path)
    check_resume_identity(records, identity, jsonl_path)
    if retry_llm_errors:
        kept = [r for r in records if r.get("failure_class") not in _RETRYABLE_CLASSES]
        if len(kept) != len(records):
            rewrite_jsonl(jsonl_path, kept)
        records = kept
    done = {r["question_id"] for r in records}
    for item in pending(items, done, key=lambda i: i["question_id"]):
        record = process_question(
            item, service=service, generator=generator, llm=llm, recorder=recorder, device=device,
            identity=identity,
        )
        append_jsonl(jsonl_path, record)
        records.append(record)
    return records


def write_summary(path: Path, records: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """汇总文件不含原始回复（保存在 JSONL 中），保持小而可读。"""

    payload = {
        **meta,
        "summary": summarize_smoke(records),
        "records": [{k: v for k, v in r.items() if k != "calls"} for r in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-interval-seconds", type=float, default=6.0)
    parser.add_argument("--allow-gpu", action="store_true", help="允许非 CPU 设备（冒烟不是数值 Gate）")
    parser.add_argument("--retry-llm-errors", action="store_true", help="重跑上次因服务错误/限流失败的题")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--jsonl", type=Path, default=JSONL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    settings = Settings()
    if not settings.llm_api_key:
        print("ERROR: LLM_API_KEY is not configured", file=sys.stderr)
        return 2
    device = resolve_model_device(settings.model_device)
    if device != "cpu" and not args.allow_gpu:
        print(f"ERROR: resolved device is {device!r}; pass --allow-gpu or set MODEL_DEVICE=cpu", file=sys.stderr)
        return 2

    # 延迟导入：重型依赖只在真正运行时加载，测试只用上面的纯函数。
    from evidenceflow.runtime import build_llm_client, build_rag_service, build_retriever

    items = json.loads(args.dataset.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        items = items[: args.limit]

    policy = RequestPolicy(min_interval_seconds=args.min_interval_seconds)
    # max_attempts=1：429 的等待与重试由 RequestPolicy 统一负责。
    inner = build_llm_client(settings.model_copy(update={"llm_max_attempts": 1}))
    llm = PolicyLLMClient(inner, policy)
    generator = Generator(llm)
    retriever = build_retriever(settings)
    actual_device = retriever.dense_retriever.embedding_provider.device
    recorder = RecordingRetriever(retriever)
    service = build_rag_service(settings, retriever=recorder, generator=generator)

    meta = {
        "base_url_host": urlparse(settings.llm_base_url).netloc,
        "model": settings.llm_model,
        "device": actual_device,
        "min_interval_seconds": args.min_interval_seconds,
        "dataset": str(args.dataset.relative_to(ROOT)) if args.dataset.is_relative_to(ROOT) else str(args.dataset),
        "query_field": "query",
        "llm_thinking": settings.llm_thinking,
        "prompt_version": GENERATOR_PROMPT_VERSION,
        "structured_output_mode": STRUCTURED_OUTPUT_MODE,
    }
    identity = identity_fields(
        structured_output_mode=STRUCTURED_OUTPUT_MODE, prompt_version=GENERATOR_PROMPT_VERSION,
        model=settings.llm_model, llm_thinking=settings.llm_thinking,
    )
    try:
        records = run_smoke(
            items, service=service, generator=generator, llm=llm, recorder=recorder,
            device=actual_device, jsonl_path=args.jsonl, identity=identity, retry_llm_errors=args.retry_llm_errors,
        )
    except FatalServiceError as exc:
        print(f"STOP: {exc}. Completed questions are kept in {args.jsonl}.", file=sys.stderr)
        return 3
    except ResumeIdentityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4
    payload = write_summary(args.output, records, meta)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
