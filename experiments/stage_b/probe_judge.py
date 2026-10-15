# -*- coding: utf-8 -*-
"""B0：评委（LLM-as-judge）连通性与结构化输出模式探测（一次性在线实验）。

- 配置只取 ``EVAL_JUDGE_BASE_URL`` / ``EVAL_JUDGE_MODEL`` / ``EVAL_JUDGE_API_KEY`` 三个键：
  进程环境优先，其次 ``.env``（经 ``dotenv_values`` 读取，不注入全局 ``Settings``）。缺任一项退出码 2，不发请求。
- 发请求前做厂商隔离检查：评委与生成器（``Settings().llm_base_url``）注册域相同，或评委 model
  以 ``deepseek`` 开头，则退出码 2，不发请求。
- 依次探测 ``json_object``、``json_schema``（strict）；只有出现网络/瞬态错误时才用第 3 次重试该项。
  客户端 ``max_attempts=1``，总真实请求数 ≤ 3。
- 产物 ``artifacts/stage_b/B0_judge_probe.json``；只记录 host 与 model，绝不输出或保存 API Key。

文件名不以 ``test_`` 开头，不会被 pytest 收集。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from dotenv import dotenv_values  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.domain import LLMMessage  # noqa: E402
from evidenceflow.llm import OpenAICompatibleLLMClient  # noqa: E402
from evidenceflow.llm.diagnostics import serialize_llm_failure  # noqa: E402

OUTPUT = ROOT / "artifacts/stage_b/B0_judge_probe.json"
KEYS = ("EVAL_JUDGE_BASE_URL", "EVAL_JUDGE_MODEL", "EVAL_JUDGE_API_KEY")
MAX_REQUESTS = 3
TRANSIENT = {"timeout", "network_connection", "rate_limit", "provider_5xx"}

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

MESSAGES = [
    LLMMessage(
        role="system",
        content=(
            "You judge whether a statement is supported by the context. Reply with a JSON object only: "
            '{"verdict": "SUPPORTED" or "UNSUPPORTED", "reason": "<one short sentence>"}.'
        ),
    ),
    LLMMessage(
        role="user",
        content="Context: The library opens at 9 am on weekdays.\nStatement: The library opens at 9 am on Monday.",
    ),
]


class Verdict(BaseModel):
    """本地校验模型：字段与 SCHEMA 一致，禁止多余字段。"""

    model_config = ConfigDict(extra="forbid")
    verdict: Literal["SUPPORTED", "UNSUPPORTED"]
    reason: str


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _registered_domain(host: str) -> str:
    """粗略取注册域：最后两段；若倒数第二段是 com/net/org 等二级后缀，则取最后三段。"""

    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"com", "net", "org", "gov", "edu", "ac", "co"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _load_config() -> dict[str, str | None]:
    file_values = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
    return {key: (os.environ.get(key) or file_values.get(key) or None) for key in KEYS}


def main() -> int:
    config = _load_config()
    missing = [key for key in KEYS if not config[key]]
    if missing:
        print(f"STOP: missing judge config keys: {missing}")
        return 2
    api_key = config["EVAL_JUDGE_API_KEY"]
    judge_host = _host(config["EVAL_JUDGE_BASE_URL"])
    judge_model = config["EVAL_JUDGE_MODEL"]
    settings = Settings()
    gen_host = _host(settings.llm_base_url)
    same_vendor = (
        _registered_domain(judge_host) == _registered_domain(gen_host)
        or judge_model.lower().startswith("deepseek")
    )
    print(f"judge={judge_host}/{judge_model} generator={gen_host}/{settings.llm_model} same_vendor={same_vendor}")
    if same_vendor:
        print("STOP: judge and generator share a vendor; no request sent")
        return 2

    requests_sent = 0
    results: dict[str, dict] = {}

    def probe(mode: str) -> dict:
        nonlocal requests_sent
        client = OpenAICompatibleLLMClient(
            api_key=api_key,
            base_url=config["EVAL_JUDGE_BASE_URL"],
            model=judge_model,
            temperature=0,
            max_attempts=1,
            structured_output_mode=mode,
            thinking="auto",
        )
        record: dict = {"mode": mode}
        started = time.perf_counter()
        requests_sent += 1
        try:
            raw = client.generate_json(MESSAGES, schema_name="judge_verdict", json_schema=SCHEMA)
            record["success"] = True
            record["raw_reply"] = raw[:500]
            try:
                Verdict.model_validate(json.loads(raw))
                record["parsed_and_valid"] = True
            except Exception as exc:
                record["parsed_and_valid"] = False
                record["validation_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        except Exception as exc:
            failure = serialize_llm_failure(exc)
            record["success"] = False
            record["parsed_and_valid"] = False
            record["error_classification"] = failure["classification"]
            record["error_root_cause_type"] = failure["root_cause_type"]
            record["error_root_cause_message"] = str(failure["root_cause_message"])[:500]
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        record["usage"] = client.last_usage
        return record

    for mode in ("json_object", "json_schema"):
        results[mode] = probe(mode)
    transient = [m for m, r in results.items() if not r["success"] and r.get("error_classification") in TRANSIENT]
    if transient and requests_sent < MAX_REQUESTS:
        mode = transient[0]
        retry = probe(mode)
        retry["retry_of_first_attempt"] = True
        results[mode] = {"first_attempt": results[mode], "retry": retry, **{k: retry[k] for k in ("success", "parsed_and_valid")}}
    assert requests_sent <= MAX_REQUESTS

    def ok(mode: str) -> bool:
        return bool(results[mode]["success"] and results[mode]["parsed_and_valid"])

    recommended = "json_schema" if ok("json_schema") else ("json_object" if ok("json_object") else None)
    payload = {
        "judge_host": judge_host,
        "judge_model": judge_model,
        "generator_host": gen_host,
        "generator_model": settings.llm_model,
        "same_vendor": same_vendor,
        "requests_sent": requests_sent,
        "results": results,
        "recommended_mode": recommended,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    # 泄漏自检：产物中不得出现 key（只做比较，不打印 key）。
    if api_key in text:
        print("STOP: API key found in output payload; not written")
        return 3
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text + "\n", encoding="utf-8")
    summary = {m: (r["success"], r["parsed_and_valid"]) for m, r in results.items()}
    print(f"requests_sent={requests_sent} results(success,valid)={summary} recommended={recommended}")
    return 0 if recommended else 2


if __name__ == "__main__":
    raise SystemExit(main())
