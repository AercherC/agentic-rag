"""把底层 LLM 异常链转换为可审计且不含敏感配置的诊断信息。"""

from __future__ import annotations

import re
from typing import Any

import httpx

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|authorization)\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
    ),
    re.compile(r"(?i)(bearer)\s+[^\s,;]+"),
)


def _safe_message(exc: BaseException) -> str:
    """清理异常文本中的常见凭据形态，并限制持久化长度。"""

    message = str(exc)
    for pattern in _SECRET_PATTERNS:
        message = pattern.sub(r"\1=[REDACTED]", message)
    return message[:1000]


def _status_code(exc: BaseException) -> int | None:
    """兼容 OpenAI SDK 与 HTTP 客户端常见的状态码暴露方式。"""

    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify_llm_failure(exc: BaseException) -> str:
    """沿异常链识别可操作的失败类别，未知类型明确归入 other。"""

    chain = exception_chain(exc)
    exceptions = [item["exception"] for item in chain]
    type_names = {type(item).__name__ for item in exceptions}
    statuses = [code for item in exceptions if (code := _status_code(item)) is not None]
    if 429 in statuses:
        return "rate_limit"
    if any(code >= 500 for code in statuses):
        return "provider_5xx"
    if any(code in {401, 403} for code in statuses):
        return "configuration"
    if "APITimeoutError" in type_names or any(
        isinstance(item, (TimeoutError, httpx.TimeoutException)) for item in exceptions
    ):
        return "timeout"
    if "APIConnectionError" in type_names or any(
        isinstance(item, (ConnectionError, httpx.NetworkError, httpx.RequestError))
        for item in exceptions
    ):
        return "network_connection"
    if statuses:
        return "request_4xx" if any(400 <= code < 500 for code in statuses) else "other"
    return "other"


def exception_chain(exc: BaseException) -> list[dict[str, Any]]:
    """保留从业务异常到底层原因的有序链，并防止循环引用。"""

    result: list[dict[str, Any]] = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        result.append(
            {
                "exception": current,
                "type": type(current).__name__,
                "module": type(current).__module__,
                "message": _safe_message(current),
                "status_code": _status_code(current),
            }
        )
        current = current.__cause__ or current.__context__
    return result


def serialize_llm_failure(exc: BaseException) -> dict[str, Any]:
    """生成可直接写入 Evaluation JSON 的脱敏失败诊断。"""

    chain = exception_chain(exc)
    return {
        "classification": classify_llm_failure(exc),
        "root_cause_type": chain[-1]["type"],
        "root_cause_message": chain[-1]["message"],
        "cause_chain": [
            {key: value for key, value in item.items() if key != "exception"}
            for item in chain
        ],
    }
