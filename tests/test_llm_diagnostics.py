"""Test LLM failure classification, cause-chain preservation and sensitive-information
scrubbing."""

import httpx

from evidenceflow.llm import LLMRequestError, serialize_llm_failure


def test_timeout_root_cause_is_preserved_and_classified() -> None:
    cause = httpx.ReadTimeout("provider timed out")
    outer = LLMRequestError("request failed")
    outer.__cause__ = cause

    diagnostic = serialize_llm_failure(outer)

    assert diagnostic["classification"] == "timeout"
    assert diagnostic["root_cause_type"] == "ReadTimeout"
    assert [item["type"] for item in diagnostic["cause_chain"]] == [
        "LLMRequestError",
        "ReadTimeout",
    ]


def test_http_status_and_secret_are_handled_safely() -> None:
    request = httpx.Request("POST", "https://example.test/v1")
    cause = httpx.HTTPStatusError(
        "authorization: Bearer actual-secret",
        request=request,
        response=httpx.Response(429, request=request),
    )

    diagnostic = serialize_llm_failure(cause)

    assert diagnostic["classification"] == "rate_limit"
    assert "actual-secret" not in diagnostic["root_cause_message"]
