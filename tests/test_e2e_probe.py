"""Fake tests for the cross-vendor probe (B3a-2′): parsing, verdicts, vendor isolation and
distractor-chunk selection.

No test touches the network: the probe only accepts an injected Fake client.
"""

import json

import pytest

from evidenceflow.eval.e2e import probe


class FakeClient:
    """A Fake LLM client that records call arguments; it makes no network request."""

    def __init__(self, reply: str = '{"verdict": "answered", "answer": "1%", "reason": "ok"}'):
        self.reply = reply
        self.calls = []

    def generate_json(self, messages, *, schema_name, json_schema):
        self.calls.append({"messages": messages, "schema_name": schema_name, "json_schema": json_schema})
        return self.reply


# ---------- message construction ----------

@pytest.mark.parametrize("builder", [probe.build_gold_messages, probe.build_distractor_messages])
def test_messages_contain_json_and_full_schema(builder):
    messages = builder("What percentage is used?", "we set 1% of the measurements")
    assert messages[0].role == "system" and messages[1].role == "user"
    # json_object mode requires the prompt to contain the word "JSON".
    assert any("json" in message.content.lower() for message in messages)
    user = messages[1].content
    for token in ("verdict", "answered", "insufficient_context", "answer", "reason", "JSON"):
        assert token in user


# ---------- parsing ----------

def test_parse_probe_response_strips_think_and_code_fence():
    raw = (
        "<think>先判断上下文是否足够</think>\n"
        '```json\n{"verdict": "answered", "answer": "1 percent", "reason": "context states it"}\n```'
    )
    verdict = probe.parse_probe_response(raw)
    assert verdict.verdict == "answered"
    assert verdict.answer == "1 percent"


def test_parse_probe_response_rejects_extra_fields():
    raw = json.dumps({"verdict": "answered", "answer": "1%", "reason": "ok", "confidence": 0.9})
    with pytest.raises(ValueError, match="JSON schema"):
        probe.parse_probe_response(raw)


def test_parse_probe_response_rejects_invalid_enum():
    raw = json.dumps({"verdict": "maybe", "answer": "1%", "reason": "ok"})
    with pytest.raises(ValueError, match="JSON schema"):
        probe.parse_probe_response(raw)


def test_parse_probe_response_rejects_invalid_json_and_unclosed_think():
    with pytest.raises(ValueError, match="not valid JSON"):
        probe.parse_probe_response("not json at all")
    with pytest.raises(ValueError, match="unclosed"):
        probe.parse_probe_response("<think>never closed")


# ---------- probe() ----------

def test_probe_calls_generate_json_with_probe_contract():
    client = FakeClient()
    verdict = probe.probe(client, query="What percentage?", context_text="we set 1% of it", mode="gold")
    assert verdict.verdict == "answered"
    call = client.calls[0]
    assert call["schema_name"] == "probe"
    assert call["json_schema"] is probe.PROBE_JSON_SCHEMA
    assert "What percentage?" in call["messages"][1].content


def test_probe_rejects_unknown_mode_without_calling_client():
    client = FakeClient()
    with pytest.raises(ValueError, match="unknown probe mode"):
        probe.probe(client, query="q", context_text="c", mode="bogus")
    assert client.calls == []


# ---------- key_match_verdict ----------

def test_key_match_verdict_hit_miss_and_empty_keys():
    keys = [["1%", "1 percent"]]
    assert probe.key_match_verdict(keys, "about 1 percent of measurements")
    assert not probe.key_match_verdict(keys, "about 5 percent of measurements")
    # Empty keys => False instead of raising, so the report can count key_matchable=false cases.
    assert probe.key_match_verdict([], "anything") is False


# ---------- vendor isolation ----------

def test_same_vendor():
    assert not probe.same_vendor(
        judge_base_url="https://api.deepseek.com",
        judge_model="glm-4-flash",
        generator_base_url="https://open.bigmodel.cn",
        generator_model="deepseek-flash",
    )
    assert probe.same_vendor(
        judge_base_url="https://api.deepseek.com",
        judge_model="glm-4-flash",
        generator_base_url="https://api.deepseek.com",
        generator_model="deepseek-flash",
    )
    # judge_model starting with deepseek => treated as the same source regardless of gateway.
    assert probe.same_vendor(
        judge_base_url="https://api.deepseek.com",
        judge_model="deepseek-chat",
        generator_base_url="https://api.deepseek.com",
        generator_model="deepseek-flash",
    )


# ---------- distractor-chunk selection ----------

def test_select_distractors_skips_gold_keeps_order_and_handles_shortage():
    ranked = ["a", "gold", "b", "a", "c"]
    assert probe.select_distractors(gold_chunk_ids={"gold"}, dense_ranked_ids=ranked, count=2) == ["a", "b"]
    assert probe.select_distractors(gold_chunk_ids={"a", "b"}, dense_ranked_ids=["a", "b"], count=2) == []
    assert probe.select_distractors(gold_chunk_ids=set(), dense_ranked_ids=["a"], count=2) == ["a"]


# ---------- repair retry (SPEC §13.1(g)) ----------

class RepairingClient:
    """Return truncated JSON on the first call and valid JSON on the second (repair) round."""

    def __init__(self, broken: str, fixed: str):
        self.broken = broken
        self.fixed = fixed
        self.calls = []

    def generate_json(self, messages, *, schema_name, json_schema):
        self.calls.append(messages)
        return self.broken if len(self.calls) == 1 else self.fixed


def test_probe_with_repair_recovers_from_truncated_json():
    """In practice the probe does return truncated JSON (Unterminated string); one repair should
    recover it."""

    truncated = '{"verdict": "answered", "answer": "128", "reason": "the contex'
    client = RepairingClient(truncated, '{"verdict": "answered", "answer": "128", "reason": "ok"}')
    verdict, repair = probe.probe_with_repair(
        client, query="How many units?", context_text="has 128 units", mode="gold"
    )
    assert verdict.verdict == "answered" and verdict.answer == "128"
    assert repair == "repaired"
    assert len(client.calls) == 2
    # The repair round must carry the original reply and the repair instruction; otherwise
    # resending the same request would be truncated again.
    repair_messages = client.calls[1]
    assert repair_messages[-2].role == "assistant" and repair_messages[-2].content == truncated
    assert repair_messages[-1].role == "user"
    assert "JSON" in repair_messages[-1].content


def test_probe_with_repair_reports_first_pass_success():
    client = FakeClient()
    verdict, repair = probe.probe_with_repair(
        client, query="Q?", context_text="ctx", mode="gold"
    )
    assert verdict.verdict == "answered"
    # On first-pass success repair is None, and the client is called exactly once.
    assert repair is None and len(client.calls) == 1


def test_probe_with_repair_raises_after_repair_fails():
    client = FakeClient(reply='{"verdict": "answered"')  # always truncated
    with pytest.raises(ValueError, match="unparseable after repair"):
        probe.probe_with_repair(client, query="Q?", context_text="ctx", mode="gold")
    # First call + one repair = 2 calls, no more and no fewer.
    assert len(client.calls) == 2


def test_probe_without_repair_still_raises_on_first_failure():
    """``probe`` keeps "call once, raise on failure" so the call count can be controlled
    precisely."""

    client = FakeClient(reply='{"verdict": "answered"')
    with pytest.raises(ValueError):
        probe.probe(client, query="Q?", context_text="ctx", mode="gold")
    assert len(client.calls) == 1


# ---------- T3 single-block probe ----------

def _verdict(text: str, status: str = "answered") -> probe.ProbeVerdict:
    return probe.ProbeVerdict(verdict=status, answer=text, reason="test")


def test_single_block_prompt_is_isomorphic_with_gold():
    """What G6 (3) compares is "give block A vs give block B", so the prompt must be byte-for-byte
    identical to the gold one."""

    gold = probe.build_gold_messages("Q?", "context body")
    single = probe.build_single_block_messages("Q?", "context body")
    assert [(m.role, m.content) for m in gold] == [(m.role, m.content) for m in single]


def test_single_block_verdict_accepts_half_hit():
    own = [["128", "256"]]
    other = [["512"]]
    passed, reason = probe.single_block_verdict(
        verdict=_verdict("128"), own_keys=own, other_keys=other
    )
    assert passed and reason == "ok"


def test_single_block_verdict_rejects_when_other_key_also_hits():
    # The other block's fact also appears in this block => the question does not really need both
    # blocks.
    passed, reason = probe.single_block_verdict(
        verdict=_verdict("128 and 512"), own_keys=[["128"]], other_keys=[["512"]]
    )
    assert not passed and reason == "single_block_other_key_hit"


def test_single_block_verdict_rejects_own_key_miss():
    passed, reason = probe.single_block_verdict(
        verdict=_verdict("512"), own_keys=[["128"]], other_keys=[["512"]]
    )
    assert not passed and reason == "single_block_own_key_miss"


def test_single_block_verdict_rejects_insufficient_context():
    # Failing to answer is not the same as half a hit: a probe failure must not be used as
    # evidence of multi-evidence need.
    passed, reason = probe.single_block_verdict(
        verdict=_verdict("", status="insufficient_context"), own_keys=[["128"]], other_keys=[["512"]]
    )
    assert not passed and reason == "single_block_not_answered"
