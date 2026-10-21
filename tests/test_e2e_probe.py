"""跨厂商探针的 Fake 测试（B3a-2′）：解析、判定、厂商隔离与干扰块选择。

全部用例不联网：探针只接受注入的 Fake client。
"""

import json

import pytest

from evidenceflow.eval.e2e import probe


class FakeClient:
    """记录调用参数的 Fake LLM client；不发起任何网络请求。"""

    def __init__(self, reply: str = '{"verdict": "answered", "answer": "1%", "reason": "ok"}'):
        self.reply = reply
        self.calls = []

    def generate_json(self, messages, *, schema_name, json_schema):
        self.calls.append({"messages": messages, "schema_name": schema_name, "json_schema": json_schema})
        return self.reply


# ---------- 消息构造 ----------

@pytest.mark.parametrize("builder", [probe.build_gold_messages, probe.build_distractor_messages])
def test_messages_contain_json_and_full_schema(builder):
    messages = builder("What percentage is used?", "we set 1% of the measurements")
    assert messages[0].role == "system" and messages[1].role == "user"
    # json_object 模式要求 Prompt 出现 "JSON" 一词。
    assert any("json" in message.content.lower() for message in messages)
    user = messages[1].content
    for token in ("verdict", "answered", "insufficient_context", "answer", "reason", "JSON"):
        assert token in user


# ---------- 解析 ----------

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
    # 空 keys ⇒ False 而不是抛错，便于报告统计 key_matchable=false 的题。
    assert probe.key_match_verdict([], "anything") is False


# ---------- 厂商隔离 ----------

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
    # judge_model 以 deepseek 开头 ⇒ 无论网关都视为同源。
    assert probe.same_vendor(
        judge_base_url="https://api.deepseek.com",
        judge_model="deepseek-chat",
        generator_base_url="https://api.deepseek.com",
        generator_model="deepseek-flash",
    )


# ---------- 干扰块选择 ----------

def test_select_distractors_skips_gold_keeps_order_and_handles_shortage():
    ranked = ["a", "gold", "b", "a", "c"]
    assert probe.select_distractors(gold_chunk_ids={"gold"}, dense_ranked_ids=ranked, count=2) == ["a", "b"]
    assert probe.select_distractors(gold_chunk_ids={"a", "b"}, dense_ranked_ids=["a", "b"], count=2) == []
    assert probe.select_distractors(gold_chunk_ids=set(), dense_ranked_ids=["a"], count=2) == ["a"]
