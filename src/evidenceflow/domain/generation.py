"""LLM messages plus the generation-result model that carries a structured answer status and has not yet parsed Citations."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMMessage(BaseModel):
    """A single vendor-neutral text Chat message."""

    role: Literal["system", "user", "assistant"] = Field(
        description="消息在 Chat 指令层级中的角色。"
    )
    content: str = Field(min_length=1, description="发送给 LLM 的非空文本内容。")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject messages that are blank and therefore meaningless, without rewriting the original message text."""

        if not value.strip():
            raise ValueError("LLM message content must not be blank")
        return value


class AnswerStatus(StrEnum):
    """Distinguish an evidence-backed answer from an explicit knowledge-base shortfall, so downstream code need not guess from natural language."""

    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class GeneratedAnswer(BaseModel):
    """Holds the status parsed by the Generator and the answer body that has not yet passed the Citation Parser."""

    status: AnswerStatus = Field(description="决定 Citation 必填语义的结构化回答状态。")

    answer_text: str = Field(min_length=1, description="LLM 原样返回或空 Context 的确定性答案。")

    @field_validator("answer_text")
    @classmethod
    def validate_answer_text(cls, value: str) -> str:
        """A blank response must never masquerade as a successful generation."""

        if not value.strip():
            raise ValueError("generated answer must not be blank")
        return value


class GeneratorOutput(BaseModel):
    """The JSON contract that rag-generator-v3 requires the model to emit; every extra field is rejected."""

    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus = Field(description="模型判断：证据足够 answered，否则 insufficient_context。")
    answer: str = Field(min_length=1, description="面向用户的回答，含 [S1] 等来源标签。")

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        """Consistent with GeneratedAnswer: a blank answer must never count as a successful generation."""

        if not value.strip():
            raise ValueError("generator output answer must not be blank")
        return value


def generator_output_json_schema() -> dict:
    """Build the JSON Schema sent to the service, taking enum values straight from AnswerStatus so the two cannot drift apart.

    Hand-inlined instead of ``model_json_schema()``: the latter emits ``$defs`` references,
    which the strict mode of some OpenAI-compatible services does not support.
    """

    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": [item.value for item in AnswerStatus]},
            "answer": {"type": "string"},
        },
        "required": ["status", "answer"],
        "additionalProperties": False,
    }
