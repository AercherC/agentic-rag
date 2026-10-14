"""LLM 消息与带结构化回答状态、尚未解析 Citation 的生成结果模型。"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMMessage(BaseModel):
    """表示供应商无关的单条文本 Chat 消息。"""

    role: Literal["system", "user", "assistant"] = Field(
        description="消息在 Chat 指令层级中的角色。"
    )
    content: str = Field(min_length=1, description="发送给 LLM 的非空文本内容。")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """拒绝无语义的空白消息，但不改写原始消息文本。"""

        if not value.strip():
            raise ValueError("LLM message content must not be blank")
        return value


class AnswerStatus(StrEnum):
    """区分有证据回答与显式知识库不足，避免下游依赖自然语言猜测。"""

    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class GeneratedAnswer(BaseModel):
    """保存 Generator 解析后的状态与尚未经过 Citation Parser 的答案正文。"""

    status: AnswerStatus = Field(description="决定 Citation 必填语义的结构化回答状态。")

    answer_text: str = Field(min_length=1, description="LLM 原样返回或空 Context 的确定性答案。")

    @field_validator("answer_text")
    @classmethod
    def validate_answer_text(cls, value: str) -> str:
        """禁止把空白响应伪装为一次成功生成。"""

        if not value.strip():
            raise ValueError("generated answer must not be blank")
        return value


class GeneratorOutput(BaseModel):
    """rag-generator-v3 要求模型输出的 JSON 合同；多余字段一律拒绝。"""

    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus = Field(description="模型判断：证据足够 answered，否则 insufficient_context。")
    answer: str = Field(min_length=1, description="面向用户的回答，含 [S1] 等来源标签。")

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        """与 GeneratedAnswer 一致，禁止空白回答被当作一次成功生成。"""

        if not value.strip():
            raise ValueError("generator output answer must not be blank")
        return value


def generator_output_json_schema() -> dict:
    """生成发给服务端的 JSON Schema，枚举值直接取自 AnswerStatus，避免两处手写漂移。

    手工内联而不用 ``model_json_schema()``：后者会产生 ``$defs`` 引用，部分
    OpenAI-compatible 服务的 strict 模式不支持引用。
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
