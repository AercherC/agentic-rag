"""定义供应商无关的同步 LLM 文本生成合同与最小异常层级。"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from evidenceflow.domain.generation import LLMMessage


class LLMError(RuntimeError):
    """所有正式 LLM Client 配置和请求失败的统一基类。"""


class LLMConfigurationError(LLMError):
    """表示 Client 无法由当前非敏感配置和凭据安全创建。"""


class LLMRequestError(LLMError):
    """表示 LLM 请求失败或供应商响应不满足纯文本合同。"""


@runtime_checkable
class LLMClient(Protocol):
    """接受项目消息模型并同步返回原始模型文本。"""

    @property
    def last_usage(self) -> dict | None:
        """最近一次调用的 ``prompt_tokens`` / ``completion_tokens``；服务未返回 usage 时为 None。"""

        ...

    def generate_json(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema_name: str,
        json_schema: dict,
    ) -> str:
        """按已选定的结构化输出模式请求 JSON，返回原始文本，不在此层解析。"""

        ...

    def generate(self, messages: Sequence[LLMMessage]) -> str:
        """生成单条非空文本，不承担 RAG 或 Citation 业务逻辑。"""

        ...
