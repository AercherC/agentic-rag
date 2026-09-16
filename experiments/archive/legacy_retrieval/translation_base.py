"""Phase 15B.5 归档的历史英文翻译合同。"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class TranslationError(RuntimeError):
    """表示 Translation Provider 可安全降级的翻译失败。"""


class TranslationResult(BaseModel):
    """保存原文、译文以及 Provider 是否实际执行翻译的判断。"""

    original_text: str = Field(min_length=1, description="Provider 收到的原始 Query。")
    translated_text: str = Field(
        description="英文译文；空文本由 MultilingualRetriever 按 fallback 处理。"
    )
    was_translated: bool = Field(description="是否需要运行独立 Translation 检索分支。")


@runtime_checkable
class TranslationProvider(Protocol):
    """把 Query 转换为英文，并显式声明是否需要第二检索分支。"""

    def translate_to_english(self, text: str) -> TranslationResult:
        """返回一次翻译判断；重试策略由具体 Adapter 自己负责。"""

        ...
