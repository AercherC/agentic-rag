"""Phase 15B.5 归档的历史 LLM Query 翻译实现。"""

from evidenceflow.domain.generation import LLMMessage
from evidenceflow.llm.base import LLMClient, LLMError
from evidenceflow.translation.base import TranslationError, TranslationResult

_TRANSLATION_SYSTEM_PROMPT = """Translate the user search query into faithful English.
Do not answer the question.
Do not expand or rewrite the query.
Do not add details, facts, or source terms that are not present.
Return only the translated query."""


class LLMTranslationProvider:
    """把 Query 交给统一 LLMClient，仅返回可供第二检索分支使用的英文译文。"""

    def __init__(self, llm_client: LLMClient) -> None:
        """注入与 Generator 共用配置的供应商无关 LLM Client。"""

        self._llm_client = llm_client

    def translate_to_english(self, text: str) -> TranslationResult:
        """忠实翻译非空 Query，并将底层 LLM 失败转换为可安全降级的 TranslationError。"""

        if not isinstance(text, str) or not text.strip():
            raise TranslationError("translation query must not be blank")
        messages = [
            LLMMessage(role="system", content=_TRANSLATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=text),
        ]
        try:
            translated_text = self._llm_client.generate(messages)
        except LLMError as exc:
            # Translation 是 Retrieval 增强分支，统一转成既有可降级异常合同。
            raise TranslationError("LLM query translation failed") from exc
        if not translated_text.strip():
            raise TranslationError("LLM query translation returned blank text")
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            was_translated=translated_text.strip() != text.strip(),
        )
