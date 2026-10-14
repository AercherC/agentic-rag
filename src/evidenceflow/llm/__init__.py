"""公开统一 LLM Client 合同、异常和 OpenAI-compatible Adapter。"""

from evidenceflow.llm.base import (
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMRequestError,
)
from evidenceflow.llm.openai_compatible import (
    STRUCTURED_OUTPUT_MODE,
    STRUCTURED_OUTPUT_MODES,
    OpenAICompatibleLLMClient,
    build_structured_request_params,
)
from evidenceflow.llm.diagnostics import classify_llm_failure, serialize_llm_failure

__all__ = [
    "LLMClient",
    "LLMConfigurationError",
    "LLMError",
    "LLMRequestError",
    "OpenAICompatibleLLMClient",
    "STRUCTURED_OUTPUT_MODE",
    "STRUCTURED_OUTPUT_MODES",
    "build_structured_request_params",
    "classify_llm_failure",
    "serialize_llm_failure",
]
