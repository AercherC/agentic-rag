"""Public unified LLM Client contract, its exceptions, and the OpenAI-compatible adapter."""

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
