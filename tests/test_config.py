"""测试 Settings 默认值、环境变量覆盖和敏感配置处理。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from evidenceflow import Settings


def test_default_settings_can_be_instantiated(monkeypatch) -> None:
    for variable in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_TIMEOUT_SECONDS",
        "EMBEDDING_MODEL",
        "SOURCE_DOCS_PATH",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "MAX_CONTEXT_TOKENS",
        "RERANKER_MODEL",
        "RERANKER_INPUT_K",
        "RERANKER_FINAL_K",
        "RERANKER_BATCH_SIZE",
        "RERANKER_MAX_LENGTH",
        "MODEL_DEVICE",
        "RERANKER_DTYPE",
        "VECTOR_DB_PATH",
        "COLLECTION_NAME",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_api_key is None
    assert settings.llm_base_url == "https://api-inference.modelscope.cn/v1/"
    assert settings.llm_model == "Qwen/Qwen3.5-35B-A3B"
    assert settings.llm_temperature == 0.0
    assert settings.llm_timeout_seconds == 60.0
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.source_docs_path == Path("data/source_docs")
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.reranker_model == "BAAI/bge-reranker-base"
    assert settings.reranker_input_k == 20
    assert settings.reranker_final_k == 5
    assert settings.reranker_batch_size == 4
    assert settings.reranker_max_length == 512
    assert settings.model_device == "auto"
    assert settings.reranker_dtype == "float32"
    assert settings.vector_db_path == Path("data/chroma")


def test_environment_variables_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-only-secret")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("CHUNK_OVERLAP", "32")
    monkeypatch.setenv("VECTOR_DB_PATH", "custom/vector-db")
    monkeypatch.setenv("SOURCE_DOCS_PATH", "custom/source-docs")
    monkeypatch.setenv("MODEL_DEVICE", "cpu")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key == "test-only-secret"
    assert settings.llm_temperature == 0.2
    assert settings.llm_timeout_seconds == 15.0
    assert settings.chunk_size == 256
    assert settings.chunk_overlap == 32
    assert settings.vector_db_path == Path("custom/vector-db")
    assert settings.source_docs_path == Path("custom/source-docs")
    assert settings.model_device == "cpu"


def test_sensitive_setting_has_no_hardcoded_default(monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    assert Settings(_env_file=None).llm_api_key is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_base_url": "   "},
        {"llm_model": "\t"},
        {"llm_temperature": -0.1},
        {"llm_temperature": 2.1},
        {"llm_temperature": float("nan")},
        {"llm_timeout_seconds": 0},
        {"llm_timeout_seconds": float("inf")},
    ],
)
def test_invalid_llm_settings_are_rejected(overrides) -> None:
    """Endpoint、模型及采样/超时数值必须具备明确可执行语义。"""

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"reranker_model": "   "},
        {"reranker_input_k": 0},
        {"reranker_final_k": 0},
        {"reranker_input_k": 4, "reranker_final_k": 5},
        {"reranker_batch_size": 0},
        {"reranker_max_length": 0},
        {"model_device": "gpu"},
        {"reranker_dtype": "float16"},
    ],
)
def test_invalid_reranker_settings_are_rejected(overrides) -> None:
    """候选深度、模型设备与 float32 精度必须符合显式配置合同。"""

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)
