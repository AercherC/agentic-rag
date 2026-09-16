"""验证 chunk_strategy 配置守卫与 IngestionService 策略分发合同（Phase 13F）。"""

import pytest
from pydantic import ValidationError

from evidenceflow.config import Settings
from evidenceflow.ingestion.service import IngestionService


def test_chunk_strategy_defaults_to_hybrid() -> None:
    """Phase 13F 晋升：默认策略为 hybrid。"""

    settings = Settings(llm_api_key=None, collection_name="ef_test_hybrid")
    assert settings.chunk_strategy == "hybrid"


def test_hybrid_must_not_reuse_fixed_identity_collection() -> None:
    """§9.4 身份隔离：hybrid 指向 fixed 身份 Collection 必须报错。"""

    for name in ("ef_research_bge_m3_c512_o64_v1", "ef_bge_m3_c512_o64"):
        with pytest.raises(ValidationError):
            Settings(collection_name=name, chunk_strategy="hybrid")
    # fixed 策略可以继续使用旧 Collection（回滚通道不受影响）。
    settings = Settings(collection_name="ef_research_bge_m3_c512_o64_v1", chunk_strategy="fixed")
    assert settings.chunk_strategy == "fixed"


def test_ingestion_service_rejects_unknown_strategy() -> None:
    """IngestionService 只接受 hybrid | fixed，防御拼写错误静默走回 fixed。"""

    with pytest.raises(ValueError):
        IngestionService(
            ingestion_root=".",
            tokenizer=object(),
            embedding_provider=object(),
            vector_store=object(),
            chunk_size=512,
            chunk_overlap=64,
            chunk_strategy="hybrid2",  # type: ignore[arg-type]
        )
