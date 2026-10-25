"""Verify the chunk_strategy configuration guard and the IngestionService strategy dispatch contract (Phase 13F)."""

import pytest
from pydantic import ValidationError

from evidenceflow.config import Settings
from evidenceflow.ingestion.service import IngestionService


def test_chunk_strategy_defaults_to_hybrid() -> None:
    """Phase 13F promotion: the default strategy is hybrid."""

    settings = Settings(llm_api_key=None, collection_name="ef_test_hybrid")
    assert settings.chunk_strategy == "hybrid"


def test_hybrid_must_not_reuse_fixed_identity_collection() -> None:
    """§9.4 identity isolation: hybrid pointing at a fixed-identity Collection must raise."""

    for name in ("ef_research_bge_m3_c512_o64_v1", "ef_bge_m3_c512_o64"):
        with pytest.raises(ValidationError):
            Settings(collection_name=name, chunk_strategy="hybrid")
    # The fixed strategy may keep using the old Collection (the rollback channel is unaffected).
    settings = Settings(collection_name="ef_research_bge_m3_c512_o64_v1", chunk_strategy="fixed")
    assert settings.chunk_strategy == "fixed"


def test_ingestion_service_rejects_unknown_strategy() -> None:
    """IngestionService accepts only hybrid | fixed, preventing a typo from silently falling back to fixed."""

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
