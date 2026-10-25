"""Session-level test configuration: skip real-model integration tests when there is no local
model cache.
"""

from __future__ import annotations

import os

import pytest

from evidenceflow.config import Settings

# Set to 1 to force requires_model tests to run (allows online model downloads).
RUN_MODEL_TESTS_ENV = "EVIDENCEFLOW_RUN_MODEL_TESTS"


def _model_is_cached(repo_id: str) -> bool:
    """Check only the local Hugging Face cache (honoring HF_HOME); never use the network."""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark requires_model tests as skipped when no model is cached, instead of failing offline."""

    if os.environ.get(RUN_MODEL_TESTS_ENV) == "1":
        return
    model_name = Settings(_env_file=None).embedding_model
    if _model_is_cached(model_name):
        return
    skip = pytest.mark.skip(
        reason=(
            f"{model_name} is not in the local Hugging Face cache; "
            f"set {RUN_MODEL_TESTS_ENV}=1 to download and run it"
        )
    )
    for item in items:
        if "requires_model" in item.keywords:
            item.add_marker(skip)
