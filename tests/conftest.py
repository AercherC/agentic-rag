"""测试会话级配置：没有本地模型缓存时跳过依赖真实模型的集成测试。"""

from __future__ import annotations

import os

import pytest

from evidenceflow.config import Settings

# 设为 1 时强制运行 requires_model 用例（允许在线下载模型）。
RUN_MODEL_TESTS_ENV = "EVIDENCEFLOW_RUN_MODEL_TESTS"


def _model_is_cached(repo_id: str) -> bool:
    """只查本地 Hugging Face 缓存（遵循 HF_HOME），不发任何网络请求。"""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """模型未缓存时把 requires_model 用例标记为 skip，而不是在离线环境报错。"""

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
