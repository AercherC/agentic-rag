"""验证 BGE-M3 Tokenizer 适配器的最小接口与特殊 Token 约束。"""

import pytest

from evidenceflow.config import Settings
from evidenceflow.ingestion.tokenizer import (
    BgeM3Tokenizer,
    TokenizerError,
    TokenizerLoadError,
)


class _RecordingBackend:
    """记录 Transformers 调用参数，避免单元测试访问网络。"""

    def __init__(self) -> None:
        self.encode_kwargs: dict[str, object] = {}
        self.decode_kwargs: dict[str, object] = {}

    def encode(self, text: str, **kwargs: object) -> list[int]:
        self.encode_kwargs = kwargs
        return [11, 12] if text else []

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        self.decode_kwargs = kwargs
        return "测试文本"


def test_adapter_uses_configured_model_and_excludes_special_tokens(monkeypatch) -> None:
    """模型名必须来自调用方，正文计数必须显式排除 special tokens。"""

    backend = _RecordingBackend()
    loaded_models: list[str] = []

    def fake_from_pretrained(model_name: str) -> _RecordingBackend:
        loaded_models.append(model_name)
        return backend

    monkeypatch.setattr(
        "evidenceflow.ingestion.tokenizer.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )
    tokenizer = BgeM3Tokenizer("configured/bge-m3")

    assert tokenizer.encode("正文") == [11, 12]
    assert tokenizer.decode([11, 12]) == "测试文本"
    assert loaded_models == ["configured/bge-m3"]
    assert backend.encode_kwargs["add_special_tokens"] is False
    assert backend.decode_kwargs == {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }


def test_adapter_wraps_loading_and_invalid_backend_outputs(monkeypatch) -> None:
    """第三方加载错误和不符合合同的输出应转换为项目异常。"""

    def fail_loading(model_name: str) -> None:
        raise OSError(model_name)

    monkeypatch.setattr(
        "evidenceflow.ingestion.tokenizer.AutoTokenizer.from_pretrained",
        fail_loading,
    )
    with pytest.raises(TokenizerLoadError):
        BgeM3Tokenizer("missing/model")

    class InvalidBackend(_RecordingBackend):
        def encode(self, text: str, **kwargs: object) -> tuple[int, ...]:
            return (1, 2)

        def decode(self, token_ids: list[int], **kwargs: object) -> int:
            return 123

    monkeypatch.setattr(
        "evidenceflow.ingestion.tokenizer.AutoTokenizer.from_pretrained",
        lambda model_name: InvalidBackend(),
    )
    tokenizer = BgeM3Tokenizer("invalid/model")
    with pytest.raises(TokenizerError):
        tokenizer.encode("text")
    with pytest.raises(TokenizerError):
        tokenizer.decode([1])


@pytest.fixture(scope="module")
def real_bge_m3_tokenizer() -> BgeM3Tokenizer:
    """从默认 Settings 加载真实 BGE-M3 Tokenizer，不加载模型权重。"""

    settings = Settings(_env_file=None)
    return BgeM3Tokenizer(settings.embedding_model)


@pytest.mark.requires_model
@pytest.mark.parametrize("text", ["Evidence must remain traceable.", "证据必须保持可追溯。"])
def test_real_bge_m3_tokenizer_handles_english_and_chinese(
    real_bge_m3_tokenizer: BgeM3Tokenizer,
    text: str,
) -> None:
    """真实 BGE-M3 Tokenizer 应能稳定处理英文和中文正文。"""

    token_ids = real_bge_m3_tokenizer.encode(text)
    decoded = real_bge_m3_tokenizer.decode(token_ids)

    assert token_ids
    assert all(type(token_id) is int for token_id in token_ids)
    assert decoded.strip()
    # 再次对照底层接口，防止正文窗口意外计入模型头尾特殊 Token。
    assert token_ids == real_bge_m3_tokenizer._tokenizer.encode(
        text, add_special_tokens=False
    )
    assert len(
        real_bge_m3_tokenizer._tokenizer.encode(text, add_special_tokens=True)
    ) > len(token_ids)
