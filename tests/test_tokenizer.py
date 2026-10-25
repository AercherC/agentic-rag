"""Verify the minimal interface and special-token constraints of the BGE-M3 Tokenizer adapter."""

import pytest

from evidenceflow.config import Settings
from evidenceflow.ingestion.tokenizer import (
    BgeM3Tokenizer,
    TokenizerError,
    TokenizerLoadError,
)


class _RecordingBackend:
    """Record the Transformers call arguments so unit tests never touch the network."""

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
    """The model name must come from the caller, and body counting must explicitly exclude special
    tokens."""

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
    """Third-party loading errors and contract-violating outputs should be converted into project
    exceptions."""

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
    """Load the real BGE-M3 Tokenizer from the default Settings without loading model weights."""

    settings = Settings(_env_file=None)
    return BgeM3Tokenizer(settings.embedding_model)


@pytest.mark.requires_model
@pytest.mark.parametrize("text", ["Evidence must remain traceable.", "证据必须保持可追溯。"])
def test_real_bge_m3_tokenizer_handles_english_and_chinese(
    real_bge_m3_tokenizer: BgeM3Tokenizer,
    text: str,
) -> None:
    """The real BGE-M3 Tokenizer should handle English and Chinese body text reliably."""

    token_ids = real_bge_m3_tokenizer.encode(text)
    decoded = real_bge_m3_tokenizer.decode(token_ids)

    assert token_ids
    assert all(type(token_id) is int for token_id in token_ids)
    assert decoded.strip()
    # Cross-check against the underlying interface again, so the body window never accidentally
    # includes the model's leading/trailing special tokens.
    assert token_ids == real_bge_m3_tokenizer._tokenizer.encode(
        text, add_special_tokens=False
    )
    assert len(
        real_bge_m3_tokenizer._tokenizer.encode(text, add_special_tokens=True)
    ) > len(token_ids)
