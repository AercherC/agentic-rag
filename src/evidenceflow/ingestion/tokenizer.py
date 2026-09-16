"""为 Chunker 提供与 Hugging Face 实现解耦的 Tokenizer 接口。"""

from typing import Protocol, runtime_checkable

from transformers import AutoTokenizer


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Chunker 所需的最小 Tokenizer 合同。"""

    def encode(self, text: str) -> list[int]:
        """将正文编码为不含模型特殊 Token 的 ID 序列。"""

        ...

    def decode(self, token_ids: list[int]) -> str:
        """将正文 Token IDs 解码为文本。"""

        ...


class TokenizerError(RuntimeError):
    """Tokenizer 加载、编码或解码失败的统一基础异常。"""


class TokenizerLoadError(TokenizerError):
    """表示 Hugging Face Tokenizer 无法加载。"""


class BgeM3Tokenizer:
    """基于配置模型名称加载 BGE-M3 兼容的 Hugging Face Tokenizer。"""

    def __init__(self, model_name: str) -> None:
        """仅加载 Tokenizer 文件，不加载模型权重或初始化 GPU。"""

        if not model_name.strip():
            raise ValueError("Tokenizer model name must not be empty")
        self.model_name = model_name
        try:
            # 模型名由 Settings.embedding_model 注入，避免 Adapter 重复硬编码。
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise TokenizerLoadError(
                f"Failed to load tokenizer for model: {model_name}"
            ) from exc

    def encode(self, text: str) -> list[int]:
        """将文本编码为正文 Token IDs，不计入模型 special tokens。"""

        try:
            # chunk_size 的合同是正文 Token 数，因此必须显式关闭自动特殊 Token。
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            raise TokenizerError("Tokenizer failed to encode text") from exc
        if not isinstance(token_ids, list) or not all(
            type(token_id) is int for token_id in token_ids
        ):
            raise TokenizerError("Tokenizer encode returned invalid token IDs")
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """解码正文 Token IDs，并关闭额外的空格清理。"""

        try:
            # 不让 Transformers 再做空格格式化，Chunker 只允许外围 strip()。
            text = self._tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise TokenizerError("Tokenizer failed to decode token IDs") from exc
        if not isinstance(text, str):
            raise TokenizerError("Tokenizer decode returned non-text output")
        return text
