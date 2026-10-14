"""使用 Sentence Transformers 提供 BGE-M3 Dense Embedding。"""

import math
from collections.abc import Sequence
from typing import Any

from sentence_transformers import SentenceTransformer

from evidenceflow.model_device import resolve_model_device


class EmbeddingError(RuntimeError):
    """Embedding 模型加载、执行或输出转换失败的统一基础异常。"""


class EmbeddingLoadError(EmbeddingError):
    """表示 Sentence Transformer 模型无法加载。"""


class EmbeddingNormalizationError(EmbeddingError):
    """表示模型向量无法安全执行 L2 归一化。"""


def _normalize_vector(vector: Any) -> list[float]:
    """将第三方向量转换为有限的 Python float，并显式执行 L2 归一化。"""

    # NumPy array 提供 tolist()；测试 Fake 则可直接返回 list/tuple。
    raw_values = vector.tolist() if hasattr(vector, "tolist") else vector
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
        raise EmbeddingNormalizationError("Embedding output is not a vector")
    if not raw_values:
        raise EmbeddingNormalizationError("Embedding vector must not be empty")

    try:
        # 在 Provider 边界统一转成内置 float，避免向业务层泄漏 numpy.float32。
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError) as exc:
        raise EmbeddingNormalizationError(
            "Embedding vector contains non-numeric values"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingNormalizationError(
            "Embedding vector contains non-finite values"
        )

    norm = math.sqrt(math.fsum(value * value for value in values))
    # 零向量无法定义方向，必须明确失败，避免产生 NaN 后写入未来的 Vector Store。
    if norm == 0.0 or not math.isfinite(norm):
        raise EmbeddingNormalizationError(
            "Embedding vector has an invalid L2 norm"
        )
    return [float(value / norm) for value in values]


def _normalize_batch(raw_vectors: Any, expected_count: int) -> list[list[float]]:
    """校验批量输出数量，并逐个转换、归一化模型向量。"""

    vectors = raw_vectors.tolist() if hasattr(raw_vectors, "tolist") else raw_vectors
    if isinstance(vectors, (str, bytes)) or not isinstance(vectors, Sequence):
        raise EmbeddingNormalizationError("Embedding batch output is invalid")
    if len(vectors) != expected_count:
        raise EmbeddingNormalizationError(
            "Embedding batch size does not match input text count"
        )
    return [_normalize_vector(vector) for vector in vectors]


def _validate_text(text: str, *, field_name: str) -> None:
    """拒绝非字符串和纯空白输入，但不改变原始文本内容。"""

    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be a string")
    if not text.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace")


class BgeM3EmbeddingProvider:
    """加载一次配置指定的 BGE-M3 模型并生成归一化 Dense Vector。"""

    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        """初始化并缓存模型实例，后续文档和查询共享同一向量空间。"""

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Embedding model name must not be empty")
        self.model_name = model_name
        self.device = resolve_model_device(device)
        try:
            # 文档与 Query 共用该实例及设备；模型库默认加载 float32 权重。
            self._model = SentenceTransformer(model_name, device=self.device)
        except Exception as exc:
            raise EmbeddingLoadError(
                f"Failed to load embedding model: {model_name}"
            ) from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """一次批量编码多个文档文本，并保证所有输出显式 L2 归一化。"""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        text_list = list(texts)
        if not text_list:
            # 空批次直接返回，不调用昂贵的模型推理。
            return []
        for index, text in enumerate(text_list):
            _validate_text(text, field_name=f"texts[{index}]")

        try:
            # 底层一次接收完整 Batch；项目层随后独立保证归一化和输出类型。
            raw_vectors = self._model.encode(
                text_list,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
                device=self.device,
            )
        except Exception as exc:
            raise EmbeddingError("Failed to embed document texts") from exc
        return _normalize_batch(raw_vectors, expected_count=len(text_list))

    def embed_query(self, query: str) -> list[float]:
        """使用与文档相同的模型生成单个归一化查询向量。"""

        _validate_text(query, field_name="query")
        try:
            # 不执行 strip、改写或翻译，校验后的原始 Query 原样交给模型。
            raw_vector = self._model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
                device=self.device,
            )
        except Exception as exc:
            raise EmbeddingError("Failed to embed query") from exc
        return _normalize_vector(raw_vector)
