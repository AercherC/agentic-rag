"""开发者 Retrieval Inspector：定性查看真实 Dense 结果。

本工具用于 Debug，不能替代固定 Gold Dataset 上的 Recall/MRR Evaluation。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 脚本位于 experiments/debug，项目根目录需向上两层。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# 纯格式化函数统一由 src 提供，debug 脚本与单元测试共用同一份实现。
from evidenceflow.inspection import format_retrieval_result


def _build_dense_retriever():
    """使用项目缓存和当前 Settings 构建真实 C512 DenseRetriever。"""

    from evidenceflow.config import Settings
    from evidenceflow.embedding import BgeM3EmbeddingProvider
    from evidenceflow.retrieval import DenseRetriever
    from evidenceflow.storage import ChromaVectorStore

    os.environ["HF_HOME"] = str((PROJECT_ROOT / ".cache/huggingface").resolve())
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    return settings, DenseRetriever(
        BgeM3EmbeddingProvider(settings.embedding_model, device=settings.model_device),
        store,
        top_k=5,
    )


def main() -> None:
    """读取命令行或交互 Query，并打印可审计的真实检索结果。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument("--full-content", action="store_true")
    args = parser.parse_args()
    query = args.query if args.query is not None else input("请输入 Query: ").strip()
    settings, dense = _build_dense_retriever()
    top_k = args.top_k
    print(f"Query: {query}")
    print("Mode: Dense")
    print(f"Collection: {settings.collection_name}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Top-K: {top_k}")
    results = dense.retrieve(query, top_k=top_k)
    for result in results:
        print(
            "\n---\n"
            + format_retrieval_result(
                result,
                preview_chars=args.preview_chars,
                full=args.full_content,
            )
        )


if __name__ == "__main__":
    main()
