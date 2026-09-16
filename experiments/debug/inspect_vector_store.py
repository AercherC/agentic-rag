"""开发者 Storage Inspector：只读查看 Chroma 中的 Chunk 文本与 Metadata。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 移入 debug 后仍以项目根目录解析源码与数据路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# 纯格式化函数统一由 src 提供，debug 脚本与单元测试共用同一份实现。
from evidenceflow.inspection import format_storage_record, select_records


def main() -> None:
    """打开当前 Collection，并只读打印有限数量的存储记录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--file", dest="file_name")
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument("--full-content", action="store_true")
    args = parser.parse_args()

    from evidenceflow.config import Settings
    from evidenceflow.storage import ChromaVectorStore

    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    raw = store._collection.get(include=["documents", "metadatas"])
    records = select_records(
        raw.get("ids", []),
        raw.get("documents", []),
        raw.get("metadatas", []),
        limit=args.limit,
        file_name=args.file_name,
    )
    print(f"Collection: {settings.collection_name}")
    print(f"Total chunks: {store._collection.count()}")
    print("Embeddings: not printed")
    for record in records:
        print("\n---\n" + format_storage_record(record, preview_chars=args.preview_chars, full=args.full_content))


if __name__ == "__main__":
    main()
