"""为文档、文件内容和 Chunk 生成确定性标识。"""

import hashlib
import json
import os
import posixpath
import unicodedata
from os import PathLike


def _sha256(data: bytes) -> str:
    """返回固定 64 位小写十六进制 SHA-256 摘要。"""

    return hashlib.sha256(data).hexdigest()


def _normalize_relative_path(relative_path: str | PathLike[str]) -> str:
    """将 ingestion root 内的相对路径规范化为跨平台稳定形式。"""

    # NFC 避免视觉相同但编码形式不同的 Unicode 路径产生不同 ID；
    # POSIX 分隔符保证 Windows 与 Unix 上的相同相对路径具有同一身份。
    path = unicodedata.normalize("NFC", os.fspath(relative_path)).replace("\\", "/")
    # 消除重复分隔符和当前目录片段，得到唯一的词法路径表示。
    normalized = posixpath.normpath(path)
    # 空路径或当前目录不能指向一个可导入的具体文件。
    if normalized in {"", "."}:
        raise ValueError("relative_path must identify a file")
    # document_id 只能基于 ingestion root 内部路径，禁止绝对路径和向上逃逸。
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise ValueError("relative_path must stay within the ingestion root")
    return normalized


def build_document_id(relative_path: str | PathLike[str]) -> str:
    """根据规范化相对路径生成稳定的 SHA-256 文档 ID。

    文件内容不参与计算，因此同一路径的文件更新后仍保留原逻辑身份。
    """

    # 必须先统一 Unicode 和路径分隔符，再编码为 UTF-8 参与 Hash。
    normalized_path = _normalize_relative_path(relative_path)
    return _sha256(normalized_path.encode("utf-8"))


def build_content_hash(file_bytes: bytes) -> str:
    """对原始文件 bytes 计算 Hash，用于识别逻辑文档的内容变化。"""

    # 只接受 bytes，避免调用方因隐式文本编码不同而得到不稳定结果。
    if not isinstance(file_bytes, bytes):
        raise TypeError("file_bytes must be bytes")
    return _sha256(file_bytes)


def build_chunk_id(
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_index: int,
    content: str,
) -> str:
    """根据文档、切分配置、位置和内容生成稳定 Chunk ID。

    使用具名字段及固定键顺序序列化，使输入边界明确且结果可重复。
    """

    # Chunk ID 使用内容 Hash 而不是直接拼接正文，保持身份载荷长度固定。
    content_hash = _sha256(content.encode("utf-8"))
    # 所有影响 Chunk 身份的输入都必须进入载荷；任一项变化都会产生新 ID。
    identity = {
        "document_id": document_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_index": chunk_index,
        "content_hash": content_hash,
    }
    # 固定键顺序并移除无意义空白，得到唯一、无分隔歧义的字节表示。
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return _sha256(serialized.encode("utf-8"))
