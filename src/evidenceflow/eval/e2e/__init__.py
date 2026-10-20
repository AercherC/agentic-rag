"""端到端评测 rag_e2e_v1 的题集、关键要素匹配与证据映射工具（PROJECT_SPEC §13.1）。

本包属于评测层：可以依赖 runtime / storage / domain，但 runtime、retrieval 等线上模块
不得反向 import 本包（`tests/test_architecture_boundaries.py` 守护）。
"""
