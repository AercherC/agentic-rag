"""Phase 13C — Scientific Ingestion Benchmark（Sandbox 专用工具包）。

本包实现 Phase 13C-A/B/C 三轮 benchmark 的公共基础设施：

- :mod:`phase13c.sandbox`  —— 沙盒路径、哈希、边界守卫（STOP-ON-ANOMALY）
- :mod:`phase13c.schema`   —— 统一 ScientificDocument element schema 与校验
- :mod:`phase13c.align`    —— Ground Truth ↔ parser 输出的元素对齐引擎
- :mod:`phase13c.metrics`  —— 11 族 parser/cleaning/chunk 评测指标（纯函数）
- :mod:`phase13c.corpus`   —— 受控模板语料生成与元素级 Ground Truth
- :mod:`phase13c.adapters` —— P0–P4 parser 适配器（统一 schema 归一化）

设计约束（PROJECT_SPEC §37 / Phase 13C 方案 §4）：

- **本包不得被 Runtime / src 引用**，只属于实验空间。
- 所有写操作都落在 sandbox 根目录；不允许写 ``data/``、``src/``、
  ``PROJECT_SPEC.md``、canonical qrels 或任何 Chroma 目录。
"""

from __future__ import annotations

__all__ = ["__version__"]

# 记录本包对应的阶段实现版本，便于产物溯源。
__version__ = "phase13c-1.0.0"
