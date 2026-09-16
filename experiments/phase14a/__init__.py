"""Phase 14A — Multi-Format Structured Ingestion Adapter Benchmark (Sandbox-Only).

本包是 Phase 13C 冻结基座（experiments/phase13c 的 sandbox/gate/adapter 模式）的**平行复用**，
不修改 src/evidenceflow/**，不覆盖现有 Chroma，不改 canonical qrels，不改源文件，不向现有项目环境安装依赖。

Phase 14A-A 范围：DOCX / XLSX adapter benchmark（原生 baseline + 跨格式候选），然后 STOP。
PDF 作为已冻结参考 adapter（13C-B B2_fused），本阶段不再重跑 PDF parser competition。
"""
