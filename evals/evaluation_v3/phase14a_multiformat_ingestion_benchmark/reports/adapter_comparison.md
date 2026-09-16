# Phase 14A-A Adapter Comparison

- **Status**: `PASS`
- **PDF Prerequisite**: `FROZEN` — 13C-B PASS，B2_fused PDF ingestion 契约已冻结，作为 Phase 14A 的已冻结 PDF 参考 adapter。13C-C chunk benchmark 仍为草案未执行——按用户决策，14A-A 范围（DOCX/XLSX adapter）不涉及 PDF 重切分，无三变量混淆风险，故直接启动。
- **Sandbox-Only**: YES  |  **Local Project Modified**: NO
- **Fake Provenance**: 0  |  **Formula Silent Overwrite**: 0  |  **Merged Range Loss**: 0
- **Blocked Arms**: W-A1, W-A2, W-A3, X-A1, X-A2

## DOCX Arms
| Arm | Status | Cases Pass All Gates |
|---|---|---|
| W-A0 | RUNNABLE | 8/8 |
| W-A1 | BLOCKED | — |
| W-A2 | BLOCKED | — |
| W-A3 | BLOCKED | — |

**DOCX Hard Gates**: 8/8 cases pass all W-G1..W-G8

### Per-case gate results (DOCX)
| Case | All Gates Pass |
|---|---|
| W1 | True |
| W2 | True |
| W3 | True |
| W4 | True |
| W5 | True |
| W6 | True |
| W7 | True |
| W8 | True |

## XLSX Arms
| Arm | Status | Cases Pass All Gates |
|---|---|---|
| X-A0 | RUNNABLE | 10/10 |
| X-A1 | BLOCKED | — |
| X-A2 | BLOCKED | — |

**XLSX Hard Gates**: 10/10 cases pass all X-G1..X-G8

### Per-case gate results (XLSX)
| Case | All Gates Pass |
|---|---|
| X1 | True |
| X2 | True |
| X3 | True |
| X4 | True |
| X5 | True |
| X6 | True |
| X7 | True |
| X8 | True |
| X9 | True |
| X10 | True |

## Corpus Self-Check
- DOCX passing: 7/8
- XLSX passing: 6/10
- All passed: False