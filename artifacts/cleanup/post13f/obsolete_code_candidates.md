# Obsolete Python Candidates

- `src/evidenceflow/domain/evaluation_v2.py`：静态 runtime 不可达；tests=6，CLI=0，config=0，frozen=9；结论：REVIEW。
- `src/evidenceflow/eval/__init__.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=86；结论：REVIEW。
- `src/evidenceflow/eval/comparison.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=10；结论：REVIEW。
- `src/evidenceflow/eval/dataset.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=22；结论：REVIEW。
- `src/evidenceflow/eval/gold.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=10；结论：REVIEW。
- `src/evidenceflow/eval/hybrid.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=18；结论：REVIEW。
- `src/evidenceflow/eval/ir_metrics.py`：静态 runtime 不可达；tests=4，CLI=0，config=0，frozen=15；结论：REVIEW。
- `src/evidenceflow/eval/lexical.py`：静态 runtime 不可达；tests=4，CLI=0，config=0，frozen=21；结论：REVIEW。
- `src/evidenceflow/eval/mapping.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=12；结论：REVIEW。
- `src/evidenceflow/eval/metrics.py`：静态 runtime 不可达；tests=4，CLI=0，config=0，frozen=20；结论：REVIEW。
- `src/evidenceflow/eval/quality.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=12；结论：REVIEW。
- `src/evidenceflow/eval/runner.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=16；结论：REVIEW。
- `src/evidenceflow/eval/stable_mapping.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=10；结论：REVIEW。
- `src/evidenceflow/eval/v2.py`：静态 runtime 不可达；tests=6，CLI=0，config=0，frozen=16；结论：REVIEW。
- `src/evidenceflow/eval/v3.py`：静态 runtime 不可达；tests=0，CLI=0，config=0，frozen=12；结论：REVIEW。
- `src/evidenceflow/inspection.py`：静态 runtime 不可达；tests=4，CLI=0，config=0，frozen=24；结论：REVIEW。
- `src/evidenceflow/knowledge/__init__.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=0；结论：REVIEW。
- `src/evidenceflow/knowledge/job.py`：静态 runtime 不可达；tests=0，CLI=0，config=0，frozen=0；结论：REVIEW。
- `src/evidenceflow/knowledge/lifecycle.py`：静态 runtime 不可达；tests=0，CLI=0，config=0，frozen=0；结论：REVIEW。
- `src/evidenceflow/knowledge/manifest.py`：静态 runtime 不可达；tests=2，CLI=0，config=0，frozen=0；结论：REVIEW。

Python 源码本轮不删除；`unreachable` 仅表示进入下一轮人工审查。
