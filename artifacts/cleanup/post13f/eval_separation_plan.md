# Eval Separation Plan

当前 `src/evidenceflow/eval/**` 未进入默认 Runtime 主路径，但被 tests、experiments 或冻结评测消费。

本轮不物理搬动；未来只有在 all imports、regression tests、frozen reproduction 均不受影响时，才评估迁移到 `evals/evidenceflow_eval/`。

盘点文件：
- `src/evidenceflow/eval/__init__.py`
- `src/evidenceflow/eval/comparison.py`
- `src/evidenceflow/eval/dataset.py`
- `src/evidenceflow/eval/gold.py`
- `src/evidenceflow/eval/hybrid.py`
- `src/evidenceflow/eval/ir_metrics.py`
- `src/evidenceflow/eval/lexical.py`
- `src/evidenceflow/eval/mapping.py`
- `src/evidenceflow/eval/metrics.py`
- `src/evidenceflow/eval/quality.py`
- `src/evidenceflow/eval/runner.py`
- `src/evidenceflow/eval/stable_mapping.py`
- `src/evidenceflow/eval/v2.py`
- `src/evidenceflow/eval/v3.py`
