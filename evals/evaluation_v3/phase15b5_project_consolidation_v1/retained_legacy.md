# Retained Legacy

## Non-executable archive

当前目录不是 Git 工作区，无法满足“由 Git history 恢复旧源码”的删除条件。因此 6 个历史
Retrieval/Translation 实现与 26 个历史/一次性实验脚本移动到 `experiments/archive/`。

这些文件：

- 不在 `src/evidenceflow/`；
- 不被 Production、测试或当前实验导入；
- 不承诺可执行，允许引用已退休的历史接口；
- 只用于阅读历史实现和在未来迁入真正版本库前保留来源。

## Data/index legacy retained

- `ef_research_bge_m3_hybrid_c512_v1`：按既有合同保持只读，不删除、不切回。
- Phase15A、Phase15B v1/v2、Phase13F、Gold、qrels、mapping/projection：冻结证据，hash Gate 通过。
- Active `ef_research_bge_m3_hybrid_c512_v2`：10 documents / 817 chunks，未修改。

## Accepted current debt

- 历史章节仍保留在 `PROJECT_SPEC.md`；顶部 Current Architecture Summary 与 §41 明确覆盖其
  Runtime/Settings 语义，避免破坏历史证据。
- `experiments/archive/` 不是可运行的 reproduction package。若未来需要重跑旧算法，应从正式
  版本历史恢复并单独立项，而不是把兼容层放回 Production。
