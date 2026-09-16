# Historical experiment source

本目录保存 Phase 15B.5 退休的实验 runner 与 Retrieval 算法源码。它们只用于阅读历史实现，
不再是可执行入口，也不受当前 Production 测试合同维护。

正式运行时固定为 `Dense Top20 -> Learned Reranker -> Top5`。历史 Translation、BM25、
Multilingual 与 RRF 结果由 `evals/` 中的冻结产物保存；当前目录不是 rollback runtime。
