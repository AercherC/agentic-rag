# Phase 15B.5 Simplification Report

## STATUS

PASS — Production behavior delta = 0。

## PRODUCTION PATH BEFORE

默认路径是 Dense Top20 → Learned Reranker → Top5，但 Composition Root 同时维护
`reranker_enabled=false` 的 Translation → Multilingual → optional BM25/RRF 回滚分支。

## PRODUCTION PATH AFTER

```text
Embedding -> Chroma -> DenseRetriever Top20 -> LearnedReranker -> Top5
          -> ContextBuilder -> Generator -> Citation -> RAGService
```

只有一条生产检索路径，无配置级旧算法分支。

## FILES DELETED

- Production/compatibility: `translation/__init__.py`、`eval/lexical.py`、`eval/hybrid.py`。
- Historical feature tests: 11 files（Translation、Multilingual、BM25/RRF/Hybrid、Phase6/8/15B
  及三组旧 research experiment tests）。
- Cache: 382 files、29 empty cache directories，8,352,699 bytes。每批均通过项目 exact
  allowlist、路径边界与删除前 hash 校验；最终 sweep 确认代码树无 cache 残留。

## FILES MOVED/ARCHIVED

- 6 个历史算法文件移至 `experiments/archive/legacy_retrieval/`。
- 26 个历史或一次性 runner 移至 `experiments/archive/`。

## MODULES MERGED

无。现有 canonical Evaluation/knowledge helper 已足够；没有为历史脚本创建新 `utils.py` 或
Cleanup framework。

## SETTINGS REMOVED

12 fields；33 → 21。详见 `architecture_audit.md`。

## DOMAIN MODELS REMOVED

`MultilingualRetrievedChunk`。正式 Context、Citation、RAGAnswer 和 RAGService 仅接受
`RetrievedChunk | RerankedRetrievedChunk`。

## TESTS REMOVED

11 个仅保护退休功能或历史 runner 的测试文件。测试总数允许下降：637 → 517；最终 0 failed。

## CACHE/TMP DELETED

382 个 pytest/bytecode cache 文件与 29 个空缓存目录，共 8,352,699 bytes。模型缓存、source PDF、Chroma、Gold、
qrels、mapping 和冻结 artifact 未进入 allowlist。清理工具现在也会删除已清空的标准缓存目录。

## RETAINED LEGACY

见 `retained_legacy.md`。核心原因是当前目录无 Git history，历史源码改为非执行 archive，
没有留在 Production。

## FULL TEST RESULT

517 passed / 0 failed / 18 warnings。

## PRODUCTION REPLAY PARITY

40Q Top5 mismatches = 0；projection hash 与全部五项指标完全一致。

## ACTIVE INDEX PARITY

PASS：`ef_research_bge_m3_hybrid_c512_v2`，10 documents，817 chunks，metadata coverage 100%，
corpus signature `bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332`，
index signature `b4e9432c7170615ade2c53803a887fa86d0f793949934a57db421f0da57739e5`。

## FROZEN HASH PARITY

PASS：Phase13F manifest files 与 Gold/qrels 四个保护文件全部通过；Phase15A Gate PASS。
另以清理前 `artifacts/cleanup/post13f/full_file_hash_manifest.json` 逐文件复算 Phase15A 与
Phase15B v1/v2：43 files checked、0 missing、0 mismatches（v1 19、v2 20、15A 4）。

## SIZE SUMMARY

- Production Python files: 69 → 60。
- Production source LOC: 6,268 after；精确 before LOC 未保留（仓库无 Git，避免伪造数字）。
- Settings fields: 33 → 21。
- Runtime retriever classes: 3 → 2（DenseRetriever、DenseRerankerRetriever）。
- Compatibility modules: 2 → 0。
- Active experiment/runner Python files: 103 → 77（26 archived）。

## KNOWN REMAINING DEBT

历史规范章节体量较大，但作为冻结实验记录保留；Current Architecture Summary 与 §41 已明确
当前合同。Archive 不可执行，不应被重新加入测试或 Production import graph。
