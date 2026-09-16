# Academic Research v1 Stage Summary

## 1. Corpus freeze and knowledge base

- 扫描来源：`D:\文献`，共发现 107 个 PDF；来源目录未移动、未重命名、未修改。
- 选择领域：Point Cloud Completion / 3D Vision，共 10 篇。
- Active Collection：`ef_research_bge_m3_c512_o64_v1`。
- Pipeline：当前 PyMuPDF Loader → 当前 Cleaner → BGE-M3 C512/O64 → Chroma；核心实现未修改。
- Corpus hash：`1e48b401aa48fc59d94bba627deaa3005a7627860e4fe5815761d45c6ba68675`。
- Manifest hash：`efd92f9de8ac30e6a1081d1641076a9166b18f671238b5d44f89f6727423bb16`。

| Paper | Year / venue | Pages | Chunks | Selection role |
|---|---|---:|---:|---|
| AdaPoinTr | 2023 / TPAMI | 22 | 75 | 长篇期刊、详细表格与 Appendix |
| Hyper-PCN | 2026 / CVPR | 17 | 33 | Hypergraph、附录与丰富可视化 |
| PCN | 2018 / 3DV | 17 | 34 | 经典 coarse-to-fine 与补充材料 |
| PF-Net | 2020 / CVPR | 9 | 25 | 紧凑双栏、公式与多阶段损失 |
| PointAttN | 2024 / AAAI | 12 | 25 | Attention 架构、消融与密集引用 |
| PoinTr | 2021 / ICCV | 14 | 39 | Transformer 代表方法与结果表 |
| ProxyFormer | 2023 / CVPR | 10 | 31 | 标准双栏、proxy alignment 与表格 |
| SeedFormer | 2022 / ECCV | 22 | 39 | 长文、明确 Supplementary 边界 |
| SnowflakeNet | 2021 / ICCV | 11 | 31 | 复杂模块名、公式与消融 |
| SVDFormer | 2023 / ICCV | 11 | 29 | 多视图、双生成器与复杂表格 |
| **Total** |  | **145** | **361** | Average 36.1；min 25；max 75 |

Extraction failure：0；空页：0；49 个代表页面已完成视觉抽查。

## 2. Academic ingestion compatibility audit

| Issue | Affected | Severity / reading |
|---|---:|---|
| Extraction failure | 0/10 | None |
| Double-column reading order | 0/10 | 代表页未发现 High issue |
| Broken hyphenation | 10/10 | Medium |
| Section metadata missing | 10/10 | Medium；标题文本存在但无结构字段 |
| Header/footer repetition | 2/10 | Low |
| Table structure loss | 10/10 | High；行列被线性化 |
| Figure text noise | 10/10 | Low |
| Formula fragmentation | 10/10 | Medium |
| References pollution risk | 10/10 | Medium |
| Supplementary boundary lost | 5/5 | Medium |
| Chunk boundary crossing | 10/10 | Medium；每篇均有正文/References 跨界 Chunk |

当前 Loader 不需要修改。Cleaner 有 dehyphenation 实验候选；Chunker 有 section-aware 实验候选；metadata 有 `section_type` / supplementary boundary 扩展候选，但本阶段均未实现。

## 3. Frozen Research Retrieval Dataset

- Question count：40。
- Dataset hash：`be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`。
- Gold：`document_id + 1-based page_range + evidence_anchor`，不含 `chunk_id`。
- Language groups：English→English 18；Chinese→English 18；Mixed / multiple sources 4。
- Question types：Motivation 6；Method 8；Implementation 5；Experiment 5；Results 6；Ablation 4；Conclusion 2；Multi-document 4。
- Evidence types：Body 22；Figure caption 4；Table 6；Supplementary 4；Multi-document 4。
- 所有锚点先在原始 PDF 中验证并冻结，再进行第一次检索。

## 4. Current Dense baseline

配置：Original Dense Top-10 + frozen English Translation Dense Top-10 + Equal RRF (`k=60`, weights `1.0/1.0`)；候选并集最多 20，正式 Runtime Top-5。未运行 BM25、Hybrid、Reranker、Query Rewrite 或参数搜索。

| Metric | Overall |
|---|---:|
| Hit@1 | 0.175 |
| Hit@3 | 0.350 |
| Hit@5 | 0.475 |
| Hit@10 | 0.650 |
| Hit@20 | 0.650 |
| MRR | 0.302 |
| nDCG@20 | 0.404 |

| Breakdown | N | Hit@5 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| English→English | 18 | 0.389 | 0.611 | 0.250 |
| Chinese→English | 18 | 0.667 | 0.833 | 0.421 |
| Mixed / multiple sources | 4 | 0.000 | 0.000 | 0.000 |
| Body text | 22 | 0.500 | 0.773 | 0.330 |
| Table | 6 | 0.500 | 0.667 | 0.288 |
| Figure caption | 4 | 0.750 | 0.750 | 0.396 |
| Supplementary | 4 | 0.500 | 0.500 | 0.375 |
| Multi-document | 4 | 0.000 | 0.000 | 0.000 |

多文档题 nDCG@20 为 0.198，说明当前系统能召回部分来源，但 4 题均无法在同一候选并集中覆盖全部 Gold 来源。

## 5. Failure diagnosis

| Primary outcome | Count |
|---|---:|
| Success within Top-5 | 19 |
| Extraction failure | 0 |
| Chunk failure | 2 |
| Candidate retrieval failure | 12 |
| Ranking failure (rank 6–10) | 7 |

失败题的 academic-specific 诊断标签：

| Label | Count |
|---|---:|
| Reference pollution | 11 |
| Table structure loss | 3 |
| Section boundary | 2 |
| Supplement boundary | 2 |
| Figure semantics missing | 1 |
| Formula representation | 0 |

全体 40 题中，22 题的 Top-5 出现至少一个 References Chunk；全部 200 个 Top-5 槽位中共有 32 个 References Chunk。标签只表示与失败共现的可审计线索，不自动宣称单一因果。

## 6. Stage decisions

- Loader：不修改；0 extraction failure，代表页双栏顺序可用。
- Cleaner：确有 dehyphenation 候选，但本阶段不修改。
- Chunker：2 个严格 Chunk failure 且存在普遍跨 section 现象；保留实验候选，不直接改 Runtime。
- Metadata：需要研究 `section_type=references`，这是当前最直接的结构缺口。
- Table handling：表格结构确实丢失，但 Table Hit@5 与 Body 同为 0.500；暂无证据立即引入复杂 Table Parser。
- References handling：有实际检索证据，值得进入下一隔离实验。
- Multimodal：无实际证据支持现在引入；当前继续定位为 text-centric Academic RAG。

**唯一推荐的下一实验：References-aware metadata / filtering experiment。**

在不改变冻结 10 篇语料、40 题 Gold、Embedding、C512/O64 和 Equal RRF 的前提下，标注 `section_type=references`，比较 baseline、默认排除或降权 References 的 Hit@K/MRR，并核查是否伤害需要引用背景的题目。选择它的依据是 22/40 Top-5 已出现 References，且与 11 个失败题共现；其检索证据比立即上 Table Parser 或 Multimodal 更直接。

## 7. Cleanup, contracts and tests

- 旧金融 PDF、金融评测目录、金融专用源码/脚本/测试、旧金融 Chroma Collection 已按用户要求永久从项目内删除；项目内无备份，`D:\文献` 原件未受影响。
- 当前 `data/source_docs` 恰好只有上述 10 篇论文；Chroma 恰好只有 Research Collection 的 361 chunks。
- Runtime：未修改。
- `PROJECT_SPEC.md`：保持 v0.14，未修改。
- Targeted tests：5 passed。
- Full pytest：272 passed，15 warnings；warning 均为 Chroma legacy embedding configuration deprecation。

本阶段到此停止，不自动开始下一实验。
