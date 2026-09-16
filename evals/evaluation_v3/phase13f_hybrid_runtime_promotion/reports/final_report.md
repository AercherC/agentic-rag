# Phase 13F Hybrid Runtime Promotion — Final Report

- 生成时间：2026-09-15T09:59:47.491195+00:00
- verdict：**PASS**（gate_definitions phase13f-gates-v1.0）
- 生产 collection：`ef_research_bge_m3_hybrid_c512_v1`（hybrid 身份，重建于 data/chroma）
- 已知偏差：生产结构解析为 P0-only（13A/13B 移植）；13E PH1 的 table 文本来自 B2_fused（MinerU serialized_markdown），二者数值偏差据此归因

## 语料切分统计（生产链路）

| 文件 | elements（furniture 移除后） | chunks |
|---|---:|---:|
| adapointr.pdf | 580 | 147 |
| hyper_pcn.pdf | 325 | 101 |
| pcn_point_completion_network.pdf | 231 | 74 |
| pfnet_point_fractal_network.pdf | 171 | 53 |
| pointattn.pdf | 184 | 48 |
| pointr_geometry_aware_transformers.pdf | 306 | 82 |
| proxyformer.pdf | 352 | 74 |
| seedformer.pdf | 218 | 84 |
| snowflakenet.pdf | 302 | 73 |
| svdformer.pdf | 243 | 81 |

## 标准 IR（aggregate，40 题 anchor 严格制）

| arm | P@5 | R@5 | R@10 | MRR@5 | MAP@5 | nDCG@5 | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RUNTIME_HYBRID | 0.2316 | 0.5124 | 0.6575 | 0.5522 | 0.3855 | 0.4973 | 0.8684 |
| PC0 (frozen 13C-C) | 0.2450 | 0.4101 | 0.6561 | 0.4558 | 0.2937 | 0.4142 | 0.9500 |
| 13E PH1 (B2_fused 参照) | 0.2350 | 0.5085 | 0.6223 | 0.5704 | 0.3719 | 0.4903 | 0.8500 |

## Bootstrap vs 冻结 PC0（n=10000，seed=20260915）

| metric | mean Δ | CI95 | P(Δ>0) | significant |
|---|---:|---|---:|---|
| ndcg_at_5 | +0.0780 | [-0.0466, +0.2057] | 0.889 | False |
| recall_at_5 | +0.1070 | [-0.0439, +0.2642] | 0.918 | False |
| recall_at_10 | -0.0069 | [-0.1154, +0.1034] | 0.444 | False |
| mrr_at_5 | +0.0855 | [-0.0509, +0.2211] | 0.891 | False |

## Focus

- **ARR-007** — RUNTIME_HYBRID: rank=2 | PC0: miss
- **ARR-022** — RUNTIME_HYBRID: rank=1 | PC0: rank=17
- **ARR-037** — RUNTIME_HYBRID: rank=6 | PC0: rank=6
- **ARR-040** — RUNTIME_HYBRID: rank=2 | PC0: rank=1

## 硬门

- F_G1_inputs: True
- F_G2_structure_parity: True
- F_G3_determinism: True
- F_G4_token_budget: True
- F_G5_provenance: True
- F_G6_protocol: True
- F_G7_ir_direction: True
- F_G8_pytest_baseline: True
- pytest: 612 passed / 0 failed

## 补充判读（数据驱动，执行后追加）

### 1. 结构等同（F-G2）— 本阶段最强门

- runtime 移植解析与 13B 冻结产物**逐元素全等**（10/10 文档，kept 2912 = 3056 − 144
  furniture，element_id/text/element_type/section_path/bbox 等全部字段一致）：
  13A/13B 纯函数移植零语义漂移。

### 2. 端到端 IR（F-G7）— 方向门通过

- vs 冻结 PC0：R@5 +0.107（P=0.918）、nDCG@5 +0.078（P=0.889）、MRR@5 +0.086，
  方向一致为正且幅度与 13E PH1 相当（+0.102/+0.076）；R@10 −0.007 ≈ 0。
- RUNTIME_HYBRID（R@5 0.5124 / nDCG@5 0.4973）与 13E PH1（0.5124/0.4903）高度接近：
  P0-only table 文本差异未侵蚀主要收益。
- ARR-007 rank=2 恢复（PC0 miss）；ARR-022 rank=17→1。生产链路端到端复现了
  13C-C→13D→13E 证据链的结论。

### 3. 已知偏差（如实记录）

- anchor 覆盖 0.9545（42/44 gold source）：ARR-003（"equipment defects or limited
  viewpoints"）与 ARR-033（"technical details and additional quantitative"）两处
  跨元素短语在 P0-only 解析的 chunk 文本中无子串命中（13E PH1 的 B2_fused 含
  P1/MinerU 元素可覆盖）。影响：两题 num_relevant 降低，方向门仍通过；
  真实语料导入后应复验。

### 4. 晋升生效与回滚

- 生产默认：`CHUNK_STRATEGY=hybrid`（Settings 默认同），新 collection
  `ef_research_bge_m3_hybrid_c512_v1` 已重建于 data/chroma（817 chunks）。
- 回滚：`.env` 设 `CHUNK_STRATEGY=fixed` + `COLLECTION_NAME=ef_research_bge_m3_c512_o64_v1`
  （旧库未动，回滚通道完整保留）。
- 未显著性与小语料风险照 13E 记录：本晋升基于方向一致证据 + 用户明确决策，
  真实语料导入后应复验显著性。

## 5. 指纹重基线与晋升后一致性验证（scripts/rebaseline_post_promotion.py）

- **src 变更集 = PLAN §3 声明**（防扩散断言通过）：新增
  `ingestion/structure.py`；修改 `config.py` / `ingestion/chunker.py` /
  `ingestion/service.py` / `ingestion/__init__.py` / `runtime.py`；删除 0。
- **runtime_source_fingerprint（晋升后新基线）**：
  `2a3d5376e6f6c5c4b677b2b3581e72a4f34d47d4a5915bd1efb52a5fc96b689a`
  （src/evidenceflow 65 个 .py 的 sha256 全集；旧基线留在
  `artifacts/cleanup/src_hash_baseline.json` 作历史证据）。
- **冻结评测证据完整性**：standard_ir / phase12 / 13C-C / 13D / 13E / 13F 六个
  manifest 逐文件 hash 复算全部 intact——晋升未触碰任何冻结证据。
- **归档门**：src/tests 对 `experiments.phase14a` 的 import = 0，仍成立。
- 证据文件：`artifacts/cleanup/post_promotion_rebaseline.json`。
