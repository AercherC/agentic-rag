# EvidenceFlow Phase 15B.1 Dense Candidate Depth Confirmation

本实验只改变 Dense candidate K=20/30/40；未修改 Production Runtime、全局 Settings、active Chroma、Gold 或 qrels。

## STATUS

- STATUS: **PASS**
- Production replay parity: `True`; D20 parity: `True`.
- Frozen identity: `True`; deterministic checks: `True`.

## Candidate metrics

| Arm | HitRate ALL | AnchorRecall ALL | AllRequired ALL | HitRate INDEXABLE | AnchorRecall INDEXABLE | AllRequired INDEXABLE |
|---|---:|---:|---:|---:|---:|---:|
| D20 | 0.8250 | 0.8182 | 0.8000 | 0.8684 | 0.8571 | 0.8421 |
| D30 | 0.9250 | 0.9318 | 0.9250 | 0.9737 | 0.9762 | 0.9737 |
| D40 | 0.9250 | 0.9318 | 0.9250 | 0.9737 | 0.9762 | 0.9737 |

## Final IR metrics @5

| Arm | P@5 | Recall@5 | MRR@5 | MAP@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| D20 | 0.2684 | 0.5103 | 0.5175 | 0.3631 | 0.4903 |
| D30 | 0.2632 | 0.5792 | 0.5412 | 0.3858 | 0.5163 |
| D40 | 0.2526 | 0.5678 | 0.5202 | 0.3677 | 0.4971 |

## Latency

| Arm | Pair total | Dense mean ms | Reranker model mean ms | Reranker E2E mean ms | Retrieval total mean ms |
|---|---:|---:|---:|---:|---:|
| D20 | 800 | 12.46 | 6957.32 | 7009.66 | 7022.12 |
| D30 | 1200 | 10.18 | 7087.99 | 7173.61 | 7183.79 |
| D40 | 1600 | 6.10 | 9562.20 | 9663.29 | 9669.39 |

## PAIRWISE DELTA

配对结果使用固定 seed 的题族 bootstrap；报告 absolute delta、wins/ties/losses 与 CI，不用任意阈值替代观察结果。

- `candidate:ALL_GOLD:AllRequired:D20_vs_D30`: delta=0.125000; wins/ties/losses=5/35/0; CI95=[0.025000, 0.225000]
- `candidate:ALL_GOLD:AllRequired:D20_vs_D40`: delta=0.125000; wins/ties/losses=5/35/0; CI95=[0.025000, 0.225000]
- `candidate:ALL_GOLD:AllRequired:D30_vs_D40`: delta=0.000000; wins/ties/losses=0/40/0; CI95=[0.000000, 0.000000]
- `candidate:ALL_GOLD:EvidenceAnchorRecall:D20_vs_D30`: delta=0.112500; wins/ties/losses=5/35/0; CI95=[0.025000, 0.212500]
- `candidate:ALL_GOLD:EvidenceAnchorRecall:D20_vs_D40`: delta=0.112500; wins/ties/losses=5/35/0; CI95=[0.025000, 0.212500]
- `candidate:ALL_GOLD:EvidenceAnchorRecall:D30_vs_D40`: delta=0.000000; wins/ties/losses=0/40/0; CI95=[0.000000, 0.000000]
- `candidate:ALL_GOLD:HitRate:D20_vs_D30`: delta=0.100000; wins/ties/losses=4/36/0; CI95=[0.025000, 0.200000]
- `candidate:ALL_GOLD:HitRate:D20_vs_D40`: delta=0.100000; wins/ties/losses=4/36/0; CI95=[0.025000, 0.200000]
- `candidate:ALL_GOLD:HitRate:D30_vs_D40`: delta=0.000000; wins/ties/losses=0/40/0; CI95=[0.000000, 0.000000]
- `candidate:INDEXABLE_ONLY:AllRequired:D20_vs_D30`: delta=0.131579; wins/ties/losses=5/33/0; CI95=[0.026316, 0.236842]
- `candidate:INDEXABLE_ONLY:AllRequired:D20_vs_D40`: delta=0.131579; wins/ties/losses=5/33/0; CI95=[0.026316, 0.236842]
- `candidate:INDEXABLE_ONLY:AllRequired:D30_vs_D40`: delta=0.000000; wins/ties/losses=0/38/0; CI95=[0.000000, 0.000000]
- `candidate:INDEXABLE_ONLY:EvidenceAnchorRecall:D20_vs_D30`: delta=0.118421; wins/ties/losses=5/33/0; CI95=[0.026316, 0.223684]
- `candidate:INDEXABLE_ONLY:EvidenceAnchorRecall:D20_vs_D40`: delta=0.118421; wins/ties/losses=5/33/0; CI95=[0.026316, 0.223684]
- `candidate:INDEXABLE_ONLY:EvidenceAnchorRecall:D30_vs_D40`: delta=0.000000; wins/ties/losses=0/38/0; CI95=[0.000000, 0.000000]
- `candidate:INDEXABLE_ONLY:HitRate:D20_vs_D30`: delta=0.105263; wins/ties/losses=4/34/0; CI95=[0.026316, 0.210526]
- `candidate:INDEXABLE_ONLY:HitRate:D20_vs_D40`: delta=0.105263; wins/ties/losses=4/34/0; CI95=[0.026316, 0.210526]
- `candidate:INDEXABLE_ONLY:HitRate:D30_vs_D40`: delta=0.000000; wins/ties/losses=0/38/0; CI95=[0.000000, 0.000000]
- `final:MAP@5:D20_vs_D30`: delta=0.022694; wins/ties/losses=4/30/4; CI95=[-0.010702, 0.064793]
- `final:MAP@5:D20_vs_D40`: delta=0.004609; wins/ties/losses=4/27/7; CI95=[-0.035020, 0.047800]
- `final:MAP@5:D30_vs_D40`: delta=-0.018085; wins/ties/losses=0/33/5; CI95=[-0.042485, -0.001681]
- `final:MRR@5:D20_vs_D30`: delta=0.023684; wins/ties/losses=4/32/2; CI95=[-0.015789, 0.069737]
- `final:MRR@5:D20_vs_D40`: delta=0.002632; wins/ties/losses=4/29/5; CI95=[-0.044737, 0.049561]
- `final:MRR@5:D30_vs_D40`: delta=-0.021053; wins/ties/losses=0/34/4; CI95=[-0.051754, -0.001316]
- `final:Recall@5:D20_vs_D30`: delta=0.068911; wins/ties/losses=4/31/3; CI95=[-0.009797, 0.168022]
- `final:Recall@5:D20_vs_D40`: delta=0.057507; wins/ties/losses=4/29/5; CI95=[-0.022704, 0.159011]
- `final:Recall@5:D30_vs_D40`: delta=-0.011404; wins/ties/losses=0/36/2; CI95=[-0.031579, 0.000000]
- `final:nDCG@5:D20_vs_D30`: delta=0.026074; wins/ties/losses=4/30/4; CI95=[-0.024387, 0.085943]
- `final:nDCG@5:D20_vs_D40`: delta=0.006859; wins/ties/losses=4/27/7; CI95=[-0.045270, 0.066956]
- `final:nDCG@5:D30_vs_D40`: delta=-0.019215; wins/ties/losses=0/33/5; CI95=[-0.041371, -0.001772]

## TRANSITIONS

见 `transition_analysis.json`，其中保存 D20→D30 与 D30→D40 的逐题状态和找回证据是否被 Reranker 利用。

## Decision

- RECOMMENDED_DENSE_K: **30**
- PROMOTION_RECOMMENDED: **True**（仅建议另立 Phase 15C，未执行 Promotion）
- KNOWN_NEXT_BOTTLENECK: **RERANKER_MISS**
- Reason: D30 与 D40 的 Candidate INDEXABLE_ONLY coverage 相同，且 D30 的最终主指标不低于 D40
