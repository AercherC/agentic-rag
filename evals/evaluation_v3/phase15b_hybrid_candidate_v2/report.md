# EvidenceFlow Phase 15B Hybrid Candidate Retrieval Benchmark

本报告只描述离线候选池实验；未修改 Production Runtime、active Chroma、Gold、qrels 或 mapping。

## Frozen inputs

- Collection: `ef_research_bge_m3_hybrid_c512_v2`; documents/chunks: `10/817`.
- Corpus signature: `bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332`.
- Index signature: `b4e9432c7170615ade2c53803a887fa86d0f793949934a57db421f0da57739e5`; chunk ID hash: `e87a613ebe65b10262996e83974b55839e0b6c27fbe22053ee7b8826dd1f153d`.
- Active relevance projection hash: `6ccf0ffeffb8799429baff192e2de75e57199d1868688ca75ac2479cfb0cc4cd`.
- Gold mapping hash: `42d067404ae01e204bdfcfce87f008d11a237d24215e9b05fb89642a6c004203`; qrels hash: `9337c26327097085dd3dc400a73e6c0baa79f375944d2be401edd39d014c9bb9`.
- Current active qrels are a read-only evidence-anchor projection because the canonical qrels contain frozen fixed-index chunk IDs.

## Candidate metrics

| Arm | HitRate@pool ALL | AnchorRecall@pool ALL | AllRequired@pool ALL | HitRate@pool INDEXABLE | AnchorRecall@pool INDEXABLE | AllRequired@pool INDEXABLE |
|---|---:|---:|---:|---:|---:|---:|
| B0 Dense20 | 0.8250 | 0.8182 | 0.8000 | 0.8684 | 0.8571 | 0.8421 |
| B1 Dense40 | 0.9250 | 0.9318 | 0.9250 | 0.9737 | 0.9762 | 0.9737 |
| B2 BM25-20 | 0.8250 | 0.8182 | 0.8000 | 0.8684 | 0.8571 | 0.8421 |
| B3 Dense20 union BM25-20 | 0.8500 | 0.8409 | 0.8250 | 0.8947 | 0.8810 | 0.8684 |

## Final Learned Reranker IR @5

| Arm | P@5 | Recall@5 | MRR@5 | MAP@5 | nDCG@5 | eligible | excluded |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Dense20 | 0.2684 | 0.5103 | 0.5175 | 0.3631 | 0.4903 | 38 | 2 |
| B1 Dense40 | 0.2526 | 0.5678 | 0.5202 | 0.3677 | 0.4971 | 38 | 2 |
| B2 BM25-20 | 0.2474 | 0.4972 | 0.5382 | 0.3547 | 0.4803 | 38 | 2 |
| B3 Dense20 union BM25-20 | 0.2526 | 0.5217 | 0.5136 | 0.3488 | 0.4786 | 38 | 2 |

## Baseline contract and replay parity

- Formal B0: `Production Dense Top20 -> BAAI/bge-reranker-base -> Top5`.
- Phase13F numeric classification: `RAW_DENSE20_TOP5 / PRE_RERANK_REFERENCE`.
- Phase13F values match raw Dense20 Top5: `True` (informational only).
- Active relevance projection hash: `6ccf0ffeffb8799429baff192e2de75e57199d1868688ca75ac2479cfb0cc4cd`.
- Independent Production replay parity: `True`.
- Diagnosis: Phase13F values are retained only as RAW_DENSE20_TOP5 / PRE_RERANK_REFERENCE; formal B0 is decided by independent production replay parity.
- B0 parity is a prerequisite for B1/B2/B3; no Runtime promotion or Phase 15C action is part of this run.

## Complementarity

Indexable evidence groups: `42`; dense-only `1`, BM25-only `1`, both `35`, neither `5`.
BM25 recovered `1` indexable Dense20 misses.

## Failure attribution

Failure labels are assigned from frozen-index membership, candidate membership and final Top-5 membership; no answer generation was run.

| Arm | UNINDEXABLE | CANDIDATE_MISS | RERANKER_MISS | SUCCESS |
|---|---:|---:|---:|---:|
| B0 Dense20 | 2 | 6 | 7 | 25 |
| B1 Dense40 | 2 | 1 | 9 | 28 |
| B2 BM25-20 | 2 | 6 | 7 | 25 |
| B3 Dense20 union BM25-20 | 2 | 5 | 7 | 26 |

## Gates and decision

- All required Phase 15B gates: **PASS**.
- Recommendation: **DO_NOT_PROMOTE**.
- This recommendation is offline evidence only; no Runtime promotion was performed.
