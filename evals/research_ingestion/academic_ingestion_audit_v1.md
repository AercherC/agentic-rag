# Academic PDF Ingestion Compatibility Audit v1

## Baseline

- Corpus: 10 point-cloud-completion papers, 145 pages.
- Collection: `ef_research_bge_m3_c512_o64_v1`.
- Current pipeline: PyMuPDF → Cleaner → C512/O64 → BGE-M3 → Chroma.
- Ingestion: 0 failures; 361 chunks.
- Visual QA: 49 representative pages.

## Per-paper statistics

| Paper | Pages | Chunks | Characters | Hyphen breaks | Ref page | Supp page | Max pages/chunk |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapointr.pdf | 22 | 75 | 110717 | 140 | 15 | 16 | 3 |
| hyper_pcn.pdf | 17 | 33 | 48802 | 168 | 9 | 11 | 6 |
| pcn_point_completion_network.pdf | 17 | 34 | 55675 | 180 | 10 | 12 | 4 |
| pfnet_point_fractal_network.pdf | 9 | 25 | 39031 | 134 | 9 | - | 2 |
| pointattn.pdf | 12 | 25 | 42391 | 5 | 10 | - | 3 |
| pointr_geometry_aware_transformers.pdf | 14 | 39 | 60776 | 177 | 11 | 12 | 2 |
| proxyformer.pdf | 10 | 31 | 48071 | 151 | 9 | - | 2 |
| seedformer.pdf | 22 | 39 | 59082 | 88 | 15 | 18 | 2 |
| snowflakenet.pdf | 11 | 31 | 51130 | 183 | 9 | - | 2 |
| svdformer.pdf | 11 | 29 | 46038 | 180 | 9 | - | 2 |

## Issue matrix

| Issue | Papers affected | Severity |
|---|---:|---|
| extraction_failure | 0/10 | None |
| double_column_order | 0/10 | None |
| broken_hyphenation | 10/10 | Medium |
| section_detection_missing | 10/10 | Medium |
| header_footer_repetition | 2/10 | Low |
| table_structure_loss | 10/10 | High |
| figure_text_noise | 10/10 | Low |
| formula_fragmentation | 10/10 | Medium |
| references_pollution | 10/10 | Medium |
| supplement_boundary_lost | 5/5 | Medium |
| chunk_boundary_crossing | 10/10 | Medium |

## Interpretation

- All titles and abstracts were visually readable; no scanned-only or extraction-empty PDF was selected.
- Representative double-column pages preserved a usable reading sequence, but the Domain Model does not expose section structure.
- Tables are visually intact in the PDF but flattened in extracted text, so row/column semantics are not guaranteed.
- Every paper contributes References as normal chunks; all five included appendices/supplements lack an explicit boundary field.
- Figure captions are present, while text embedded in diagrams can enter the stream as low-level noise; this is not yet evidence for multimodal parsing.
- No core component is changed in this stage. Retrieval results must determine the single next experiment.
