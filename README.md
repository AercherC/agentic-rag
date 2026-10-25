# EvidenceFlow

A scientific-document RAG system for evidence-grounded retrieval and answering over
research PDFs. Current active corpus: **10 point-cloud completion papers**.

This repository contains the **implementation, tests, and CLI entry points**. Evaluation
contracts, experimental artifacts, and internal working notes are intentionally not
published here.

## Capability Layers

### 1. Scientific PDF Ingestion

```text
Documents -> Clean -> Structured ScientificDocument -> Hybrid Chunk (BGE-M3, <=512 tokens)
```

PDF parsing and cleaning use a fused strategy (selected from a parser benchmark). Table
structure is linearized, not preserved.

### 2. Indexing & Retrieval

```text
BGE-M3 Embedding -> Chroma
Query -> BGE-M3 Dense -> Exact Dense Top20 (in-process ExactDenseIndex, numpy.exact)
```

The production collection is resolved through the lifecycle active pointer, never
hard-coded. Chroma is the sole persistence source; the exact dense path is reproducible
across processes. The ACTIVE index is never ingested into directly — rebuilds go through
`scripts/build_staging_index.py` into a staging collection first.

### 3. Learned Reranking

```text
Exact Dense Top20 -> BGE Reranker -> Top5
```

`runtime.py` is the single composition root for the `ingest` / `ask` production wiring.
The CLI only triggers assembly; question-answering orchestration lives solely in
`RAGService`.

### 4. Evidence Context

```text
Top5 -> Context Builder
```

### 5. Answer & Citation

```text
Context Builder -> LLM (JSON output) -> Citation Validation -> RAGAnswer(status)
```

- Structured generation: the model returns only
  `{"status": "answered" | "insufficient_context", "answer": "..."}`. Output is validated
  with Pydantic; malformed output is repaired at most once, then reported as a generation
  format error.
- `RAGAnswer.status` distinguishes "answered" from "insufficient evidence" abstention.
  `answered` requires valid citations; `insufficient_context` may carry none.
- The default LLM is DeepSeek. Switching providers requires overriding `LLM_BASE_URL`,
  `LLM_MODEL`, and usually `LLM_THINKING`.
- Endpoints, model names, and API keys come from `.env` or environment variables. No real
  credentials are stored in this repository.

### 6. Evaluation

Retrieval quality is measured with standard IR metrics (Precision, Recall, MRR, MAP, nDCG)
implemented canonically in `src/evidenceflow/eval/ir_metrics.py` following TREC / BEIR
semantics. Gold labels use `document_id + page_range + evidence_anchor`, so they do not
depend on chunk IDs and stay stable across re-chunking.

## Usage

```powershell
conda activate agent-gpu

# Optional: reuse a local model cache and stay offline
$env:HF_HOME = (Resolve-Path .cache/huggingface)
$env:TRANSFORMERS_OFFLINE = '1'

python -m evidenceflow ask "How does PoinTr represent point clouds for a Transformer?"
```

`python -m evidenceflow ingest` is only for a fresh environment (no lifecycle state file,
target is not the ACTIVE collection). Running it against the ACTIVE index is refused.

Copy `.env.example` to `.env` and fill in your own credentials before the first run.

## Repository layout

```text
src/evidenceflow/     implementation
  config.py           Settings (single source of run-time parameters)
  runtime.py          composition root for ingest / ask
  domain/             domain models
  ingestion/          PDF loading, cleaning, structure parsing, chunking
  retrieval/          dense retrieval and reranking
  eval/               IR metrics, e2e evaluation, gates
  storage/            Chroma persistence
scripts/              CLI entry points and maintenance scripts
tests/                offline test suite (Fake/Stub, no network)
evals/research_*/     research question sets and dense baseline used by tests
```

## Testing

```powershell
python -m pytest -q
```

The suite is designed to run offline against fixed inputs. Model-backed paths are
exercised through fakes and stubs rather than real API calls. A small number of tests
assert on files that are not published in this repository; those are skipped or expected
to fail in this trimmed checkout.

## Known limitations

- Academic PDFs hyphenate across line breaks, which complicates anchoring.
- Tables are linearized; row/column structure is not preserved.
- References and supplementary sections have no structured metadata yet.
- Multi-document questions can only reliably retrieve a subset of their sources.

The system is currently text-centric. There is no evidence yet that multimodal parsing
would pay off.
