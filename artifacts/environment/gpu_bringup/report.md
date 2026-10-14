# EvidenceFlow agent-gpu environment validation

## Result

CUDA and both real model forward paths pass in `agent-gpu`. The GPU-visible full test suite has one
existing environment-sensitive test failure, so the strict all-gates `GPU_ENVIRONMENT_READY` flag
remains false until that CPU-specific test explicitly requests CPU or is otherwise corrected in a
separately authorized code/test task.

## Environment

- Conda: `agent-gpu` (`D:\miniconda\envs\agent-gpu`), Python 3.12.14, pip 26.2.1.
- PyTorch: `2.13.0+cu130`, CUDA runtime 13.0; `pip check` reports no broken requirements.
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU; driver 616.92, NVIDIA-SMI CUDA UMD 13.4,
  compute capability 12.0, reported memory 8151 MiB.
- NumPy 2.5.3 removed the original PyTorch NumPy initialization warning.
- Project dependencies came from `pyproject.toml` plus the test extra. No torchvision, torchaudio,
  or FAISS package was installed; FAISS is not declared and Exact Dense retains its NumPy fallback.

## Smoke tests

- CUDA tensor test: 2048×2048 float32 matrix multiplication completed on `cuda:0`; peak allocated
  memory was 80 MiB.
- BGE-M3: parameters and observed `input_ids`/`attention_mask` were on `cuda:0`; CPU tokenizer,
  float32 forward passed; returned normalized 1024-dimensional Python-float vector; peak allocated
  VRAM 2198.71 MiB.
- BGE Reranker: parameters and observed inputs were on `cuda:0`; tokenizer remained CPU; float32,
  batch-size setting 4 and max_length 512 retained; one pair forward passed; peak allocated VRAM
  1094.05 MiB.

## Tests

- GPU visible: 533 passed, 1 failed. The failing unit test asserts that the default provider uses
  CPU, while `MODEL_DEVICE=auto` resolves to CUDA on this machine. No runtime or algorithm code was
  changed to hide this mismatch.
- CUDA hidden only for the full deterministic regression process: 534 passed, 0 failed, 19 existing
  Chroma deprecation warnings. This does not replace the GPU-visible result above.

No CPU/GPU parity or performance benchmark was run. No Retrieval architecture, project source,
Gold/qrels, index, or frozen evaluation artifact was modified. Repository `.pytest_cache` and 17
`__pycache__` directories were removed; Hugging Face model caches were preserved.
