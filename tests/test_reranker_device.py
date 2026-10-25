"""Verify that the Reranker model and the tokenizer output tensors use the same device."""

import sys
from types import ModuleType, SimpleNamespace

import torch

from evidenceflow.retrieval.reranker import TransformersCrossEncoderScorer


class _FakeTensor:
    """Record the device migration of the encoded tensors in the test."""

    def __init__(self) -> None:
        self.device = "cpu"

    def to(self, device: str) -> "_FakeTensor":
        self.device = device
        return self


class _FakeBatch(dict):
    """Simulate the batched tensor migration behavior of a Hugging Face BatchEncoding."""

    def to(self, device: str) -> "_FakeBatch":
        for tensor in self.values():
            tensor.to(device)
        return self


class _FakeLogits:
    """Provide the minimal Tensor interface needed by the scorer post-processing."""

    def view(self, *_shape: int) -> "_FakeLogits":
        return self

    def float(self) -> "_FakeLogits":
        return self

    def cpu(self) -> "_FakeLogits":
        return self

    def tolist(self) -> list[float]:
        return [0.75]


def test_reranker_model_and_tokenized_inputs_share_device(monkeypatch) -> None:
    """The model stays on the selected device while the batch produced by the CPU tokenizer is migrated before forward."""

    observed: dict[str, object] = {}
    batch = _FakeBatch(input_ids=_FakeTensor(), attention_mask=_FakeTensor())

    class FakeTokenizer:
        def __call__(self, _pairs, *, padding, truncation, **kwargs):
            if not padding and not truncation:
                return {"input_ids": [[1, 2]]}
            observed["tokenizer_device_before_transfer"] = batch["input_ids"].device
            return batch

    class FakeModel:
        def to(self, *, device, dtype):
            observed["model_device"] = device
            observed["model_dtype"] = dtype
            return self

        def eval(self):
            return self

        def __call__(self, *, input_ids, attention_mask, return_dict):
            observed["forward_device"] = input_ids.device
            observed["mask_device"] = attention_mask.device
            observed["return_dict"] = return_dict
            return SimpleNamespace(logits=_FakeLogits())

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda _name: FakeTokenizer())
    transformers.AutoModelForSequenceClassification = SimpleNamespace(
        from_pretrained=lambda _name: FakeModel()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "evidenceflow.retrieval.reranker.resolve_model_device",
        lambda requested: "cuda:0",
    )

    scorer = TransformersCrossEncoderScorer("test/reranker", device="auto")
    scores = scorer.score_pairs([("query", "passage")])

    assert scores == [0.75]
    assert scorer.device == "cuda:0"
    assert observed["tokenizer_device_before_transfer"] == "cpu"
    assert observed["model_device"] == "cuda:0"
    assert observed["forward_device"] == observed["model_device"]
    assert observed["mask_device"] == observed["model_device"]
    assert observed["model_dtype"] is torch.float32


def test_concurrent_score_pairs_never_overlap_and_report_per_thread_counts(monkeypatch) -> None:
    """A fast tokenizer is not safe for concurrent calls: score_pairs must enter serially, and the
    truncation count is reported per thread."""

    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    active = {"count": 0, "max": 0}
    guard = threading.Lock()

    class SlowTokenizer:
        def __call__(self, pairs, *, padding, truncation, **kwargs):
            with guard:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(0.01)  # stretch the critical section so an unlocked implementation must overlap
            with guard:
                active["count"] -= 1
            if not padding and not truncation:
                # The query text length is the number of "over-long" pairs, which makes it easy to
                # assert that each thread gets its own truncation count.
                return {"input_ids": [[0] * 10] * len(pairs[0][0])}
            return _FakeBatch(input_ids=_FakeTensor(), attention_mask=_FakeTensor())

    class FakeModel:
        def to(self, *, device, dtype):
            return self

        def eval(self):
            return self

        def __call__(self, **_inputs):
            return SimpleNamespace(logits=_FakeLogits())

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda _name: SlowTokenizer())
    transformers.AutoModelForSequenceClassification = SimpleNamespace(
        from_pretrained=lambda _name: FakeModel()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    scorer = TransformersCrossEncoderScorer("test/reranker", device="cpu", max_length=5)

    def worker(index: int) -> int:
        scorer.score_pairs([("q" * (index + 1), "passage")])
        return scorer.last_truncated_pair_count

    with ThreadPoolExecutor(max_workers=6) as pool:
        counts = list(pool.map(worker, range(6)))

    assert active["max"] == 1
    assert counts == [index + 1 for index in range(6)]
    assert scorer.forward_passes == 6
