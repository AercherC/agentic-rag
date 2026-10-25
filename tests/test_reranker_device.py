"""验证 Reranker 模型与 tokenizer 输出张量使用一致设备。"""

import sys
from types import ModuleType, SimpleNamespace

import torch

from evidenceflow.retrieval.reranker import TransformersCrossEncoderScorer


class _FakeTensor:
    """记录测试中编码张量的设备迁移。"""

    def __init__(self) -> None:
        self.device = "cpu"

    def to(self, device: str) -> "_FakeTensor":
        self.device = device
        return self


class _FakeBatch(dict):
    """模拟 Hugging Face BatchEncoding 的批量张量迁移行为。"""

    def to(self, device: str) -> "_FakeBatch":
        for tensor in self.values():
            tensor.to(device)
        return self


class _FakeLogits:
    """提供 scorer 后处理所需的最小 Tensor 接口。"""

    def view(self, *_shape: int) -> "_FakeLogits":
        return self

    def float(self) -> "_FakeLogits":
        return self

    def cpu(self) -> "_FakeLogits":
        return self

    def tolist(self) -> list[float]:
        return [0.75]


def test_reranker_model_and_tokenized_inputs_share_device(monkeypatch) -> None:
    """模型常驻选定设备，CPU tokenizer 产出的 batch 在 forward 前迁移。"""

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
    """Fast tokenizer 并发调用不安全：score_pairs 必须串行进入，截断数按线程返回。"""

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
            time.sleep(0.01)  # 拉长临界区，让未加锁的实现必然发生重叠
            with guard:
                active["count"] -= 1
            if not padding and not truncation:
                # 查询文本长度即“超长”对数，便于断言每个线程拿到自己的截断数。
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
