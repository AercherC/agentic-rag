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
