"""验证模型推理设备解析规则，不依赖真实 CUDA 硬件。"""

import sys
from types import SimpleNamespace

import pytest

from evidenceflow.model_device import ModelDeviceError, resolve_model_device


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [("auto", True, "cuda:0"), ("auto", False, "cpu"), ("cpu", True, "cpu")],
)
def test_resolves_automatic_and_forced_cpu_devices(
    monkeypatch, requested: str, available: bool, expected: str
) -> None:
    """auto 仅在 CUDA 可用时选首张卡，强制 CPU 不受 CUDA 状态影响。"""

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_model_device(requested) == expected


def test_explicit_cuda_fails_clearly_when_unavailable(monkeypatch) -> None:
    """显式 CUDA 请求不可用时必须报错，不静默回退到 CPU。"""

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ModelDeviceError, match="no available CUDA device"):
        resolve_model_device("cuda")


def test_invalid_device_is_rejected_without_cuda_probe() -> None:
    """设备配置只接受 auto、cpu、cuda 三种稳定取值。"""

    with pytest.raises(ValueError, match="auto, cpu, cuda"):
        resolve_model_device("gpu")
