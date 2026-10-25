"""Verify the model inference device resolution rules without depending on real CUDA hardware."""

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
    """auto picks the first card only when CUDA is available; a forced CPU is unaffected by the CUDA state."""

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_model_device(requested) == expected


def test_explicit_cuda_fails_clearly_when_unavailable(monkeypatch) -> None:
    """An explicit CUDA request must raise when unavailable, and must not silently fall back to CPU."""

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ModelDeviceError, match="no available CUDA device"):
        resolve_model_device("cuda")


def test_invalid_device_is_rejected_without_cuda_probe() -> None:
    """The device setting accepts only the three stable values auto, cpu and cuda."""

    with pytest.raises(ValueError, match="auto, cpu, cuda"):
        resolve_model_device("gpu")
