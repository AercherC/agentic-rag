"""Minimal inference-device selection rules shared by the Embedding and Reranker models."""


class ModelDeviceError(RuntimeError):
    """The requested model inference device is currently unavailable."""


def resolve_model_device(requested: str) -> str:
    """Resolve a Settings device option into a device name PyTorch can use directly."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("model device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        # Forcing CPU avoids probing or initializing the CUDA runtime.
        return "cpu"

    try:
        import torch
    except ImportError as exc:
        if requested == "auto":
            return "cpu"
        raise ModelDeviceError(
            "CUDA was requested, but PyTorch is unavailable in this environment"
        ) from exc

    if torch.cuda.is_available():
        # Pin to the first CUDA device so that each model does not infer a different default
        # device.
        return "cuda:0"
    if requested == "cuda":
        raise ModelDeviceError(
            "CUDA was requested, but this PyTorch environment has no available CUDA device"
        )
    return "cpu"
