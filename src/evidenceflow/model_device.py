"""提供 Embedding 与 Reranker 共用的最小推理设备选择规则。"""


class ModelDeviceError(RuntimeError):
    """请求的模型推理设备当前不可用。"""


def resolve_model_device(requested: str) -> str:
    """将 Settings 设备选项解析为可直接用于 PyTorch 的设备名。"""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("model device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        # 强制 CPU 不需要探测或初始化 CUDA Runtime。
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
        # 固定到首张 CUDA 设备，避免各模型各自推断不同的默认设备。
        return "cuda:0"
    if requested == "cuda":
        raise ModelDeviceError(
            "CUDA was requested, but this PyTorch environment has no available CUDA device"
        )
    return "cpu"
