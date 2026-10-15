"""Physical channel removal and CPU dynamic quantization, including reloadable exports."""

from copy import deepcopy
from pathlib import Path
import platform

import torch
from torch import nn

from .model import Reranker


def prune_hidden_channels(model: Reranker, keep_fraction: float = 0.5) -> tuple[Reranker, dict]:
    """Select channels by incoming/outgoing L2 product and copy compact dense matrices.

    This removes channels, rather than retaining masked weights in the original
    matrices. ReLU is channel-wise, so slicing adjacent matrices is exact relative
    to an original network with the removed hidden activations set to zero.
    """
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    with torch.no_grad():
        importance1 = model.fc1.weight.norm(dim=1) * model.fc2.weight.norm(dim=0)
        importance2 = model.fc2.weight.norm(dim=1) * model.out.weight.norm(dim=0)
        keep1 = importance1.topk(max(1, int(model.hidden[0] * keep_fraction))).indices.sort().values
        keep2 = importance2.topk(max(1, int(model.hidden[1] * keep_fraction))).indices.sort().values
        compact = Reranker(model.input_dim, (len(keep1), len(keep2)))
        compact.fc1.weight.copy_(model.fc1.weight[keep1])
        compact.fc1.bias.copy_(model.fc1.bias[keep1])
        compact.fc2.weight.copy_(model.fc2.weight[keep2][:, keep1])
        compact.fc2.bias.copy_(model.fc2.bias[keep2])
        compact.out.weight.copy_(model.out.weight[:, keep2])
        compact.out.bias.copy_(model.out.bias)
    return compact.eval(), {
        "keep_fraction": keep_fraction,
        "criterion": "incoming_weight_l2_times_outgoing_weight_l2",
        "kept_hidden1": keep1.tolist(),
        "kept_hidden2": keep2.tolist(),
        "original_hidden": list(model.hidden),
        "compact_hidden": list(compact.hidden),
    }


def select_quantized_engine() -> str:
    supported = set(torch.backends.quantized.supported_engines) - {"none"}
    preferred = ("qnnpack", "onednn", "x86", "fbgemm") if platform.machine().lower() in {"arm64", "aarch64"} else ("x86", "fbgemm", "onednn", "qnnpack")
    for engine in preferred:
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError("This PyTorch build has no CPU int8 engine; run a supported CPU build.")


def quantize_int8(model: Reranker, engine: str | None = None) -> tuple[nn.Module, str]:
    if engine is None:
        engine = select_quantized_engine()
    if engine not in torch.backends.quantized.supported_engines or engine == "none":
        raise RuntimeError(f"quantized engine {engine!r} is not supported by this PyTorch build")
    torch.backends.quantized.engine = engine
    quantized = torch.ao.quantization.quantize_dynamic(deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
    return quantized.eval(), engine


def save_checkpoint(model: nn.Module, path: Path, architecture: dict, mean: torch.Tensor, std: torch.Tensor, engine: str | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": architecture,
        "quantized_engine": engine,
        "feature_mean": mean,
        "feature_std": std,
        "state_dict": model.state_dict(),
    }, path)
    return path.stat().st_size


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    arch = payload["architecture"]
    model = Reranker(arch["input_dim"], tuple(arch["hidden"])).eval()
    if payload["quantized_engine"] is not None:
        model, _ = quantize_int8(model, payload["quantized_engine"])
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload
