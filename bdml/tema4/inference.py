from __future__ import annotations

from pathlib import Path

import torch

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise ImportError(
        "Pillow is required for image loading in the Tkinter app"
    ) from exc

try:
    from .constants import CIFAR10_CLASSES
    from .data import build_eval_transform
    from .model import load_checkpoint
except ImportError:
    from constants import CIFAR10_CLASSES
    from data import build_eval_transform
    from model import load_checkpoint


def load_image_tensor(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = build_eval_transform()
    return transform(image)


def predict_top_k(
    model: torch.nn.Module,
    image_path: str | Path,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    model.eval()
    image_tensor = load_image_tensor(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_k = max(1, min(top_k, len(CIFAR10_CLASSES)))
    scores, indices = torch.topk(probabilities, k=top_k)
    return [
        (CIFAR10_CLASSES[index.item()], float(score.item()))
        for score, index in zip(scores, indices)
    ]


def load_trained_model(
    model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device
) -> torch.nn.Module:
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model
