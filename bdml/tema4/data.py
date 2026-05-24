from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise ImportError(
        "torchvision is required for CIFAR-10 dataset loading and image transforms"
    ) from exc

try:
    from .constants import CIFAR10_MEAN, CIFAR10_STD, DEFAULT_DATA_DIR
except ImportError:
    from constants import CIFAR10_MEAN, CIFAR10_STD, DEFAULT_DATA_DIR


@dataclass
class TrainingHistory:
    train_loss: list[float]
    train_accuracy: list[float]
    validation_loss: list[float]
    validation_accuracy: list[float]


def build_train_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_eval_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_dataloaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    data_path = Path(data_dir)
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=download,
        transform=build_train_transform(),
    )
    validation_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=download,
        transform=build_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean().item()


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    batches = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            total_accuracy += _accuracy_from_logits(logits, targets)
            batches += 1

    if batches == 0:
        return 0.0, 0.0

    return total_loss / batches, total_accuracy / batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_factory: (
        Callable[[list[torch.Tensor]], torch.optim.Optimizer] | None
    ) = None,
    scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
    progress_callback: Callable[[dict[str, float | int | str]], None] | None = None,
) -> TrainingHistory:
    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optimizer_factory(model.parameters())
        if optimizer_factory is not None
        else torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    )
    scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None

    history = TrainingHistory([], [], [], [])
    model.to(device)
    total_batches = max(1, len(train_loader))

    for epoch_index in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        batch_count = 0

        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += _accuracy_from_logits(logits.detach(), targets)
            batch_count += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "train_batch",
                        "epoch": epoch_index + 1,
                        "epochs": epochs,
                        "batch": batch_index,
                        "batches": total_batches,
                        "batch_loss": float(loss.item()),
                        "batch_accuracy": float(
                            _accuracy_from_logits(logits.detach(), targets)
                        ),
                    }
                )

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, batch_count)
        train_accuracy = running_accuracy / max(1, batch_count)
        validation_loss, validation_accuracy = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_accuracy)
        history.validation_loss.append(validation_loss)
        history.validation_accuracy.append(validation_accuracy)

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "epoch_end",
                    "epoch": epoch_index + 1,
                    "epochs": epochs,
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_accuracy),
                    "validation_loss": float(validation_loss),
                    "validation_accuracy": float(validation_accuracy),
                }
            )

    return history
