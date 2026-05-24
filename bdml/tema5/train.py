from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .data import TrainingHistory
    from .model import save_checkpoint
except ImportError:
    from data import TrainingHistory
    from model import save_checkpoint


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            logits, output_lengths = model(features, feature_lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            total_loss += float(loss.item())
            batches += 1

    return total_loss / max(1, batches)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingHistory:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    history = TrainingHistory(train_loss=[], validation_loss=[])
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        total_batches = max(1, len(train_loader))

        for batch_index, batch in enumerate(train_loader, start=1):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, output_lengths = model(features, feature_lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "train_batch",
                        "epoch": epoch,
                        "epochs": epochs,
                        "batch": batch_index,
                        "batches": total_batches,
                        "batch_loss": float(loss.item()),
                    }
                )

        epoch_train_loss = running_loss / max(1, batches)
        epoch_validation_loss = evaluate_model(
            model, validation_loader, criterion, device
        )
        history.train_loss.append(epoch_train_loss)
        history.validation_loss.append(epoch_validation_loss)

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "epoch_end",
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": epoch_train_loss,
                    "validation_loss": epoch_validation_loss,
                }
            )

    return history


def train_and_save(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    checkpoint_path: str | Path,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    extra_state: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingHistory:
    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        progress_callback=progress_callback,
    )
    save_checkpoint(
        model,
        checkpoint_path,
        {
            **extra_state,
            "history": asdict(history),
        },
    )
    return history
