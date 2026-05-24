from __future__ import annotations

import argparse
import importlib
from typing import Any
from pathlib import Path

try:
    from .constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATA_DIR
    from .data import get_dataloaders, train_model
    from .gui import CIFAR10ClassifierApp
    from .model import create_model, save_checkpoint
except ImportError:
    from constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATA_DIR
    from data import get_dataloaders, train_model
    from gui import CIFAR10ClassifierApp
    from model import create_model, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN classifier")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--train", action="store_true", help="Train the CNN before opening the GUI"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        return "cuda"
    try:
        torch = importlib.import_module("torch")
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def train_and_save_model(
    data_dir: Path,
    checkpoint_path: Path,
    device: str,
    epochs: int,
    batch_size: int,
) -> None:
    model = create_model(device)
    train_loader, validation_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        download=True,
    )
    train_model(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=epochs,
        progress_callback=make_console_progress_callback(epochs),
    )
    save_checkpoint(model, checkpoint_path, {"device": str(device), "epochs": epochs})


def make_console_progress_callback(total_epochs: int):
    def callback(payload: dict[str, Any]) -> None:
        phase = payload.get("phase")
        if phase == "train_batch":
            epoch = int(payload["epoch"])
            epochs = int(payload.get("epochs", total_epochs))
            batch = int(payload["batch"])
            batches = int(payload["batches"])
            batch_loss = float(payload["batch_loss"])
            batch_accuracy = float(payload["batch_accuracy"])
            print(
                f"Epoch {epoch}/{epochs} | Batch {batch}/{batches} | loss={batch_loss:.4f} | acc={batch_accuracy:.3f}",
                end="\r",
                flush=True,
            )
        elif phase == "epoch_end":
            epoch = int(payload["epoch"])
            epochs = int(payload.get("epochs", total_epochs))
            train_loss = float(payload["train_loss"])
            train_accuracy = float(payload["train_accuracy"])
            validation_loss = float(payload["validation_loss"])
            validation_accuracy = float(payload["validation_accuracy"])
            print(
                f"Epoch {epoch}/{epochs} done | train_loss={train_loss:.4f} | train_acc={train_accuracy:.3f} | val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.3f}"
            )

    return callback


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.train:
        train_and_save_model(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    app = CIFAR10ClassifierApp(checkpoint_path=args.checkpoint)
    app.run()


if __name__ == "__main__":
    main()
