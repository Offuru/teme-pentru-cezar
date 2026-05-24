from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

try:
    from .constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATA_DIR, N_MELS
    from .data import get_dataloaders
    from .gui import SpeechToTextApp
    from .model import create_model
    from .train import train_and_save
except ImportError:
    from constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATA_DIR, N_MELS
    from data import get_dataloaders
    from gui import SpeechToTextApp
    from model import create_model
    from train import train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LibriSpeech LSTM speech-to-text")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--train", action="store_true", help="Train model before launching GUI"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(preferred: str) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def progress_printer(payload: dict[str, Any]) -> None:
    phase = payload.get("phase")
    if phase == "train_batch":
        print(
            "Epoch {epoch}/{epochs} | Batch {batch}/{batches} | loss={batch_loss:.4f}".format(
                **payload
            ),
            end="\r",
            flush=True,
        )
    elif phase == "epoch_end":
        print(
            "Epoch {epoch}/{epochs} done | train_loss={train_loss:.4f} | val_loss={validation_loss:.4f}".format(
                **payload
            )
        )


def run_training(args: argparse.Namespace, device: torch.device) -> None:
    train_loader, validation_loader, tokenizer, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )
    model = create_model(
        input_dim=N_MELS,
        vocab_size=tokenizer.vocab_size,
        device=device,
    )
    train_and_save(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        checkpoint_path=args.checkpoint,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        extra_state={
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "vocab": tokenizer.vocab,
            "input_dim": N_MELS,
        },
        progress_callback=progress_printer,
    )
    print(f"Saved checkpoint to {args.checkpoint}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.train:
        run_training(args, device)

    app = SpeechToTextApp(checkpoint_path=args.checkpoint)
    app.run()


if __name__ == "__main__":
    main()
