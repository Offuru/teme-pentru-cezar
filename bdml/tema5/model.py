from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class SpeechToTextLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        vocab_size: int,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, vocab_size),
        )

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(features)
        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            lengths=feature_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        unpacked, unpacked_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
        )
        logits = self.classifier(unpacked)
        return logits, unpacked_lengths


def create_model(
    input_dim: int,
    vocab_size: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.2,
    device: str | torch.device | None = None,
) -> SpeechToTextLSTM:
    model = SpeechToTextLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        vocab_size=vocab_size,
    )
    if device is not None:
        model = model.to(device)
    return model


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    extra_state: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "extra_state": extra_state,
        },
        Path(checkpoint_path),
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return (
        checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": state_dict}
    )
