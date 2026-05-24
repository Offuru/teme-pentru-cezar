from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchaudio

try:
    from .constants import N_MELS
    from .data import SpeechFeatureExtractor
    from .model import create_model, load_checkpoint
    from .text import CharTokenizer
except ImportError:
    from constants import N_MELS
    from data import SpeechFeatureExtractor
    from model import create_model, load_checkpoint
    from text import CharTokenizer


def greedy_decode(logits: torch.Tensor, tokenizer: CharTokenizer) -> str:
    token_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    return tokenizer.decode_ctc(token_ids)


def transcribe_waveform(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    sample_rate: int,
    feature_extractor: SpeechFeatureExtractor,
    tokenizer: CharTokenizer,
    device: torch.device,
) -> str:
    model.eval()
    features = feature_extractor(waveform, sample_rate)
    features = features.unsqueeze(0).to(device)
    lengths = torch.tensor([features.size(1)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(features, lengths)
    return greedy_decode(logits, tokenizer)


def transcribe_file(
    model: torch.nn.Module,
    audio_path: str | Path,
    feature_extractor: SpeechFeatureExtractor,
    tokenizer: CharTokenizer,
    device: torch.device,
) -> str:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    return transcribe_waveform(
        model=model,
        waveform=waveform,
        sample_rate=sample_rate,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, CharTokenizer, SpeechFeatureExtractor, dict[str, Any]]:
    tokenizer = CharTokenizer()
    feature_extractor = SpeechFeatureExtractor()
    model = create_model(
        input_dim=N_MELS,
        vocab_size=tokenizer.vocab_size,
        device=device,
    )
    checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model, tokenizer, feature_extractor, checkpoint
