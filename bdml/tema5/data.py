from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torchaudio
from torch.utils.data import DataLoader

try:
    from .constants import (
        DEFAULT_DATA_DIR,
        DEFAULT_TRAIN_URL,
        DEFAULT_VALIDATION_URL,
        HOP_LENGTH,
        N_FFT,
        N_MELS,
        SAMPLE_RATE,
        WIN_LENGTH,
    )
    from .text import CharTokenizer
except ImportError:
    from constants import (
        DEFAULT_DATA_DIR,
        DEFAULT_TRAIN_URL,
        DEFAULT_VALIDATION_URL,
        HOP_LENGTH,
        N_FFT,
        N_MELS,
        SAMPLE_RATE,
        WIN_LENGTH,
    )
    from text import CharTokenizer


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]


class SpeechFeatureExtractor:
    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
        )

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, SAMPLE_RATE
            )
        mel = self.mel(waveform)
        log_mel = torch.log(mel + 1e-6)
        return log_mel.squeeze(0).transpose(0, 1)


def make_collate_fn(
    tokenizer: CharTokenizer,
    feature_extractor: SpeechFeatureExtractor,
) -> Callable:
    def collate(batch):
        feature_list: list[torch.Tensor] = []
        feature_lengths: list[int] = []
        targets: list[int] = []
        target_lengths: list[int] = []
        transcripts: list[str] = []

        for waveform, sample_rate, transcript, *_ in batch:
            features = feature_extractor(waveform, sample_rate)
            feature_list.append(features)
            feature_lengths.append(features.size(0))

            encoded = tokenizer.encode(transcript)
            targets.extend(encoded)
            target_lengths.append(len(encoded))
            transcripts.append(transcript.lower())

        padded_features = torch.nn.utils.rnn.pad_sequence(
            feature_list, batch_first=True
        )
        return {
            "features": padded_features,
            "feature_lengths": torch.tensor(feature_lengths, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "transcripts": transcripts,
        }

    return collate


def get_dataloaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = 8,
    num_workers: int = 0,
    download: bool = True,
    train_url: str = DEFAULT_TRAIN_URL,
    validation_url: str = DEFAULT_VALIDATION_URL,
) -> tuple[DataLoader, DataLoader, CharTokenizer, SpeechFeatureExtractor]:
    data_root = Path(data_dir)
    tokenizer = CharTokenizer()
    feature_extractor = SpeechFeatureExtractor()
    collate_fn = make_collate_fn(tokenizer, feature_extractor)

    train_dataset = torchaudio.datasets.LIBRISPEECH(
        root=data_root,
        url=train_url,
        download=download,
    )
    validation_dataset = torchaudio.datasets.LIBRISPEECH(
        root=data_root,
        url=validation_url,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, validation_loader, tokenizer, feature_extractor
