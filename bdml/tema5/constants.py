from pathlib import Path

SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400

DEFAULT_DATA_DIR = Path(__file__).with_name("data")
DEFAULT_CHECKPOINT_PATH = Path(__file__).with_name("librispeech_lstm_ctc.pt")

DEFAULT_TRAIN_URL = "train-clean-100"
DEFAULT_VALIDATION_URL = "dev-clean"
