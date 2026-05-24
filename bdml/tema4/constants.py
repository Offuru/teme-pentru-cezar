from pathlib import Path

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_IMAGE_SIZE = 32

DEFAULT_DATA_DIR = Path(__file__).with_name("data")
DEFAULT_CHECKPOINT_PATH = Path(__file__).with_name("cifar10_cnn.pt")
