import torch
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration for the tea-leaf classification project"""

    # Reproducibility
    SEED = int(os.getenv("SEED", 1337))

    # Device
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Paths
    DATA_ROOT = os.getenv("DATA_ROOT", "./data/teaLeafBD/teaLeafBD")
    EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "./output"))

    # Data
    IMG_SIZE = int(os.getenv("IMG_SIZE", 320))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
    VAL_RATIO = float(os.getenv("VAL_RATIO", 0.15))

    # Training
    EPOCHS = int(os.getenv("EPOCHS", 50))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-4))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
    WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", 2))
    GRAD_CLIP = float(os.getenv("GRAD_CLIP", 1.0))

    # Model Architecture
    BACKBONE_NAME = os.getenv("BACKBONE_NAME", "efficientnet-b0")
    PROTOS_PER_CLASS = int(os.getenv("PROTOS_PER_CLASS", 12))
    PROTOTYPE_DIM = int(os.getenv("PROTOTYPE_DIM", 256))
    PUSH_EVERY_EPOCH = int(os.getenv("PUSH_EVERY_EPOCH", 2))

    # Visualization
    TOPK_PROTOTYPE_OVERLAYS = int(os.getenv("TOPK_PROTOTYPE_OVERLAYS", 3))
    N_TILES_PER_PROTO = int(os.getenv("N_TILES_PER_PROTO", 4))
    TILE_PX = int(os.getenv("TILE_PX", 112))
    TSNE_MAX_ITEMS = int(os.getenv("TSNE_MAX_ITEMS", 1600))

    # OOD Configuration
    UNKNOWN_CLASS_NAME = os.getenv("UNKNOWN_CLASS_NAME", "Helopeltis")
    RUN_OOD_SWEEP = os.getenv("RUN_OOD_SWEEP", "False").lower() == "true"

    # Class imbalance handling
    CLASS_WEIGHT_SMOOTHING = float(os.getenv("CLASS_WEIGHT_SMOOTHING", 0.1))
    USE_OVERSAMPLING = os.getenv("USE_OVERSAMPLING", "True").lower() == "true"

    def __init__(self):
        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        os.makedirs(self.EXPORT_DIR / self.UNKNOWN_CLASS_NAME, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=== Configuration ===")
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
