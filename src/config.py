"""
config.py — Central configuration for Fashion-MNIST Project
All hyperparameters, paths, and constants are defined here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
NUM_CLASSES   = 10
IMG_HEIGHT    = 28
IMG_WIDTH     = 28
IMG_CHANNELS  = 1

# ─── Training ─────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
BATCH_SIZE    = 64
ANN_EPOCHS    = 30
CNN_EPOCHS    = 50          # Enough headroom; EarlyStopping will halt early
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2     # 20% val split → better generalization signal for CNN

# ─── ANN Architecture ─────────────────────────────────────────────────────────
ANN_HIDDEN_UNITS = [512, 256, 128]
ANN_DROPOUT_RATE = 0.3

# ─── CNN Architecture ─────────────────────────────────────────────────────────
CNN_FILTERS      = [32, 64, 128]
CNN_KERNEL_SIZE  = (3, 3)
CNN_POOL_SIZE    = (2, 2)
CNN_DROPOUT_RATE = 0.4
CNN_DENSE_UNITS  = [256, 128]

# ─── Early Stopping ───────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 8    # Increased patience: prevents stopping during val_acc dips
EARLY_STOPPING_MONITOR  = "val_accuracy"

# ─── Data Augmentation ────────────────────────────────────────────────────────
AUGMENTATION_ROTATION    = 10   # degrees
AUGMENTATION_WIDTH_SHIFT = 0.10
AUGMENTATION_HEIGHT_SHIFT= 0.10
AUGMENTATION_ZOOM        = 0.10
AUGMENTATION_HORIZONTAL_FLIP = True
