"""
train_cnn_only.py — Standalone CNN training script.

Trains the CNN model for up to CNN_EPOCHS epochs (with EarlyStopping),
then saves the best weights to models/CNN_final.keras.

Usage:
    python train_cnn_only.py

Expected outcome: ~91–93 % test accuracy on Fashion-MNIST.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import load_fashion_mnist, preprocess_for_cnn
from src.cnn_model   import build_cnn_model
from src.trainer     import train_model, save_history, save_model_final, get_callbacks
from src.config      import MODELS_DIR, CNN_EPOCHS, BATCH_SIZE, VALIDATION_SPLIT

# ─── 1. Load & Preprocess ─────────────────────────────────────────────────────
print("\n[INFO] Loading Fashion-MNIST …")
(X_train, y_train), (X_test, y_test) = load_fashion_mnist()
X_train_4d, X_test_4d = preprocess_for_cnn(X_train, X_test)
print(f"       Train shape: {X_train_4d.shape}  |  Test shape: {X_test_4d.shape}")

# ─── 2. Build CNN ─────────────────────────────────────────────────────────────
print("\n[INFO] Building CNN architecture …")
cnn = build_cnn_model()
cnn.summary()

# ─── 3. Train ─────────────────────────────────────────────────────────────────
print(f"\n[INFO] Training CNN for up to {CNN_EPOCHS} epochs "
      f"(batch={BATCH_SIZE}, val_split={VALIDATION_SPLIT:.0%}) …")

history = train_model(
    model=cnn,
    X_train=X_train_4d,
    y_train=y_train,
    epochs=CNN_EPOCHS,         # 50 — EarlyStopping (patience=8) will stop early if needed
    batch_size=BATCH_SIZE,     # 64
    validation_split=VALIDATION_SPLIT,  # 0.2
    model_name="CNN"
)

# Persist CSV history for later analysis / plotting
save_history(history, "CNN")

# ─── 4. Evaluate on Test Set ──────────────────────────────────────────────────
print("\n[INFO] Evaluating on test set …")
test_loss, test_acc = cnn.evaluate(X_test_4d, y_test, verbose=0)
print(f"       Test loss     : {test_loss:.4f}")
print(f"       Test accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)")

# ─── 5. Save Final Model ──────────────────────────────────────────────────────
# EarlyStopping already restored best weights; save_model_final writes CNN_final.keras
save_path = save_model_final(cnn, "CNN")

print(f"\n✅  CNN training complete!")
print(f"    Test accuracy : {test_acc * 100:.2f}%")
print(f"    Model saved   : {save_path}")
