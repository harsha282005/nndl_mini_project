"""
trainer.py — Training utilities for Fashion-MNIST models.

Provides:
  • get_callbacks()   — EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
  • train_model()     — Wrapper that trains, logs, and returns history
  • save_model_final()— Save the final (best-weights) model to disk
  • save_history()    — Persist training history as CSV
  • load_history()    — Reload a previously saved history
"""

import os
import csv
import numpy as np
import tensorflow as tf

from src.config import (
    MODELS_DIR, REPORTS_DIR,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MONITOR,
    BATCH_SIZE, VALIDATION_SPLIT
)


# ─── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str, patience: int = EARLY_STOPPING_PATIENCE):
    """
    Returns a list of Keras callbacks:
      1. EarlyStopping       — stops when val_accuracy plateaus
      2. ReduceLROnPlateau   — halves LR on val_loss plateau
      3. ModelCheckpoint     — saves the best weights
      4. LambdaCallback      — verbose per-epoch accuracy/val_accuracy log
    """
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")

    def _epoch_end(epoch, logs):
        acc     = logs.get("accuracy",     0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        loss    = logs.get("loss",         0.0)
        val_loss= logs.get("val_loss",     0.0)
        print(
            f"  [Epoch {epoch + 1:>3}]  "
            f"loss={loss:.4f}  acc={acc:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=EARLY_STOPPING_MONITOR,
            save_best_only=True,
            verbose=0
        ),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=_epoch_end),
    ]
    return callbacks


# ─── Train ────────────────────────────────────────────────────────────────────

def train_model(model: tf.keras.Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                epochs: int,
                batch_size: int = BATCH_SIZE,
                validation_split: float = VALIDATION_SPLIT,
                model_name: str = "model") -> tf.keras.callbacks.History:
    """
    Train a Keras model with callbacks and verbose logging.

    Parameters
    ----------
    model            : Compiled Keras model
    X_train          : Training features
    y_train          : Training labels
    epochs           : Maximum number of epochs (EarlyStopping may halt sooner)
    batch_size       : Samples per gradient update  (default 64)
    validation_split : Fraction of training data for validation (default 0.2)
    model_name       : Used for checkpoint filename and log prefix

    Returns
    -------
    Keras History object
    """
    print(f"\n{'='*60}")
    print(f"  Training  : {model_name}")
    print(f"  Max epochs: {epochs}  |  Batch size: {batch_size}")
    print(f"  Val split : {validation_split*100:.0f}%  |  "
          f"EarlyStopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"{'='*60}\n")

    callbacks = get_callbacks(model_name)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=0          # Suppress default output; LambdaCallback prints cleanly
    )

    best_val = max(history.history["val_accuracy"])
    final_val = history.history["val_accuracy"][-1]
    n_epochs  = len(history.history["loss"])

    print(f"\n[INFO] {model_name} training complete.")
    print(f"       Epochs run         : {n_epochs}")
    print(f"       Best  val_accuracy : {best_val:.4f}  ({best_val*100:.2f}%)")
    print(f"       Final val_accuracy : {final_val:.4f}  ({final_val*100:.2f}%)\n")

    return history


# ─── Save Final Model ─────────────────────────────────────────────────────────

def save_model_final(model: tf.keras.Model, model_name: str) -> str:
    """
    Save the final (best-weights restored) model to models/<model_name>_final.keras.

    Returns the save path.
    """
    # Canonical lowercase name expected by api.py: cnn_final.keras / ann_final.keras
    filename  = f"{model_name}_final.keras"
    save_path = os.path.join(MODELS_DIR, filename)
    model.save(save_path)
    print(f"[INFO] Model saved -> {save_path}")
    return save_path


# ─── Persist History ─────────────────────────────────────────────────────────

def save_history(history: tf.keras.callbacks.History, model_name: str):
    """Save training history dict to a CSV file in outputs/reports/."""
    path = os.path.join(REPORTS_DIR, f"{model_name}_history.csv")
    keys = list(history.history.keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + keys)
        writer.writeheader()
        n_epochs = len(history.history[keys[0]])
        for i in range(n_epochs):
            row = {"epoch": i + 1}
            row.update({k: history.history[k][i] for k in keys})
            writer.writerow(row)

    print(f"[INFO] Training history saved -> {path}")


def load_history(model_name: str) -> dict:
    """Load a previously saved training history CSV as a dict."""
    path = os.path.join(REPORTS_DIR, f"{model_name}_history.csv")
    history = {}

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k == "epoch":
                    continue
                history.setdefault(k, []).append(float(v))

    return history
