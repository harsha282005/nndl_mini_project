"""
visualizer.py — Prediction visualization utilities.

Provides:
  • visualize_predictions()  — Grid of test images with pred vs true labels
  • visualize_wrong()        — Grid showing only misclassified images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

from src.config import CLASS_NAMES, PLOTS_DIR


def _setup_dark_grid(n_rows, n_cols, title, figsize=None):
    if figsize is None:
        figsize = (n_cols * 1.8, n_rows * 2.2)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.3)
    fig.suptitle(title, fontsize=14, color="white", fontweight="bold", y=1.01)
    return fig, gs


# ─── Prediction Grid ─────────────────────────────────────────────────────────

def visualize_predictions(model: tf.keras.Model,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           model_name: str,
                           n_rows: int = 4,
                           n_cols: int = 8,
                           save: bool = True):
    """
    Display a grid of test images annotated with:
      • Ground-truth label (white)
      • Predicted label   (green if correct, red if wrong)
    """
    n = n_rows * n_cols
    indices = np.random.choice(len(X_test), n, replace=False)

    X_sample = X_test[indices]
    y_true   = y_test[indices]

    y_pred_proba = model.predict(X_sample, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    fig, gs = _setup_dark_grid(
        n_rows, n_cols,
        f"{model_name} — Predictions (green=correct, red=wrong)"
    )

    for i, (img, true, pred) in enumerate(zip(X_sample, y_true, y_pred)):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.imshow(img.squeeze(), cmap="gray")
        correct = (true == pred)
        color   = "#00e676" if correct else "#ff5252"
        ax.set_title(
            f"T: {CLASS_NAMES[true][:6]}\nP: {CLASS_NAMES[pred][:6]}",
            fontsize=6.5, color=color, pad=2
        )
        ax.axis("off")
        # Highlight border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"{model_name}_predictions.png")
        plt.savefig(path, bbox_inches="tight", dpi=150,
                    facecolor=fig.get_facecolor())
        print(f"[INFO] Prediction grid saved → {path}")
    plt.show()
    plt.close()


# ─── Wrong Predictions ───────────────────────────────────────────────────────

def visualize_wrong_predictions(model: tf.keras.Model,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 model_name: str,
                                 max_images: int = 32,
                                 save: bool = True):
    """Show images the model got wrong, annotated with true & predicted labels."""
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    wrong_idx = np.where(y_pred != y_test)[0]
    print(f"[INFO] {model_name}: {len(wrong_idx)} / {len(y_test)} wrong")

    wrong_idx = wrong_idx[:max_images]
    n = len(wrong_idx)
    n_cols = min(8, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, gs = _setup_dark_grid(
        n_rows, n_cols,
        f"{model_name} — Misclassified Images"
    )

    for i, idx in enumerate(wrong_idx):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        ax.imshow(X_test[idx].squeeze(), cmap="gray")
        ax.set_title(
            f"T: {CLASS_NAMES[y_test[idx]][:6]}\nP: {CLASS_NAMES[y_pred[idx]][:6]}",
            fontsize=6.5, color="#ff5252", pad=2
        )
        ax.axis("off")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"{model_name}_wrong_predictions.png")
        plt.savefig(path, bbox_inches="tight", dpi=150,
                    facecolor=fig.get_facecolor())
        print(f"[INFO] Wrong-predictions grid saved → {path}")
    plt.show()
    plt.close()
