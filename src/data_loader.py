"""
data_loader.py — Fashion-MNIST dataset loading and preprocessing.

Responsibilities:
  • Download / cache the dataset via Keras
  • Normalize pixel values to [0, 1]
  • Reshape images for ANN (flat) and CNN (4-D) consumption
  • Visualize sample images with their ground-truth labels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tensorflow as tf

from src.config import (
    CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    RANDOM_SEED, PLOTS_DIR
)

NUM_CLASSES = len(CLASS_NAMES)   # 10


# ─── Load & Preprocess ────────────────────────────────────────────────────────

def load_fashion_mnist():
    """
    Returns
    -------
    (X_train, y_train), (X_test, y_test)
        Pixel values normalised to [0, 1], dtype float32.
        Shape: (N, 28, 28) — channel-dim added later per model.
    """
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("[INFO] Loading Fashion-MNIST dataset …")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    print(f"       Train samples : {X_train.shape[0]:,}")
    print(f"       Test  samples : {X_test.shape[0]:,}")
    print(f"       Image shape   : {X_train.shape[1:]}")
    print(f"       Classes       : {NUM_CLASSES}\n")

    return (X_train, y_train), (X_test, y_test)


NUM_CLASSES = len(CLASS_NAMES)


def preprocess_for_ann(X_train, X_test):
    """Flatten images → (N, 784) for a fully-connected ANN."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0],  -1)
    return X_train_flat, X_test_flat


def preprocess_for_cnn(X_train, X_test):
    """Add channel dimension → (N, 28, 28, 1) for CNN."""
    X_train_4d = X_train[..., np.newaxis]
    X_test_4d  = X_test[...,  np.newaxis]
    return X_train_4d, X_test_4d


# ─── Visualisation ────────────────────────────────────────────────────────────

def visualize_samples(X, y, n_rows=4, n_cols=8, save=True):
    """Display a grid of sample images with class labels."""
    fig = plt.figure(figsize=(n_cols * 1.6, n_rows * 1.8))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.3)

    indices = np.random.choice(len(X), n_rows * n_cols, replace=False)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])

        # Handle both (28,28) and (28,28,1) shapes
        img = X[idx].squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(CLASS_NAMES[y[idx]], fontsize=7, color="white", pad=3)
        ax.axis("off")

    fig.suptitle(
        "Fashion-MNIST — Sample Images",
        fontsize=14, color="white", fontweight="bold", y=1.01
    )

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "sample_images.png")
        plt.savefig(path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        print(f"[INFO] Sample grid saved → {path}")
    plt.show()
    plt.close()


def visualize_class_distribution(y_train, y_test, save=True):
    """Bar chart of per-class sample counts for train and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, NUM_CLASSES))

    for ax, (y, split) in zip(axes, [(y_train, "Train"), (y_test, "Test")]):
        counts = [np.sum(y == i) for i in range(NUM_CLASSES)]
        bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_facecolor("#16213e")
        ax.set_title(f"{split} Set Distribution", color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white", labelrotation=45)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        # Annotate bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{count:,}",
                ha="center", va="bottom", color="white", fontsize=8
            )

    fig.suptitle("Class Distribution", fontsize=14, color="white", fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "class_distribution.png")
        plt.savefig(path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        print(f"[INFO] Class distribution saved → {path}")
    plt.show()
    plt.close()
