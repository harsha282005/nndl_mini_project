"""
evaluator.py — Model evaluation utilities.

Provides:
  • evaluate_model()         — Loss + Accuracy on test set
  • plot_training_curves()   — Accuracy & Loss vs Epochs
  • plot_confusion_matrix()  — Heatmap of predictions vs ground-truth
  • print_classification_report()
  • compare_models()         — Side-by-side bar chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import tensorflow as tf

from src.config import CLASS_NAMES, PLOTS_DIR, REPORTS_DIR

# Dark theme for all plots
plt.rcParams.update({
    "axes.facecolor":   "#16213e",
    "figure.facecolor": "#1a1a2e",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "axes.edgecolor":   "#444",
    "grid.color":       "#333",
})


# ─── Test-set Evaluation ──────────────────────────────────────────────────────

def evaluate_model(model: tf.keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   model_name: str = "model") -> dict:
    """Run model.evaluate() and return a results dict."""
    print(f"\n[INFO] Evaluating {model_name} on test set …")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results = {"model": model_name, "test_loss": loss, "test_accuracy": acc}
    print(f"       Test Loss     : {loss:.4f}")
    print(f"       Test Accuracy : {acc * 100:.2f}%\n")
    return results


# ─── Training Curves ─────────────────────────────────────────────────────────

def plot_training_curves(history: dict,
                         model_name: str,
                         save: bool = True):
    """
    Plot Accuracy vs Epochs and Loss vs Epochs side by side.

    Parameters
    ----------
    history    : dict with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'
    model_name : Used for title and save filename
    """
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax1.plot(epochs, history["accuracy"],     color="#00d2ff", lw=2, label="Train")
    ax1.plot(epochs, history["val_accuracy"], color="#ff6b6b", lw=2,
             linestyle="--", label="Validation")
    ax1.set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#555")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax2.plot(epochs, history["loss"],     color="#00d2ff", lw=2, label="Train")
    ax2.plot(epochs, history["val_loss"], color="#ff6b6b", lw=2,
             linestyle="--", label="Validation")
    ax2.set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#555")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f"{model_name}_training_curves.png")
        plt.savefig(path, bbox_inches="tight", dpi=150,
                    facecolor=fig.get_facecolor())
        print(f"[INFO] Training curves saved → {path}")
    plt.show()
    plt.close()


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(model: tf.keras.Model,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          model_name: str,
                          save: bool = True):
    """Predict on test set and plot a normalised confusion matrix heatmap."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True, fmt=".2f",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        linecolor="#333",
        annot_kws={"size": 8}
    )
    ax.set_title(f"{model_name} — Confusion Matrix (Normalised)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label",      labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
        plt.savefig(path, bbox_inches="tight", dpi=150,
                    facecolor=fig.get_facecolor())
        print(f"[INFO] Confusion matrix saved → {path}")
    plt.show()
    plt.close()

    return y_pred


# ─── Classification Report ───────────────────────────────────────────────────

def print_classification_report(y_test: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str,
                                 save: bool = True) -> str:
    """Print and optionally save the sklearn classification report."""
    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(f"\n{'='*60}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*60}")
    print(report)

    if save:
        path = os.path.join(REPORTS_DIR, f"{model_name}_classification_report.txt")
        with open(path, "w") as f:
            f.write(f"Classification Report — {model_name}\n\n")
            f.write(report)
        print(f"[INFO] Report saved → {path}")

    return report


# ─── Model Comparison ────────────────────────────────────────────────────────

def compare_models(results: list, save: bool = True):
    """
    Bar chart comparing test accuracy of multiple models.

    Parameters
    ----------
    results : list of dicts with keys 'model', 'test_accuracy', 'test_loss'
    """
    models_   = [r["model"]         for r in results]
    accs      = [r["test_accuracy"] * 100 for r in results]
    losses    = [r["test_loss"]     for r in results]

    x = np.arange(len(models_))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#00d2ff", "#a855f7"]

    # Accuracy bars
    bars1 = ax1.bar(x, accs, width=0.5,
                    color=colors[:len(models_)], edgecolor="white", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_, fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Model Comparison — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{acc:.2f}%", ha="center", va="bottom",
                 fontsize=11, color="white", fontweight="bold")

    # Loss bars
    bars2 = ax2.bar(x, losses, width=0.5,
                    color=colors[:len(models_)], edgecolor="white", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_, fontsize=11)
    ax2.set_ylabel("Test Loss")
    ax2.set_title("Model Comparison — Loss", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, loss in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"{loss:.4f}", ha="center", va="bottom",
                 fontsize=11, color="white", fontweight="bold")

    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "model_comparison.png")
        plt.savefig(path, bbox_inches="tight", dpi=150,
                    facecolor=fig.get_facecolor())
        print(f"[INFO] Comparison chart saved → {path}")
    plt.show()
    plt.close()
