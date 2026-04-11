"""
main.py — End-to-end training & evaluation pipeline for Fashion-MNIST.

Run:
    python main.py

Steps executed:
  1. Load & visualise dataset
  2. Train ANN  → evaluate → visualise
  3. Train CNN  → evaluate → visualise
  4. Compare both models
  5. Save final models
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

# ── Make sure project root is on the path ─────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Local imports ─────────────────────────────────────────────────────────────
from src.config import (
    ANN_EPOCHS, CNN_EPOCHS, MODELS_DIR, CLASS_NAMES
)
from src.data_loader import (
    load_fashion_mnist,
    preprocess_for_ann,
    preprocess_for_cnn,
    visualize_samples,
    visualize_class_distribution,
)
from src.ann_model  import build_ann
from src.cnn_model  import build_cnn_model   # primary builder (no augmentation)
from src.trainer    import train_model, save_history, save_model_final
from src.evaluator  import (
    evaluate_model,
    plot_training_curves,
    plot_confusion_matrix,
    print_classification_report,
    compare_models,
)
from src.visualizer import (
    visualize_predictions,
    visualize_wrong_predictions,
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Dataset
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 — Load & Visualise Dataset")
print("═"*60)

(X_train, y_train), (X_test, y_test) = load_fashion_mnist()

# Only visualise in non-CI environments (will try, silently skip if no display)
try:
    visualize_samples(X_train, y_train)
    visualize_class_distribution(y_train, y_test)
except Exception as e:
    print(f"[WARN] Could not render plot: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ANN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 — ANN Training")
print("═"*60)

X_train_flat, X_test_flat = preprocess_for_ann(X_train, X_test)

ann = build_ann()
ann.summary()

history_ann = train_model(
    ann, X_train_flat, y_train,
    epochs=ANN_EPOCHS,
    model_name="ANN"
)
save_history(history_ann, "ANN")

# Evaluate
ann_results = evaluate_model(ann, X_test_flat, y_test, "ANN")

try:
    plot_training_curves(history_ann.history, "ANN")
    y_pred_ann = plot_confusion_matrix(ann, X_test_flat, y_test, "ANN")
    print_classification_report(y_test, y_pred_ann, "ANN")
    visualize_predictions(ann, X_test_flat, y_test, "ANN")
    visualize_wrong_predictions(ann, X_test_flat, y_test, "ANN")
except Exception as e:
    print(f"[WARN] Visualisation error: {e}")

# Save final ANN
ann_save_path = os.path.join(MODELS_DIR, "ANN_final.keras")
ann.save(ann_save_path)
print(f"[INFO] ANN saved → {ann_save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CNN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 — CNN Training")
print("═"*60)

X_train_4d, X_test_4d = preprocess_for_cnn(X_train, X_test)

# Use build_cnn_model() — cleaner Sequential architecture, no data augmentation
# (augmentation on 28×28 images hurts convergence and triggers early stopping prematurely)
cnn = build_cnn_model()
cnn.summary()

history_cnn = train_model(
    cnn, X_train_4d, y_train,
    epochs=CNN_EPOCHS,
    model_name="CNN"
)
save_history(history_cnn, "CNN")

# Evaluate
cnn_results = evaluate_model(cnn, X_test_4d, y_test, "CNN")

try:
    plot_training_curves(history_cnn.history, "CNN")
    y_pred_cnn = plot_confusion_matrix(cnn, X_test_4d, y_test, "CNN")
    print_classification_report(y_test, y_pred_cnn, "CNN")
    visualize_predictions(cnn, X_test_4d, y_test, "CNN")
    visualize_wrong_predictions(cnn, X_test_4d, y_test, "CNN")
except Exception as e:
    print(f"[WARN] Visualisation error: {e}")

# Save final CNN (best weights already restored by EarlyStopping)
cnn_save_path = save_model_final(cnn, "CNN")
print(f"[INFO] CNN saved -> {cnn_save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 — Model Comparison")
print("═"*60)

results = [ann_results, cnn_results]

try:
    compare_models(results)
except Exception as e:
    print(f"[WARN] Comparison plot error: {e}")

print("\n" + "═"*60)
print("  ✅  PIPELINE COMPLETE")
print("═"*60)
print(f"\n  ANN — Test Accuracy: {ann_results['test_accuracy']*100:.2f}%")
print(f"  CNN — Test Accuracy: {cnn_results['test_accuracy']*100:.2f}%")

ann_acc = ann_results["test_accuracy"] * 100
cnn_acc = cnn_results["test_accuracy"] * 100
delta   = cnn_acc - ann_acc
print(f"\n  📈  CNN outperforms ANN by {delta:.2f} percentage points.")
print(
    "\n  Why? CNNs learn spatial hierarchies (edges -> textures -> shapes)\n"
    "  through shared convolutional filters, making them far more\n"
    "  parameter-efficient and accurate on image data than flat ANNs.\n"
)
print("  All outputs saved in outputs/plots/ and outputs/reports/")
print("═"*60 + "\n")
