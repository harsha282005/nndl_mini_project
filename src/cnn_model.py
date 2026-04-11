"""
cnn_model.py — Convolutional Neural Network for Fashion-MNIST.

Architecture (3 conv-blocks + dense head):
  Input(28, 28, 1)
    → Conv2D(32, 3×3) → BN → ReLU → Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
    → Conv2D(64, 3×3) → BN → ReLU → Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
    → Conv2D(128,3×3) → BN → ReLU                           → MaxPool(2×2) → Dropout(0.25)
    → Flatten
    → Dense(256, ReLU) → BN → Dropout(0.5)
    → Dense(128, ReLU) → BN → Dropout(0.5)
    → Dense(10, Softmax)

Expected test accuracy on Fashion-MNIST: ~91–93 % after ≥20 epochs.

Public API
----------
build_cnn_model()  — Simple Sequential model (matches task spec, used by trainer)
build_cnn()        — Full functional-API model (backward-compat with main.py / api.py)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, Activation, Input,
)

from src.config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    CNN_FILTERS, CNN_KERNEL_SIZE, CNN_POOL_SIZE,
    CNN_DROPOUT_RATE, CNN_DENSE_UNITS, LEARNING_RATE,
)

NUM_CLASSES = 10


# ─── Primary model builder (matches task specification) ───────────────────────

def build_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                    learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Build and compile the CNN model (Sequential API).

    This is the primary builder used by trainer.py / train_cnn_only.py.

    Architecture
    ------------
    3 convolutional blocks (32 → 64 → 128 filters) with BatchNorm,
    followed by a two-layer dense head (256 → 128) with Dropout(0.5).

    Parameters
    ----------
    input_shape   : (H, W, C)  default (28, 28, 1)
    learning_rate : Adam learning rate

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential(name="CNN_FashionMNIST", layers=[
        # ── Block 1: 32 filters ──────────────────────────────────────────────
        Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4),
               input_shape=input_shape, name="conv1_a"),
        BatchNormalization(name="bn1_a"),
        Activation("relu", name="relu1_a"),
        Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4),
               name="conv1_b"),
        BatchNormalization(name="bn1_b"),
        Activation("relu", name="relu1_b"),
        MaxPooling2D((2, 2), name="pool1"),
        Dropout(0.25, name="drop_conv1"),

        # ── Block 2: 64 filters ──────────────────────────────────────────────
        Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4),
               name="conv2_a"),
        BatchNormalization(name="bn2_a"),
        Activation("relu", name="relu2_a"),
        Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4),
               name="conv2_b"),
        BatchNormalization(name="bn2_b"),
        Activation("relu", name="relu2_b"),
        MaxPooling2D((2, 2), name="pool2"),
        Dropout(0.25, name="drop_conv2"),

        # ── Block 3: 128 filters ─────────────────────────────────────────────
        Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4),
               name="conv3_a"),
        BatchNormalization(name="bn3_a"),
        Activation("relu", name="relu3_a"),
        MaxPooling2D((2, 2), name="pool3"),
        Dropout(0.25, name="drop_conv3"),

        # ── Dense Head ───────────────────────────────────────────────────────
        Flatten(name="flatten"),

        Dense(256, kernel_regularizer=regularizers.l2(1e-4), name="fc1"),
        BatchNormalization(name="bn_fc1"),
        Activation("relu", name="relu_fc1"),
        Dropout(0.5, name="drop_fc1"),

        Dense(128, kernel_regularizer=regularizers.l2(1e-4), name="fc2"),
        BatchNormalization(name="bn_fc2"),
        Activation("relu", name="relu_fc2"),
        Dropout(0.5, name="drop_fc2"),

        # ── Output ───────────────────────────────────────────────────────────
        Dense(NUM_CLASSES, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ─── Backward-compatible alias (used by main.py / api.py) ────────────────────

def build_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
              filters=CNN_FILTERS,
              kernel_size=CNN_KERNEL_SIZE,
              pool_size=CNN_POOL_SIZE,
              dropout_rate=CNN_DROPOUT_RATE,
              dense_units=CNN_DENSE_UNITS,
              learning_rate=LEARNING_RATE,
              use_augmentation: bool = False) -> tf.keras.Model:
    """
    Functional-API CNN builder (backward-compatible with main.py / api.py).

    NOTE: `use_augmentation` is kept for API compatibility but defaults to
    False — on 28×28 Fashion-MNIST images, heavy on-the-fly augmentation
    degrades val_accuracy and causes premature EarlyStopping.  If you want
    augmentation, set use_augmentation=True; only gentle transforms are applied.

    Parameters
    ----------
    input_shape      : (H, W, C)
    filters          : List of filter counts per conv block
    kernel_size      : Convolution kernel size
    pool_size        : MaxPool size
    dropout_rate     : Dropout probability for conv-block drops (dense uses 0.5)
    dense_units      : Neurons per FC layer
    learning_rate    : Adam learning rate
    use_augmentation : Prepend gentle augmentation layers (default False)

    Returns
    -------
    Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs

    # ── Optional gentle augmentation ──────────────────────────────────────────
    if use_augmentation:
        x = layers.RandomTranslation(0.05, 0.05, name="aug_translate")(x)
        x = layers.RandomZoom(0.05, name="aug_zoom")(x)

    # ── Convolutional Blocks ──────────────────────────────────────────────────
    for i, f in enumerate(filters):
        # First conv in block
        x = layers.Conv2D(f, kernel_size, padding="same",
                          kernel_regularizer=regularizers.l2(1e-4),
                          name=f"conv{i+1}_a")(x)
        x = layers.BatchNormalization(name=f"bn{i+1}_a")(x)
        x = layers.Activation("relu", name=f"relu{i+1}_a")(x)

        # Second conv (all but last block)
        if i < len(filters) - 1:
            x = layers.Conv2D(f, kernel_size, padding="same",
                              kernel_regularizer=regularizers.l2(1e-4),
                              name=f"conv{i+1}_b")(x)
            x = layers.BatchNormalization(name=f"bn{i+1}_b")(x)
            x = layers.Activation("relu", name=f"relu{i+1}_b")(x)

        x = layers.MaxPooling2D(pool_size, name=f"pool{i+1}")(x)
        x = layers.Dropout(dropout_rate * 0.5, name=f"drop_conv{i+1}")(x)

    # ── Fully Connected Head ──────────────────────────────────────────────────
    x = layers.Flatten(name="flatten")(x)

    for j, units in enumerate(dense_units):
        x = layers.Dense(units,
                         kernel_regularizer=regularizers.l2(1e-4),
                         name=f"fc{j+1}")(x)
        x = layers.BatchNormalization(name=f"bn_fc{j+1}")(x)
        x = layers.Activation("relu", name=f"relu_fc{j+1}")(x)
        x = layers.Dropout(0.5, name=f"drop_fc{j+1}")(x)

    # ── Output ────────────────────────────────────────────────────────────────
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="CNN_FashionMNIST")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ─── Utility ──────────────────────────────────────────────────────────────────

def print_cnn_summary():
    model = build_cnn_model()
    model.summary()
    return model


if __name__ == "__main__":
    print_cnn_summary()
