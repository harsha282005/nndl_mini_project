"""
ann_model.py — Fully-Connected (ANN) Model for Fashion-MNIST.

Architecture:
  Input(784) → Dense(512, ReLU) → BN → Dropout
             → Dense(256, ReLU) → BN → Dropout
             → Dense(128, ReLU) → BN → Dropout
             → Dense(10, Softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from src.config import (
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES,
    ANN_HIDDEN_UNITS, ANN_DROPOUT_RATE, LEARNING_RATE
)

NUM_CLASSES = 10


def build_ann(input_dim: int = IMG_HEIGHT * IMG_WIDTH,
              hidden_units: list = ANN_HIDDEN_UNITS,
              dropout_rate: float = ANN_DROPOUT_RATE,
              learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Build and compile the ANN model.

    Parameters
    ----------
    input_dim     : Flattened input size (default 784 for 28×28 images)
    hidden_units  : List of neuron counts for each hidden layer
    dropout_rate  : Dropout probability after each hidden layer
    learning_rate : Adam learning rate

    Returns
    -------
    Compiled Keras Model
    """
    model = models.Sequential(name="ANN_FashionMNIST")

    # ── Input + first hidden layer ──────────────────────────────────────────
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"dense_{i+1}"
        ))
        model.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    # ── Output layer ─────────────────────────────────────────────────────────
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="output"))

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def print_ann_summary():
    model = build_ann()
    model.summary()
    return model


if __name__ == "__main__":
    print_ann_summary()
