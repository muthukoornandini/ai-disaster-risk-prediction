"""
model.py
--------
Defines a multi-output deep neural network:
  - Shared dense trunk over environmental/sensor features
  - Head 1: risk_level classification (4 classes: low/medium/high/critical)
  - Head 2: disaster_type classification (none/flood/cyclone/earthquake/drought)
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_dim, n_risk_classes=4, n_type_classes=5, dropout=0.3):
    inputs = layers.Input(shape=(input_dim,), name="sensor_features")

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    shared = layers.Dense(32, activation="relu", name="shared_representation")(x)

    # Risk level head
    r = layers.Dense(16, activation="relu")(shared)
    risk_output = layers.Dense(n_risk_classes, activation="softmax", name="risk_level")(r)

    # Disaster type head
    t = layers.Dense(16, activation="relu")(shared)
    type_output = layers.Dense(n_type_classes, activation="softmax", name="disaster_type")(t)

    model = models.Model(inputs=inputs, outputs=[risk_output, type_output], name="disaster_risk_net")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "risk_level": "sparse_categorical_crossentropy",
            "disaster_type": "sparse_categorical_crossentropy",
        },
        loss_weights={"risk_level": 1.0, "disaster_type": 0.7},
        metrics={
            "risk_level": ["accuracy"],
            "disaster_type": ["accuracy"],
        },
    )
    return model
