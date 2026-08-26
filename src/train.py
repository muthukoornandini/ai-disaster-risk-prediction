"""
train.py
--------
Trains the disaster risk prediction model end-to-end:
  1. Loads/prepares data (generates synthetic data if none exists)
  2. Builds the model
  3. Trains with early stopping + checkpointing
  4. Evaluates on the held-out test set
  5. Saves the final model + a training history plot
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import prepare_datasets, FEATURE_COLUMNS
from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "data", "disaster_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EPOCHS = 60
BATCH_SIZE = 64


def ensure_data():
    if not os.path.exists(DATA_CSV):
        print("No dataset found — generating synthetic sample data...")
        sys.path.append(os.path.join(BASE_DIR, "data"))
        import generate_sample_data
        generate_sample_data.main()


def plot_history(history, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(history.history["risk_level_accuracy"], label="train")
    axes[0, 0].plot(history.history["val_risk_level_accuracy"], label="val")
    axes[0, 0].set_title("Risk Level Accuracy")
    axes[0, 0].legend()

    axes[0, 1].plot(history.history["disaster_type_accuracy"], label="train")
    axes[0, 1].plot(history.history["val_disaster_type_accuracy"], label="val")
    axes[0, 1].set_title("Disaster Type Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(history.history["loss"], label="train")
    axes[1, 0].plot(history.history["val_loss"], label="val")
    axes[1, 0].set_title("Total Loss")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.1, 0.5,
        "Disaster Risk Prediction Model\nTraining Summary",
        fontsize=13, weight="bold"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved training history plot to {out_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensure_data()

    print("Preparing datasets...")
    data = prepare_datasets(DATA_CSV)
    X_train, y_train_risk, y_train_type = data["train"]
    X_val, y_val_risk, y_val_type = data["val"]
    X_test, y_test_risk, y_test_type = data["test"]

    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train,
        {"risk_level": y_train_risk, "disaster_type": y_train_type},
        validation_data=(X_val, {"risk_level": y_val_risk, "disaster_type": y_val_type}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nEvaluating on test set...")
    results = model.evaluate(
        X_test, {"risk_level": y_test_risk, "disaster_type": y_test_type},
        verbose=0, return_dict=True
    )
    print(json.dumps(results, indent=2))

    final_model_path = os.path.join(MODEL_DIR, "disaster_model.keras")
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    with open(os.path.join(MODEL_DIR, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
