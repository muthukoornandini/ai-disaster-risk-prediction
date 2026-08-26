"""
predict.py
----------
Loads the trained model + preprocessors and runs predictions on new
sensor readings. Can be used as a library (predict_risk) or run
directly from the command line with example data.
"""

import os
import json
import numpy as np
import tensorflow as tf

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_preprocessors, FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "disaster_model.keras")

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

_model = None
_scaler = None
_type_encoder = None


def _load_all():
    global _model, _scaler, _type_encoder
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
        _scaler, _type_encoder = load_preprocessors()
    return _model, _scaler, _type_encoder


def predict_risk(sensor_reading: dict):
    """
    sensor_reading: dict with keys matching FEATURE_COLUMNS, e.g.
        {
          "rainfall_mm": 120.5,
          "temperature_c": 26.3,
          "humidity_pct": 88.0,
          "wind_speed_kmh": 45.0,
          "river_level_m": 6.2,
          "soil_moisture_pct": 70.0,
          "seismic_magnitude": 0.1,
          "pressure_hpa": 995.0
        }
    Returns a dict with predicted risk level, disaster type, and
    confidence scores.
    """
    model, scaler, type_encoder = _load_all()

    x = np.array([[sensor_reading[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    x_scaled = scaler.transform(x)

    risk_probs, type_probs = model.predict(x_scaled, verbose=0)
    risk_idx = int(np.argmax(risk_probs[0]))
    type_idx = int(np.argmax(type_probs[0]))

    return {
        "risk_level": RISK_LABELS[risk_idx],
        "risk_level_index": risk_idx,
        "risk_confidence": float(np.max(risk_probs[0])),
        "risk_probabilities": {
            RISK_LABELS[i]: float(p) for i, p in enumerate(risk_probs[0])
        },
        "predicted_disaster_type": type_encoder.inverse_transform([type_idx])[0],
        "type_confidence": float(np.max(type_probs[0])),
        "type_probabilities": {
            label: float(p) for label, p in zip(type_encoder.classes_, type_probs[0])
        },
    }


def predict_batch(readings: list):
    return [predict_risk(r) for r in readings]


if __name__ == "__main__":
    example = {
        "rainfall_mm": 145.0,
        "temperature_c": 25.0,
        "humidity_pct": 92.0,
        "wind_speed_kmh": 38.0,
        "river_level_m": 7.1,
        "soil_moisture_pct": 78.0,
        "seismic_magnitude": 0.2,
        "pressure_hpa": 990.0,
    }
    result = predict_risk(example)
    print(json.dumps(result, indent=2))
