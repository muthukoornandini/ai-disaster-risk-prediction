"""
preprocess.py
-------------
Loads the raw sensor/environmental CSV, cleans it, encodes labels,
scales features, and splits into train/val/test sets.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

FEATURE_COLUMNS = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "river_level_m",
    "soil_moisture_pct",
    "seismic_magnitude",
    "pressure_hpa",
]

RISK_TARGET = "risk_level"
TYPE_TARGET = "disaster_type"

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURE_COLUMNS + [RISK_TARGET, TYPE_TARGET])
    return df


def build_preprocessors(df):
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS].values)

    type_encoder = LabelEncoder()
    type_encoder.fit(df[TYPE_TARGET].values)

    return scaler, type_encoder


def transform(df, scaler, type_encoder):
    X = scaler.transform(df[FEATURE_COLUMNS].values).astype(np.float32)
    y_risk = df[RISK_TARGET].values.astype(np.int64)
    y_type = type_encoder.transform(df[TYPE_TARGET].values).astype(np.int64)
    return X, y_risk, y_type


def prepare_datasets(csv_path, test_size=0.15, val_size=0.15, random_state=42):
    df = load_data(csv_path)

    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), random_state=random_state,
        stratify=df[RISK_TARGET]
    )
    relative_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), random_state=random_state,
        stratify=temp_df[RISK_TARGET]
    )

    scaler, type_encoder = build_preprocessors(train_df)

    X_train, y_train_risk, y_train_type = transform(train_df, scaler, type_encoder)
    X_val, y_val_risk, y_val_type = transform(val_df, scaler, type_encoder)
    X_test, y_test_risk, y_test_type = transform(test_df, scaler, type_encoder)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(type_encoder, os.path.join(ARTIFACT_DIR, "type_encoder.pkl"))

    return {
        "train": (X_train, y_train_risk, y_train_type),
        "val": (X_val, y_val_risk, y_val_type),
        "test": (X_test, y_test_risk, y_test_type),
        "scaler": scaler,
        "type_encoder": type_encoder,
        "feature_columns": FEATURE_COLUMNS,
    }


def load_preprocessors():
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    type_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "type_encoder.pkl"))
    return scaler, type_encoder
