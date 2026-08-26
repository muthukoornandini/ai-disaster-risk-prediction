"""
generate_sample_data.py
------------------------
Generates a synthetic environmental/sensor dataset for training the
disaster risk prediction model. Replace this with real historical
weather / seismic / hydrological data for production use.

Features:
    rainfall_mm        - 24h rainfall in millimeters
    temperature_c       - average temperature in Celsius
    humidity_pct         - relative humidity percentage
    wind_speed_kmh       - wind speed in km/h
    river_level_m        - river water level in meters
    soil_moisture_pct    - soil moisture percentage
    seismic_magnitude    - recorded seismic magnitude (0 if none)
    pressure_hpa         - atmospheric pressure in hPa

Targets:
    disaster_type  - none | flood | cyclone | earthquake | drought
    risk_level     - low | medium | high | critical  (ordinal 0-3)
"""

import numpy as np
import pandas as pd
import os

RNG_SEED = 42
N_SAMPLES = 12000

np.random.seed(RNG_SEED)


def generate_row(rng):
    disaster_type = rng.choice(
        ["none", "flood", "cyclone", "earthquake", "drought"],
        p=[0.45, 0.20, 0.15, 0.10, 0.10],
    )

    # Baseline "normal" conditions
    rainfall = rng.gamma(2.0, 8.0)
    temperature = rng.normal(27, 5)
    humidity = np.clip(rng.normal(60, 15), 5, 100)
    wind_speed = rng.gamma(2.0, 6.0)
    river_level = np.clip(rng.normal(3.0, 1.0), 0, None)
    soil_moisture = np.clip(rng.normal(35, 12), 0, 100)
    seismic_magnitude = np.clip(rng.exponential(0.4), 0, None)
    pressure = rng.normal(1013, 8)

    if disaster_type == "flood":
        rainfall += rng.gamma(4.0, 25.0)
        river_level += rng.gamma(3.0, 1.5)
        soil_moisture = np.clip(soil_moisture + rng.uniform(20, 45), 0, 100)
        humidity = np.clip(humidity + rng.uniform(10, 25), 0, 100)
        pressure -= rng.uniform(2, 10)

    elif disaster_type == "cyclone":
        wind_speed += rng.gamma(4.0, 20.0)
        rainfall += rng.gamma(3.0, 20.0)
        pressure -= rng.uniform(15, 45)
        humidity = np.clip(humidity + rng.uniform(15, 30), 0, 100)

    elif disaster_type == "earthquake":
        seismic_magnitude += rng.uniform(3.5, 8.5)

    elif disaster_type == "drought":
        rainfall = max(0, rainfall - rng.uniform(5, 15))
        soil_moisture = np.clip(soil_moisture - rng.uniform(15, 30), 0, 100)
        temperature += rng.uniform(3, 9)
        humidity = np.clip(humidity - rng.uniform(15, 35), 0, 100)

    # Derive a risk_level (0=low,1=medium,2=high,3=critical) from a
    # weighted severity score so the label is consistent with features.
    severity = 0.0
    severity += min(rainfall / 150.0, 1.0) * 1.2
    severity += min(wind_speed / 140.0, 1.0) * 1.2
    severity += min(river_level / 9.0, 1.0) * 1.3
    severity += min(seismic_magnitude / 8.0, 1.0) * 1.5
    severity += min(max(0, (35 - soil_moisture)) / 35.0, 1.0) * 0.6  # drought pull
    severity += min(max(0, (1013 - pressure)) / 45.0, 1.0) * 0.8

    if disaster_type == "none":
        severity *= 0.35

    severity = np.clip(severity + rng.normal(0, 0.15), 0, 5)

    if severity < 0.9:
        risk_level = 0  # low
    elif severity < 1.8:
        risk_level = 1  # medium
    elif severity < 2.8:
        risk_level = 2  # high
    else:
        risk_level = 3  # critical

    return {
        "rainfall_mm": round(rainfall, 2),
        "temperature_c": round(temperature, 2),
        "humidity_pct": round(humidity, 2),
        "wind_speed_kmh": round(wind_speed, 2),
        "river_level_m": round(river_level, 2),
        "soil_moisture_pct": round(soil_moisture, 2),
        "seismic_magnitude": round(seismic_magnitude, 2),
        "pressure_hpa": round(pressure, 2),
        "disaster_type": disaster_type,
        "risk_level": risk_level,
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    rows = [generate_row(rng) for _ in range(N_SAMPLES)]
    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "disaster_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["disaster_type"].value_counts())
    print(df["risk_level"].value_counts().sort_index())


if __name__ == "__main__":
    main()
