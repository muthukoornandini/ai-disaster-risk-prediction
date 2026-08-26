# AI-Based Disaster Risk Prediction and Alert System

A deep learning project that predicts disaster risk levels (Low / Medium / High /
Critical) and the most likely disaster type (flood, cyclone, earthquake, drought,
or none) from environmental sensor readings, and turns high-risk predictions into
actionable alerts.

Built with **TensorFlow/Keras**, **scikit-learn**, and a **Flask** web dashboard + API.

---

## 1. Project Structure

```
disaster_risk_prediction/
├── app.py                     # Flask web app (dashboard + REST API)
├── config.yaml                # Central configuration
├── requirements.txt
├── README.md
├── data/
│   ├── generate_sample_data.py    # Synthetic dataset generator
│   └── disaster_data.csv          # Generated dataset (created on first run)
├── src/
│   ├── preprocess.py          # Data loading, scaling, train/val/test split
│   ├── model.py                # Multi-output deep neural network architecture
│   ├── train.py                 # Training script (with early stopping)
│   ├── predict.py               # Inference helper (predict_risk / predict_batch)
│   └── alert_system.py          # Risk -> alert mapping + notification pipeline
├── models/                    # Saved model, scaler, encoders, metrics, plots
│   ├── disaster_model.keras
│   ├── best_model.keras
│   ├── scaler.pkl
│   ├── type_encoder.pkl
│   ├── feature_columns.json
│   ├── test_metrics.json
│   └── training_history.png
├── templates/
│   └── index.html             # Dashboard UI
└── static/                    # (reserved for CSS/JS/images)
```

A model is already trained and included under `models/` so you can run
predictions and the dashboard immediately without retraining.

---

## 2. Setup

```bash
cd disaster_risk_prediction
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Usage

### Generate the dataset (synthetic — replace with real data for production)
```bash
python data/generate_sample_data.py
```
This creates `data/disaster_data.csv` with 12,000 rows of environmental readings
(rainfall, temperature, humidity, wind speed, river level, soil moisture, seismic
magnitude, pressure) labeled with a disaster type and a derived risk level.

### Train the model
```bash
python src/train.py
```
This will:
- Auto-generate the dataset if it doesn't exist
- Split into train/val/test (70/15/15, stratified by risk level)
- Train a multi-output neural network with early stopping + LR scheduling
- Save `models/disaster_model.keras`, the scaler/encoder, metrics, and a
  training curves plot (`models/training_history.png`)

Typical results on the synthetic dataset: **~88-90% risk-level accuracy** and
**~95-97% disaster-type accuracy** on the held-out test set.

### Run predictions from Python
```python
from src.predict import predict_risk

reading = {
    "rainfall_mm": 145.0,
    "temperature_c": 25.0,
    "humidity_pct": 92.0,
    "wind_speed_kmh": 38.0,
    "river_level_m": 7.1,
    "soil_moisture_pct": 78.0,
    "seismic_magnitude": 0.2,
    "pressure_hpa": 990.0,
}
result = predict_risk(reading)
print(result)
# {'risk_level': 'High', 'predicted_disaster_type': 'flood', ...}
```

Or from the command line:
```bash
python src/predict.py
```

### Run the alert pipeline
```bash
python src/alert_system.py
```
`AlertManager` takes a prediction + sensor reading, builds a structured alert
(severity, recommended action, human-readable message), and dispatches it to
one or more **notifiers**. Included notifiers:
- `console_notifier` — prints to stdout
- `file_log_notifier(path)` — appends JSON lines to a log file
- `webhook_notifier(url)` — POSTs the alert to any webhook (Slack, SMS
  gateway, emergency-management API, etc.)

Only predictions at or above `min_level` (default `"Medium"`) trigger a
dispatch — configurable in `config.yaml` or the `AlertManager(min_level=...)`
constructor.

### Run the web dashboard + REST API
```bash
python app.py
```
Then open **http://localhost:5000** in a browser. The dashboard lets you enter
sensor values and see the predicted risk level, disaster type, and confidence
scores live.

**REST endpoints:**

| Method | Endpoint             | Description                                   |
|--------|-----------------------|------------------------------------------------|
| GET    | `/api/health`          | Health check                                   |
| POST   | `/api/predict`         | Returns risk prediction JSON for a sensor reading |
| POST   | `/api/alert`           | Runs prediction + alert pipeline, returns both |
| GET    | `/api/alerts/recent`   | Last 20 alerts processed by this server        |

Example request:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall_mm": 145, "temperature_c": 25, "humidity_pct": 92,
    "wind_speed_kmh": 38, "river_level_m": 7.1, "soil_moisture_pct": 78,
    "seismic_magnitude": 0.2, "pressure_hpa": 990
  }'
```

---

## 4. Model Architecture

A shared dense trunk (128 → 64 → 32 units, with batch normalization and
dropout for regularization) feeds two output heads:

- **risk_level head** — 4-way softmax (Low / Medium / High / Critical)
- **disaster_type head** — 5-way softmax (none / flood / cyclone / earthquake / drought)

Both heads are trained jointly with weighted categorical cross-entropy losses,
Adam optimizer, early stopping on validation loss, and learning-rate reduction
on plateau.

---

## 5. Using Real Data

Replace `data/disaster_data.csv` with real historical data (e.g. from national
meteorological/seismological agencies, river gauge networks, or IoT sensor
deployments) using the same column schema:

```
rainfall_mm, temperature_c, humidity_pct, wind_speed_kmh, river_level_m,
soil_moisture_pct, seismic_magnitude, pressure_hpa, disaster_type, risk_level
```

Then simply re-run `python src/train.py`.

For real-time deployment, wire live sensor feeds (weather APIs, IoT gateways,
seismograph networks) into `predict_risk()` on a schedule (e.g. every 15
minutes via a cron job or task scheduler), and connect `webhook_notifier` to
your SMS/email/emergency-broadcast provider.

---

## 6. Disclaimer

This project uses **synthetic data** for demonstration and learning purposes.
It is not validated against real disaster events and must not be used for
actual emergency decision-making without retraining on verified historical
data, rigorous evaluation, and review by domain experts and disaster
management authorities.
