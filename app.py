"""
app.py
------
Flask web application exposing:
  GET  /                -> simple dashboard UI
  POST /api/predict      -> JSON in, risk prediction JSON out
  POST /api/alert        -> JSON in, runs prediction + alert pipeline
  GET  /api/health       -> health check

Run with:
    python app.py
Then open http://localhost:5000
"""

import os
import sys
from flask import Flask, request, jsonify, render_template

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import predict_risk, FEATURE_COLUMNS
from alert_system import AlertManager, console_notifier, file_log_notifier

app = Flask(__name__)

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "alerts_log.jsonl")
alert_manager = AlertManager(
    notifiers=[console_notifier, file_log_notifier(LOG_PATH)],
    min_level="Medium",
)


@app.route("/")
def index():
    return render_template("index.html", features=FEATURE_COLUMNS)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        result = predict_risk({k: float(data[k]) for k in FEATURE_COLUMNS})
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({
            "error": "Model not found. Train the model first with: python src/train.py"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alert", methods=["POST"])
def api_alert():
    data = request.get_json(force=True)
    location = data.get("location", "Unknown location")
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        sensor_reading = {k: float(data[k]) for k in FEATURE_COLUMNS}
        prediction = predict_risk(sensor_reading)
        alert = alert_manager.process(location, prediction, sensor_reading)
        return jsonify({
            "prediction": prediction,
            "alert_triggered": alert is not None,
            "alert": alert,
        })
    except FileNotFoundError:
        return jsonify({
            "error": "Model not found. Train the model first with: python src/train.py"
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts/recent")
def recent_alerts():
    return jsonify(alert_manager.recent_alerts(20))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
