"""
alert_system.py
----------------
Turns model predictions into actionable alerts.

- Maps risk level -> alert severity + recommended actions
- Formats human-readable alert messages
- Provides pluggable "notifier" functions (console, email, SMS, webhook)
  so the system can be wired into real notification channels without
  changing the core logic.
"""

import json
import datetime
from typing import Callable, Dict, List, Optional

SEVERITY_CONFIG = {
    "Low": {
        "level": 0,
        "color": "green",
        "action": "Monitor conditions. No action required.",
    },
    "Medium": {
        "level": 1,
        "color": "yellow",
        "action": "Increase monitoring frequency. Inform local authorities on standby.",
    },
    "High": {
        "level": 2,
        "color": "orange",
        "action": "Issue public advisory. Prepare evacuation routes and emergency shelters.",
    },
    "Critical": {
        "level": 3,
        "color": "red",
        "action": "Issue immediate evacuation order. Deploy emergency response teams.",
    },
}


def build_alert(location: str, prediction: dict, sensor_reading: dict) -> dict:
    risk_level = prediction["risk_level"]
    config = SEVERITY_CONFIG[risk_level]

    alert = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "location": location,
        "risk_level": risk_level,
        "severity_score": config["level"],
        "color_code": config["color"],
        "predicted_disaster_type": prediction["predicted_disaster_type"],
        "confidence": round(prediction["risk_confidence"] * 100, 1),
        "recommended_action": config["action"],
        "sensor_snapshot": sensor_reading,
        "message": format_message(location, prediction, config),
    }
    return alert


def format_message(location: str, prediction: dict, config: dict) -> str:
    return (
        f"[{prediction['risk_level'].upper()} RISK ALERT] {location}: "
        f"Model predicts elevated risk of '{prediction['predicted_disaster_type']}' "
        f"with {round(prediction['risk_confidence']*100, 1)}% confidence. "
        f"Recommended action: {config['action']}"
    )


def should_trigger_alert(prediction: dict, min_level: str = "Medium") -> bool:
    threshold = SEVERITY_CONFIG[min_level]["level"]
    return SEVERITY_CONFIG[prediction["risk_level"]]["level"] >= threshold


# ---------------------------------------------------------------------
# Notifier implementations (pluggable). Add real integrations
# (Twilio SMS, SMTP email, Slack/webhook, push notifications, etc.)
# by writing a function with signature: def notifier(alert: dict) -> None
# ---------------------------------------------------------------------

def console_notifier(alert: dict) -> None:
    print("=" * 70)
    print(alert["message"])
    print(json.dumps(alert, indent=2, default=str))
    print("=" * 70)


def file_log_notifier(log_path: str) -> Callable[[dict], None]:
    def _notify(alert: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(alert, default=str) + "\n")
    return _notify


def webhook_notifier(url: str):
    """Returns a notifier that POSTs the alert JSON to a webhook URL
    (e.g. Slack incoming webhook, custom emergency-management API)."""
    def _notify(alert: dict) -> None:
        try:
            import requests
            requests.post(url, json=alert, timeout=5)
        except Exception as e:
            print(f"[webhook_notifier] Failed to send alert: {e}")
    return _notify


class AlertManager:
    """Coordinates prediction -> alert -> multi-channel dispatch."""

    def __init__(self, notifiers: Optional[List[Callable[[dict], None]]] = None,
                 min_level: str = "Medium"):
        self.notifiers = notifiers or [console_notifier]
        self.min_level = min_level
        self.history: List[dict] = []

    def process(self, location: str, prediction: dict, sensor_reading: dict) -> Optional[dict]:
        alert = build_alert(location, prediction, sensor_reading)
        self.history.append(alert)

        if should_trigger_alert(prediction, self.min_level):
            for notify in self.notifiers:
                notify(alert)
            return alert
        return None

    def recent_alerts(self, n: int = 10) -> List[dict]:
        return self.history[-n:]


if __name__ == "__main__":
    # Demo run
    demo_prediction = {
        "risk_level": "High",
        "risk_confidence": 0.87,
        "predicted_disaster_type": "flood",
    }
    demo_reading = {
        "rainfall_mm": 150,
        "river_level_m": 7.4,
    }
    manager = AlertManager(notifiers=[console_notifier], min_level="Medium")
    manager.process("Chennai, Tamil Nadu", demo_prediction, demo_reading)
