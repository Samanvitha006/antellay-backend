import io
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _make_sample_csv(include_ground_truth: bool = False) -> bytes:
    """Generate a 60-row telemetry CSV with an injected power/thermal anomaly."""
    np.random.seed(42)
    n = 60
    rows = []
    for t in range(n):
        if t < 45:
            batt = 92.0 - 0.1 * t
            volt = 28.0 + np.random.normal(0, 0.03)
            temp = 24.0 + np.random.normal(0, 0.1)
            gt = 0
        elif t < 53:
            batt = 87.5 - 1.5 * (t - 44)
            volt = 24.0 + np.random.normal(0, 0.05)
            temp = 38.0 + np.random.normal(0, 0.2)
            gt = 1
        else:
            batt = 75.5 - 1.0 * (t - 52)
            volt = 26.5 + np.random.normal(0, 0.03)
            temp = 30.0 + np.random.normal(0, 0.1)
            gt = 0

        row = {
            "timestep": t,
            "battery_pct": round(batt, 2),
            "voltage": round(volt, 3),
            "temperature": round(temp, 2),
        }
        if include_ground_truth:
            row["ground_truth"] = gt
        rows.append(row)

    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["service"] == "antellay-aegis-backend"


def test_channels_endpoint():
    res = client.get("/channels")
    assert res.status_code == 200
    body = res.json()
    assert "channels" in body
    assert isinstance(body["channels"], list)
    assert len(body["channels"]) > 0


def test_benchmark_inference():
    res = client.get(
        "/run-inference",
        params={
            "channel": "P-1",
            "detector_type": "Isolation Forest (Unsupervised ML)",
            "threshold_pct": 1.0,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["data_source"]["is_live"] is False
    assert len(body["predictions"]) > 0
    assert "metrics" in body
    assert "root_cause" in body
    assert "predictive_health" in body


def test_upload_unlabeled_csv():
    csv_bytes = _make_sample_csv(include_ground_truth=False)
    res = client.post(
        "/upload-telemetry",
        params={"detector_type": "Isolation Forest (Unsupervised ML)", "threshold_pct": 2.0},
        files={"file": ("unlabeled.csv", csv_bytes, "text/csv")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["data_source"]["type"] == "customer_upload"
    # Unlabeled files must return None / N/A for supervised metrics
    assert body["metrics"]["precision"] is None
    assert body["metrics"]["f1"] is None
    # Battery column is present, so predictive health must be computed
    assert body["predictive_health"]["available"] is True
    assert body["root_cause"]["primary_suspect"] == "Power / Battery Subsystem"


def test_upload_labeled_csv():
    csv_bytes = _make_sample_csv(include_ground_truth=True)
    res = client.post(
        "/upload-telemetry",
        params={"detector_type": "Isolation Forest (Unsupervised ML)", "threshold_pct": 2.0},
        files={"file": ("labeled.csv", csv_bytes, "text/csv")},
    )
    assert res.status_code == 200
    body = res.json()
    # Labeled files must dynamically compute supervised metrics
    assert body["metrics"]["precision"] is not None
    assert body["metrics"]["recall"] is not None
    assert body["metrics"]["f1"] is not None
    assert body["metrics"]["total_sequences"] == 1


def test_upload_invalid_file_type():
    res = client.post(
        "/upload-telemetry",
        files={"file": ("notes.txt", b"not a csv", "text/plain")},
    )
    assert res.status_code == 400
