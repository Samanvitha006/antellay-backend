import io
import os
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from telemanom_engine import (
    BENCHMARK_CHANNELS,
    load_channel_data,
    run_detectors,
    evaluate_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("antellay-aegis")

app = FastAPI(
    title="ANTELLAY AEGIS Telemetry API",
    version="2.0.0",
    description="Satellite telemetry anomaly detection, root-cause hypothesis generation, and predictive health API.",
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://antellay-ui.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_origin_regex=r"https://antellay-.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "antellay-aegis-backend", "version": "2.0.0"}


@app.get("/channels")
def list_channels():
    return {"channels": BENCHMARK_CHANNELS}


def validate_and_clean_dataframe(df: pd.DataFrame):
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data found after removing empty rows/columns.")

    if "timestep" in df.columns:
        df = df.drop_duplicates(subset=["timestep"]).reset_index(drop=True)
    elif "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        df["timestep"] = range(len(df))
    else:
        df["timestep"] = range(len(df))

    # Normalize optional Ground_Truth column name to ground_truth
    if "Ground_Truth" in df.columns and "ground_truth" not in df.columns:
        df["ground_truth"] = df["Ground_Truth"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in numeric_cols
        if c not in ["timestep", "Predicted_Anomaly", "Anomaly_Score", "ground_truth", "Ground_Truth"]
    ]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="File must contain at least one numeric telemetry column.")

    df[feature_cols] = df[feature_cols].interpolate(method="linear").bfill().ffill()
    return df, feature_cols


def extract_sequences_from_labels(gt_series: pd.Series) -> list[list[int]]:
    gt = gt_series.fillna(0).astype(int).values
    sequences, start = [], None
    for i, val in enumerate(gt):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            sequences.append([start, i - 1])
            start = None
    if start is not None:
        sequences.append([start, len(gt) - 1])
    return sequences


def build_predictive_health(df: pd.DataFrame) -> dict:
    lower_cols = [c.lower() for c in df.columns]
    battery_col = next((df.columns[lower_cols.index(k)] for k in ["battery", "soc", "battery_pct"] if k in lower_cols), None)

    if not battery_col:
        return {
            "available": False,
            "battery_level": "N/A",
            "time_to_critical": "N/A",
            "risk_level": "N/A",
            "limitation": "Battery/SoC telemetry parameter not present in this dataset.",
        }

    series = df[battery_col].dropna()
    current_val = float(series.iloc[-1])
    if len(series) >= 5:
        slope = float(series.iloc[-1] - series.iloc[-5]) / 5.0
        if slope < 0 and current_val > 40:
            steps_to_40 = int((current_val - 40) / abs(slope))
            ttc = f"~ {steps_to_40} steps"
            risk = "HIGH RISK" if steps_to_40 < 20 else "MODERATE"
        else:
            ttc = "Stable / Charging"
            risk = "LOW RISK"
    else:
        ttc, risk = "N/A", "INSUFFICIENT DATA"

    return {
        "available": True,
        "battery_level": f"{current_val:.1f}%",
        "time_to_critical": ttc,
        "risk_level": risk,
        "limitation": "Linear extrapolation over recent window; human review required.",
    }


def extract_triage_and_rca(out_df: pd.DataFrame, df: pd.DataFrame) -> tuple[dict, list]:
    anomalies = out_df[out_df["Predicted_Anomaly"] == 1]
    lower_cols = [c.lower() for c in df.columns]

    if any(k in lower_cols for k in ["voltage", "battery", "soc", "power", "current"]):
        suspect = "Power / Battery Subsystem"
    elif any(k in lower_cols for k in ["temp", "temperature", "thermal"]):
        suspect = "Thermal Control Subsystem"
    elif any(k in lower_cols for k in ["signal", "rssi", "snr", "comm"]):
        suspect = "Communications Subsystem"
    else:
        suspect = "Anonymized / Insufficient Subsystem Metadata"

    if anomalies.empty:
        return (
            {
                "status": "nominal",
                "disclaimer": "AI-generated hypothesis requiring human engineering review.",
                "primary_suspect": "None detected",
                "evidence": ["No anomalous intervals exceeded the calibrated threshold."],
                "triage_action": "Continue nominal monitoring.",
            },
            [],
        )

    latest_anom = anomalies.iloc[-1]
    triage_cols = [
        c for c in [
            "timestep", "value", "baseline_value", "Anomaly_Score", "Severity",
            "Supporting_Evidence", "Hypothesis", "Triage_Action", "ground_truth"
        ]
        if c in anomalies.columns
    ]
    triage_log = anomalies[triage_cols].tail(15).to_dict(orient="records")

    root_cause = {
        "status": "hypothesis",
        "disclaimer": "AI-generated hypothesis requiring human engineering review.",
        "primary_suspect": suspect,
        "evidence": [
            str(latest_anom.get("Supporting_Evidence", "Statistical deviation detected.")),
            str(latest_anom.get("Hypothesis", "Correlated channel anomaly hypothesis.")),
        ],
        "triage_action": str(latest_anom.get("Triage_Action", "Human review required before action.")),
    }
    return root_cause, triage_log


@app.get("/run-inference")
def analyze_benchmark(
    channel: str = Query(default="P-1"),
    detector_type: str = Query(default="Isolation Forest (Unsupervised ML)"),
    threshold_pct: float = Query(default=1.0, ge=0.5, le=10.0),
):
    try:
        logger.info(f"Benchmark inference: channel={channel}, model={detector_type}, thresh={threshold_pct}")
        bundle = load_channel_data(channel)
        out_df = run_detectors(
            bundle["train_df"],
            bundle["test_df"],
            bundle["feature_cols"],
            detector_type=detector_type,
            threshold_pct=threshold_pct,
        )
        metrics = evaluate_predictions(out_df, bundle["anomaly_sequences"])
        root_cause, triage_log = extract_triage_and_rca(out_df, bundle["test_df"])

        pred_cols = [
            c for c in ["timestep", "value", "baseline_value", "Predicted_Anomaly", "Anomaly_Score", "Severity", "ground_truth"]
            if c in out_df.columns
        ]

        return {
            "status": "success",
            "data_source": {
                "type": "synthetic_demo" if bundle.get("is_synthetic") else "public_benchmark",
                "label": (
                    f"Synthetic Demo Data · Offline Fallback ({channel})"
                    if bundle.get("is_synthetic")
                    else f"NASA Telemanom Public Benchmark ({bundle['spacecraft']} · {channel})"
                ),
                "is_live": False,
                "disclosure": (
                    "Offline synthetic fallback telemetry with an injected anomaly. Not live spacecraft data."
                    if bundle.get("is_synthetic")
                    else "Anonymized, dimensionless public benchmark telemetry. Not a live spacecraft feed."
                ),
            },
            "metrics": metrics,
            "predictions": out_df[pred_cols].tail(200).to_dict(orient="records"),
            "root_cause": root_cause,
            "triage_log": triage_log,
            "predictive_health": build_predictive_health(bundle["test_df"]),
        }
    except Exception as e:
        logger.exception("Benchmark inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-telemetry")
async def upload_custom_telemetry(
    file: UploadFile = File(...),
    detector_type: str = Query(default="Isolation Forest (Unsupervised ML)"),
    threshold_pct: float = Query(default=1.0, ge=0.5, le=10.0),
):
    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".parquet")):
        raise HTTPException(status_code=400, detail="Only .csv and .parquet files are supported.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(contents)) if filename.endswith(".csv") else pd.read_parquet(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    df, feature_cols = validate_and_clean_dataframe(df)
    if "value" not in df.columns:
        df["value"] = df[feature_cols[0]]

    # Chronological 60/40 split to prevent data leakage
    split_idx = max(10, int(len(df) * 0.6))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy() if len(df) > 20 else df.copy()

    out_df = run_detectors(train_df, test_df, feature_cols, detector_type=detector_type, threshold_pct=threshold_pct)
    has_ground_truth = "ground_truth" in test_df.columns

    if has_ground_truth:
        out_df["ground_truth"] = test_df["ground_truth"].fillna(0).astype(int).values
        sequences = extract_sequences_from_labels(test_df["ground_truth"])
        metrics = evaluate_predictions(out_df, sequences)
        disclosure = "User-uploaded telemetry file with ground-truth labels — supervised metrics dynamically computed."
    else:
        metrics = {
            "precision": None,
            "recall": None,
            "f1": None,
            "fpr": None,
            "detected_sequences": None,
            "total_sequences": None,
            "mean_detection_delay": "N/A",
        }
        disclosure = "User-uploaded telemetry file without ground-truth labels — supervised metrics are N/A."

    root_cause, triage_log = extract_triage_and_rca(out_df, test_df)

    pred_cols = [
        c for c in ["timestep", "value", "baseline_value", "Predicted_Anomaly", "Anomaly_Score", "Severity", "ground_truth"]
        if c in out_df.columns
    ]

    return {
        "status": "success",
        "data_source": {
            "type": "customer_upload",
            "label": f"Uploaded Telemetry ({filename})",
            "is_live": False,
            "disclosure": disclosure,
        },
        "metrics": metrics,
        "predictions": out_df[pred_cols].tail(200).to_dict(orient="records"),
        "root_cause": root_cause,
        "triage_log": triage_log,
        "predictive_health": build_predictive_health(test_df),
    }
