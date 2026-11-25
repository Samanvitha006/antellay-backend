import io
import random
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from telemanom_engine import evaluate_predictions, load_channel_data, run_detectors
# 1. Structured API Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("antellay-aegis")

app = FastAPI(
    title="ANTELLAY AEGIS Telemetry API",
    version="2.0.0",
    description="Satellite telemetry anomaly detection, root-cause hypothesis generation, and predictive health API."
)

# 2. Strict CORS Configuration (set ALLOWED_ORIGINS in Render env vars)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://antellay-ui.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# 3. Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "antellay-aegis-backend", "version": "2.0.0"}


# 4. Shared Validation & Analysis Helpers
def validate_and_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Drop completely empty rows/columns
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid data found after removing empty rows/columns.")

    # Ensure timestep exists and remove duplicate timestamps
    if "timestep" in df.columns:
        df = df.drop_duplicates(subset=["timestep"]).reset_index(drop=True)
    elif "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        df["timestep"] = range(len(df))
    else:
        df["timestep"] = range(len(df))

    # Keep numeric feature columns and interpolate/fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["timestep", "Predicted_Anomaly", "Anomaly_Score", "Ground_Truth"]]

    if not feature_cols:
        raise HTTPException(status_code=400, detail="File must contain at least one numeric telemetry column.")

    df[feature_cols] = df[feature_cols].interpolate(method="linear").bfill().ffill()
    return df, feature_cols


def build_root_cause_hypothesis(df: pd.DataFrame, out_df: pd.DataFrame, feature_cols: list) -> dict:
    lower_cols = [c.lower() for c in df.columns]
    anomaly_rows = out_df[out_df["Predicted_Anomaly"] == 1]

    if anomaly_rows.empty:
        return {
            "status": "nominal",
            "disclaimer": "AI-generated hypothesis requiring human review.",
            "primary_suspect": "None detected",
            "confidence": "N/A",
            "evidence": ["No anomalous intervals exceeded detection thresholds."]
        }

    # Identify subsystem only when matching telemetry parameters exist
    if any(k in lower_cols for k in ["voltage", "battery", "soc", "power", "current"]):
        suspect = "Power / Battery Subsystem"
    elif any(k in lower_cols for k in ["temp", "temperature", "thermal"]):
        suspect = "Thermal Control Subsystem"
    elif any(k in lower_cols for k in ["signal", "rssi", "snr", "comm"]):
        suspect = "Communications Subsystem"
    else:
        suspect = "Insufficient subsystem metadata"

    # Compute actual statistical evidence from the primary feature
    primary_col = feature_cols[0]
    mean_val = float(df[primary_col].mean())
    max_anom_val = float(anomaly_rows["value"].max())
    pct_diff = ((max_anom_val - mean_val) / (abs(mean_val) + 1e-6)) * 100

    return {
        "status": "hypothesis",
        "disclaimer": "AI-generated hypothesis requiring human review.",
        "primary_suspect": suspect,
        "confidence": "Medium" if suspect != "Insufficient subsystem metadata" else "Low",
        "evidence": [
            f"Detected {len(anomaly_rows)} anomalous timesteps out of {len(out_df)} total.",
            f"Peak deviation in '{primary_col}': {pct_diff:+.1f}% relative to dataset mean ({mean_val:.2f})."
        ]
    }


def build_predictive_health(df: pd.DataFrame) -> dict:
    lower_cols = [c.lower() for c in df.columns]
    battery_col = None
    for candidate in ["battery", "soc", "battery_pct"]:
        if candidate in lower_cols:
            battery_col = df.columns[lower_cols.index(candidate)]
            break

    if not battery_col:
        return {
            "available": False,
            "battery_level": "N/A",
            "time_to_critical": "N/A",
            "risk_level": "N/A",
            "limitation": "Battery/SoC telemetry column not present in dataset."
        }

    series = df[battery_col].dropna()
    current_val = float(series.iloc[-1])
    # Estimate simple linear drain rate per step if enough points exist
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
        ttc = "N/A"
        risk = "INSUFFICIENT DATA"

    return {
        "available": True,
        "battery_level": f"{current_val:.1f}%",
        "time_to_critical": ttc,
        "risk_level": risk,
        "limitation": "Linear extrapolation based on recent telemetry window; subject to orbital eclipse changes."
    }


# 5. Standardized NASA Benchmark Endpoint (Dynamic Channel)
@app.get("/run-inference")
def analyze_benchmark(channel: str = Query(default="P-1", description="NASA SMAP channel ID, e.g., P-1, T-1")):
    try:
        logger.info(f"Running benchmark inference for channel: {channel}")
        data = load_channel_data(channel)
        out_df = run_detectors(data["train_df"], data["test_df"], data["feature_cols"])
        metrics = evaluate_predictions(out_df, data["anomaly_sequences"])

        return {
            "status": "success",
            "data_source": {
                "type": "public_benchmark",
                "label": f"NASA SMAP Public Benchmark ({channel})",
                "is_live": False
            },
            "metrics": metrics,
            "predictions": out_df[["timestep", "value", "Predicted_Anomaly", "Anomaly_Score"]]
                .tail(150).to_dict(orient="records"),
            "root_cause": build_root_cause_hypothesis(data["test_df"], out_df, data["feature_cols"]),
            "predictive_health": build_predictive_health(data["test_df"]),
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Channel '{channel}' not found in benchmark dataset.")
    except Exception as e:
        logger.exception("Benchmark inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# 6. Standardized Custom Upload Endpoint (CSV + Parquet)
@app.post("/upload-telemetry")
async def upload_custom_telemetry(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".parquet")):
        raise HTTPException(status_code=400, detail="Only .csv and .parquet files are supported.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_parquet(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    df, feature_cols = validate_and_clean_dataframe(df)
    if "value" not in df.columns:
        df["value"] = df[feature_cols[0]]

    # Split 60/40 chronologically to avoid fitting and testing on the exact same rows
    split_idx = max(10, int(len(df) * 0.6))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy() if len(df) > 20 else df.copy()

    out_df = run_detectors(train_df, test_df, feature_cols)

    return {
        "status": "success",
        "data_source": {
            "type": "customer_upload",
            "label": f"Uploaded Telemetry ({filename})",
            "is_live": False
        },
        "metrics": {
            "precision": None,
            "recall": None,
            "f1_score": None,
            "false_positive_rate": None,
            "detection_delay": None,
            "note": "Unlabeled custom dataset — supervised metrics unavailable."
        },
        "predictions": out_df[["timestep", "value", "Predicted_Anomaly", "Anomaly_Score"]]
            .tail(150).to_dict(orient="records"),
        "root_cause": build_root_cause_hypothesis(test_df, out_df, feature_cols),
        "predictive_health": build_predictive_health(test_df),
    }
