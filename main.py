import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from telemanom_engine import evaluate_predictions, load_channel_data, run_detectors

app = FastAPI(title="Antellay AEGIS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Antellay AEGIS Backend Online"}

@app.get("/run-inference")
def analyze_default():
    data = load_channel_data("P-1")
    out_df = run_detectors(data["train_df"], data["test_df"], data["feature_cols"])
    metrics = evaluate_predictions(out_df, data["anomaly_sequences"])
    return {
        "status": "success",
        "metrics": metrics,
        "predictions": out_df[
            ["timestep", "value", "Predicted_Anomaly", "Anomaly_Score"]
        ]
        .tail(100)
        .to_dict(orient="records"),
    }

@app.post("/api/analyze/custom")
async def analyze_custom(file: UploadFile = File(...)):
    contents = await file.read()
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_parquet(io.BytesIO(contents))

    feature_cols = [c for c in df.columns if c not in ["timestep", "ground_truth"]]
    out_df = run_detectors(df, df, feature_cols)

    return {
        "status": "success",
        "source": file.filename,
        "predictions": out_df[
            ["timestep", "value", "Predicted_Anomaly", "Anomaly_Score"]
        ]
        .tail(100)
        .to_dict(orient="records"),
    }

@app.post("/upload-telemetry")
async def upload_custom_telemetry(file: UploadFile = File(...)):
    # Read the uploaded file into memory
    contents = await file.read()

    # (Your inference logic for the uploaded CSV/Parquet file goes here)

    return {
        "filename": file.filename, 
        "status": "success", 
        "message": "File received and processed."
    }
