# Antellay Space — AEGIS Spacecraft Telemetry Anomaly Detection

AEGIS is a spacecraft telemetry anomaly-detection and triage dashboard validated against the public **NASA Telemanom (SMAP Satellite & MSL Curiosity Rover)** benchmark dataset.

## 1. Setup & Reproducible Execution
```bash
pip install -r requirements.txt
# Run unit tests (missing values, invalid inputs, duplicate timestamps, empty datasets)
pytest tests/test_adapter.py
# Launch the Streamlit dashboard (automatically loads channel P-1 by default)
streamlit run app.py
## 2. Public Dataset Access & Provenance
* **Source**: Loaded dynamically via the Hugging Face Parquet mirror (`appleparan/telemanom`)[cite: 4] and the official ground-truth interval file (`khundman/telemanom/labeled_anomalies.csv`)[cite: 4].
* **Missions**: NASA SMAP Earth-observation satellite (55 channels) and MSL Curiosity rover (27 channels)[cite: 4]. Note: MSL is a rover benchmark used alongside SMAP for spacecraft telemetry validation[cite: 4].
* **Units & Scaling Caveat**: All telemetry values in Telemanom are anonymized and scaled (dimensionless)[cite: 4] and timestamps are anonymized relative step indices[cite: 4]. They do not represent live physical °C, V, or W engineering units[cite: 4].

## 3. Model Training & Leak-Free Thresholding
* **Rolling Robust Baseline**: Computes rolling median and Median Absolute Deviation (MAD) residuals[cite: 5] calibrated strictly on the held-out training split[cite: 5].
* **Isolation Forest (Unsupervised ML)**: Fits a multivariate Isolation Forest (`n_estimators=150`) across the primary telemetry channel (`value`) and spacecraft command states (`cmd_*`)[cite: 5]. Both `StandardScaler` parameters and percentile decision thresholds are fit exclusively on train data to prevent test data leakage[cite: 5].

## 4. Evaluation Methodology
Predictions on held-out test data are compared directly against the benchmark's `anomaly_sequences` intervals[cite: 5], reporting:
* Pointwise Precision, Recall, F1 Score, and False Positive Rate (FPR)[cite: 5].
* Sequence-level detection rate and Mean Detection Delay (timesteps between interval onset and first alert)[cite: 5].

## 5. Operational Boundaries & Limitations
* This system is validated on a historical research benchmark[cite: 5], not a live operating spacecraft feed[cite: 5].
* Root-cause attributions represent statistical hypotheses derived from correlated signal Z-scores[cite: 5] and require human flight-controller review before any operational action is taken[cite: 5].
## Step 6: Commit and Push to Update Your Live Streamlit App

Once `pytest` passes and `streamlit run app.py` loads locally in your browser, push your changes to GitHub:

```bash
git add requirements.txt telemanom_engine.py app.py tests/test_adapter.py README.md
git commit -m "Validate AEGIS against NASA Telemanom benchmark with evaluation metrics and unit tests"
git push origin main


# ANTELLAY AEGIS — System Architecture & Technical Limitations

## 1. System Architecture

AEGIS is a decoupled, full-stack telemetry anomaly detection and triage platform built for reproducibility, zero data leakage, and transparent reporting.

| Layer | Technology | Hosting / Deployment | Responsibility |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | Next.js (React, JavaScript) | Vercel (`https://antellay-ui.vercel.app`) | Interactive time-series visualization, dynamic metric cards, custom file upload interface, and triage/RCA panels. |
| **Backend API** | FastAPI (Python 3.11, Uvicorn) | Render (`https://antellay-backend.onrender.com`) | Telemetry ingestion, validation, leak-free ML inference, supervised metric evaluation, and predictive health computation. |
| **Containerization** | Docker (`python:3.11-slim`) | Local / Cloud Portable | Standardized runtime environment via `Dockerfile` and `.dockerignore` exposing port `8080`. |
| **Verification** | `pytest` + FastAPI `TestClient` | Local / CI (`tests/test_api.py`) | Automated end-to-end validation of health, benchmark inference, labeled/unlabeled uploads, and input validation. |

### Core API Endpoints
* **`GET /health`** — Service liveness probe returning version and status metadata.
* **`GET /channels`** — Lists available NASA Telemanom benchmark channels.
* **`GET /run-inference`** — Executes anomaly detection on a selected NASA benchmark channel (`Isolation Forest` or `Rolling Robust Baseline`) with configurable tail threshold (`0.5%–10.0%`).
* **`POST /upload-telemetry`** — Ingests user-supplied `.csv` or `.parquet` telemetry files (up to 10 MB), cleans/interpolates missing values, runs leak-free inference, and returns dynamic metrics, RCA hypotheses, and predictive health.
* **`GET /docs`** — Auto-generated OpenAPI / Swagger interactive API documentation.

### Security & CORS Policy
* Cross-Origin Resource Sharing (`CORSMiddleware`) restricts API access to configured production origins (`https://antellay-ui.vercel.app`) and Vercel preview deployments via `allow_origin_regex`.
* Uploaded files are strictly validated for extension (`.csv`, `.parquet`), size (`<= 10 MB`), and presence of numeric telemetry columns before processing.

---

## 2. ML Methodology & Data Leakage Prevention

A core design requirement of AEGIS is **zero data leakage** between training calibration and test evaluation:

1. **Strict Chronological Splitting:** For custom telemetry uploads, datasets are split chronologically (**60% train / 40% test**) rather than randomly shuffled, preserving temporal causality.
2. **Causal Baseline Computation:** Rolling robust baselines and anomaly thresholds are calibrated exclusively on historical/training distributions without peeking at future test windows.
3. **Feature Isolation:** Metadata and label columns (`timestep`, `ground_truth`, `Ground_Truth`, `Predicted_Anomaly`, `Anomaly_Score`) are explicitly stripped from `feature_cols` prior to model fitting and scoring.
4. **Dynamic Evaluation:**
   * When `ground_truth` labels are present (NASA benchmarks or labeled custom uploads), AEGIS dynamically computes **Precision**, **Recall**, **F1 Score**, **False Positive Rate (FPR)**, **Sequences Caught**, and **Mean Detection Delay**.
   * When a custom upload is unlabeled, the API and UI transparently return `N/A` rather than synthetic or hardcoded placeholder scores.

---

## 3. Data Provenance & Honest UI Disclosures

AEGIS eliminates hardcoded dashboard states and explicitly labels data provenance on every inference run:

* **`[PUBLIC BENCHMARK]`:** Anonymized, dimensionless NASA Telemanom spacecraft telemetry. Marked as `NON-LIVE / ANALYTICS`.
* **`[SYNTHETIC DEMO]`:** Offline fallback telemetry with injected anomalies used only when remote benchmark assets are unreachable.
* **`[CUSTOMER UPLOAD]`:** User-provided `.csv` / `.parquet` telemetry files, explicitly indicating whether supervised metrics were computed from `ground_truth` labels or marked `N/A`.

---

## 4. Root Cause Analysis (RCA) & Predictive Health

* **Root Cause Analysis:** Evaluates anomalous feature deviations and maps them to physical subsystems (**Power / Battery**, **Thermal Control**, **Communications**) *only* when matching physical sensor metadata exists in the dataset. For anonymized benchmarks (e.g., NASA Telemanom), it explicitly reports `Anonymized / Insufficient Subsystem Metadata`. All outputs carry an **AI-generated hypothesis requiring human engineering review** disclaimer.
* **Predictive Health:** Actively checks for Battery / State-of-Charge (`battery`, `soc`, `battery_pct`) columns. When present, it calculates a linear discharge slope over the trailing 5-timestep window to estimate **Time to Critical (40% SoC)** and assigns a risk tier (`LOW RISK`, `MODERATE`, `HIGH RISK`). When absent, the panel honestly reports `UNAVAILABLE / N/A`.

---

## 5. Known Limitations & Trade-offs

1. **Benchmark Windowing (`P-1` Row Cap):** To keep inference latency low on free-tier cloud instances, benchmark series are capped in length. On channels where ground-truth anomalies occur beyond the cap (e.g., `P-1`), `SEQ CAUGHT` evaluates to `0 / 0` and supervised metrics evaluate to `0.0%` / `N/A`. Channels with earlier anomaly windows (e.g., `S-1`, `E-1`, `T-1`) or labeled custom uploads demonstrate full supervised scoring.
2. **Linear Extrapolation in Predictive Health:** Time-to-critical estimation uses a short-window linear slope over battery/SoC telemetry. It does not model non-linear battery chemistry, orbital eclipse cycles, or sudden load shedding; human operator verification is required.
3. **Heuristic Subsystem Mapping:** Root-cause subsystem attribution relies on column-name semantics (`voltage`, `temp`, `signal`) combined with statistical deviation scores rather than a physical spacecraft digital twin or fault tree.
4. **Cloud Cold Starts:** Hosting the FastAPI backend on Render's free tier may introduce a 30–50 second cold-start delay on the first request after inactivity.
5. **Non-Live Batch Architecture:** AEGIS operates on batch/historical telemetry files and benchmark channels rather than a live, streaming satellite downlink (e.g., Kafka/MQTT).

---

## 6. Reproducibility & Local Verification

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated test suite
pytest -v

# 3. Run locally via Uvicorn
uvicorn main:app --reload --port 8080

# 4. Build and run via Docker
docker build -t antellay-backend .
docker run -p 8080:8080 antellay-backend
