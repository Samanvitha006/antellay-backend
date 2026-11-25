# Antellay Space — AEGIS Spacecraft Telemetry Anomaly Detection

**AEGIS** is a decoupled, full-stack spacecraft telemetry anomaly-detection, root-cause triage, and predictive health platform. It is validated against the public **NASA Telemanom (SMAP Satellite & MSL Curiosity Rover)** benchmark dataset and supports leak-free inference on custom `.csv` and `.parquet` telemetry uploads.

* **Live Frontend UI:** `https://antellay-ui.vercel.app`
* **Live Backend API:** `https://antellay-backend.onrender.com`
* **Interactive API Docs (Swagger):** `https://antellay-backend.onrender.com/docs`

---

## 1. System Architecture

| Layer | Technology | Hosting / Deployment | Responsibility |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | Next.js (React, JavaScript) | Vercel | Interactive time-series visualization, dynamic metric cards, custom file upload interface, and triage/RCA panels. |
| **Backend API** | FastAPI (Python 3.11, Uvicorn) | Render | Telemetry ingestion, validation, leak-free ML inference, supervised metric evaluation, and predictive health computation. |
| **Containerization** | Docker (`python:3.11-slim`) | Local / Cloud Portable | Standardized runtime environment via `Dockerfile` and `.dockerignore` exposing port `8080`. |
| **Verification** | `pytest` + FastAPI `TestClient` | Local / CI (`tests/test_api.py`) | Automated end-to-end validation of health, benchmark inference, labeled/unlabeled uploads, and input validation. |

### Core API Endpoints
* **`GET /health`** — Service liveness probe returning version and status metadata.
* **`GET /channels`** — Lists available NASA Telemanom benchmark channels.
* **`GET /run-inference`** — Executes anomaly detection on a selected NASA benchmark channel (`Isolation Forest` or `Rolling Robust Baseline`) with configurable tail threshold (`0.5%–10.0%`).
* **`POST /upload-telemetry`** — Ingests user-supplied `.csv` or `.parquet` telemetry files (up to 10 MB), cleans/interpolates missing values, runs leak-free inference, and returns dynamic metrics, RCA hypotheses, and predictive health.
* **`GET /docs`** — Auto-generated OpenAPI / Swagger interactive API documentation.

### Security & CORS Policy
* `CORSMiddleware` restricts API access to configured production origins (`https://antellay-ui.vercel.app`) and Vercel preview deployments via `allow_origin_regex`.
* Uploaded files are strictly validated for extension (`.csv`, `.parquet`), size (`<= 10 MB`), and presence of numeric telemetry columns before processing.

---

## 2. Public Dataset Access & Data Provenance

AEGIS eliminates hardcoded dashboard states and explicitly labels data provenance on every inference run:

* **`[PUBLIC BENCHMARK]` (NASA Telemanom):**
  * **Source:** Loaded dynamically via the Hugging Face Parquet mirror (`appleparan/telemanom`) and the official ground-truth interval file (`khundman/telemanom/labeled_anomalies.csv`).
  * **Missions:** NASA SMAP Earth-observation satellite (55 channels) and MSL Curiosity rover (27 channels).
  * **Units & Scaling Caveat:** Telemetry values are anonymized and dimensionless, and timestamps are relative step indices—they do not represent live physical °C, V, or W engineering units.
* **`[SYNTHETIC DEMO]`:** Offline fallback telemetry with injected anomalies used only when remote benchmark assets are unreachable.
* **`[CUSTOMER UPLOAD]`:** User-provided `.csv` / `.parquet` telemetry files, explicitly indicating whether supervised metrics were computed from `ground_truth` labels or marked `N/A`.

---

## 3. ML Methodology & Data Leakage Prevention

A core design requirement of AEGIS is **zero data leakage** between training calibration and test evaluation:

1. **Detectors:**
   * **Rolling Robust Baseline:** Computes causal rolling median and Median Absolute Deviation (MAD) residuals calibrated strictly on the held-out training split.
   * **Isolation Forest (Unsupervised ML):** Fits a multivariate Isolation Forest (`n_estimators=150`) across telemetry channels and command states. Both `StandardScaler` parameters and percentile decision thresholds are fit exclusively on training data.
2. **Strict Chronological Splitting:** For custom telemetry uploads, datasets are split chronologically (**60% train / 40% test**) rather than randomly shuffled, preserving temporal causality.
3. **Feature Isolation:** Metadata and label columns (`timestep`, `ground_truth`, `Ground_Truth`, `Predicted_Anomaly`, `Anomaly_Score`) are explicitly stripped from `feature_cols` prior to model fitting and scoring.
4. **Dynamic Evaluation:**
   * When `ground_truth` labels are present (NASA benchmarks or labeled custom uploads), AEGIS dynamically computes **Pointwise Precision**, **Recall**, **F1 Score**, **False Positive Rate (FPR)**, **Sequence-Level Detection Rate**, and **Mean Detection Delay**.
   * When a custom upload is unlabeled, the API and UI transparently return `N/A` rather than synthetic or hardcoded placeholder scores.

---

## 4. Root Cause Analysis (RCA) & Predictive Health

* **Root Cause Analysis:** Evaluates anomalous feature deviations (correlated signal Z-scores) and maps them to physical subsystems (**Power / Battery**, **Thermal Control**, **Communications**) *only* when matching physical sensor metadata exists in the dataset. For anonymized benchmarks, it explicitly reports `Anonymized / Insufficient Subsystem Metadata`. All outputs carry an **AI-generated hypothesis requiring human engineering review** disclaimer.
* **Predictive Health:** Checks for Battery / State-of-Charge (`battery`, `soc`, `battery_pct`) columns. When present, it computes a linear discharge slope over the trailing 5-timestep window to estimate **Time to Critical (40% SoC)** and assigns a risk tier (`LOW RISK`, `MODERATE`, `HIGH RISK`). When absent, the panel honestly reports `UNAVAILABLE / N/A`.

---

## 5. Known Limitations & Trade-offs

1. **Benchmark Windowing (`P-1` Row Cap):** To keep inference latency low on free-tier cloud instances, benchmark series are capped in length. On channels where ground-truth anomalies occur beyond the cap (e.g., `P-1`), `SEQ CAUGHT` evaluates to `0 / 0` and supervised metrics evaluate to `0.0%` / `N/A`. Channels with earlier anomaly windows (e.g., `S-1`, `E-1`, `T-1`) or labeled custom uploads demonstrate full supervised scoring.
2. **Linear Extrapolation in Predictive Health:** Time-to-critical estimation uses a short-window linear slope over battery/SoC telemetry. It does not model non-linear battery chemistry, orbital eclipse cycles, or sudden load shedding; human operator verification is required.
3. **Heuristic Subsystem Mapping:** Root-cause subsystem attribution relies on column-name semantics (`voltage`, `temp`, `signal`) combined with statistical deviation scores rather than a physical spacecraft digital twin or fault tree.
4. **Cloud Cold Starts:** Hosting the FastAPI backend on Render's free tier may introduce a 30–50 second cold-start delay on the first request after inactivity.
5. **Non-Live Batch Architecture:** AEGIS operates on batch/historical telemetry files and benchmark channels rather than a live, streaming satellite downlink (e.g., Kafka/MQTT).

---

## 6. Setup & Reproducible Local Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated test suite
pytest -v

# 3. Run the FastAPI backend locally
uvicorn main:app --reload --port 8080

# 4. Build and run via Docker
docker build -t antellay-backend .
docker run -p 8080:8080 antellay-backend
```
