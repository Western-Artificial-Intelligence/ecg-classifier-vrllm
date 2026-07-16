# NeuralApnea Triage: Machine Learning Powered ECG Analysis System for Sleep Apnea Detection

**CUCAI Paper:** [NeuralApnea_Triage.pdf](https://drive.google.com/file/d/1NaCSDMGVQSJCKw_JXpoV5d5Il9V-vTxL/view?usp=sharing)

The paper is also viewable in-app via the "Read the Paper" button on the landing page (`frontend/src/components/landing/Hero.tsx`). See the paper for the full model and system architecture diagrams, which aren't reproduced here.

## Authors

- Oliver Olejar
- Daniel Kaminsky
- Annie Liu
- John MacPhie
- Sneha Shah
- Noah Kostesku

## Overview

This project implements a CNN-Transformer model for detecting sleep apnea from ECG signals. The codebase is organized into a modular structure with a clear separation between the frontend UI and the backend ML API: a FastAPI backend serves predictions, Grad-CAM visualizations, and an LLM-based triage agent to a React frontend.

## Project Structure

```
.
├── backend/                    # Backend Python application
│   ├── api/
│   │   └── main.py             # FastAPI app: model loading, prediction/Grad-CAM/agent endpoints
│   ├── src/
│   │   ├── agent.py            # Gemini-based evaluator/triage agent
│   │   ├── cli.py              # CLI for the ML pipeline (preprocess/train/evaluate/predict/agent)
│   │   ├── config.py           # Single source of truth for paths, sampling frequency, model params
│   │   ├── data_loader.py      # Data loading and preparation
│   │   ├── database.py         # SQLite patient/record database manager
│   │   ├── evaluate.py         # Evaluation and single-record prediction
│   │   ├── model.py            # CNN-Transformer architecture
│   │   ├── preprocessing.py    # Batch preprocessing pipeline (training-time)
│   │   ├── train.py            # Training loop (GroupKFold cross-validation)
│   │   └── utilities/
│   │       ├── gradcam.py      # Grad-CAM heatmap generation/visualization
│   │       ├── hrv_edr.py      # HRV / ECG-derived respiration features
│   │       ├── preprocess.py   # Inference-time preprocessing
│   │       ├── rate_limiter.py # API rate limiting
│   │       └── splits.py       # GroupKFold / record-grouped split utilities
│   └── __init__.py
├── frontend/                   # React + Vite + TypeScript frontend
│   ├── src/
│   │   ├── components/         # App views (patient dashboard, ECG chart, analysis, etc.)
│   │   │   └── landing/        # Landing page sections (Hero, Architecture, Demo, FAQ, etc.)
│   │   ├── services/api.ts     # API client to the backend
│   │   └── styles/             # Per-component CSS modules
│   └── index.html              # Entry point for the web UI
├── data/                        # Centralized data directory
│   ├── raw/                    # Original PhysioNet Apnea-ECG WFDB records
│   └── processed/               # Preprocessed features, cache, Grad-CAM images, agent reports
├── models/                      # Trained model checkpoints (per-fold + final)
├── results/                     # Evaluation plots, metrics CSVs, run metadata
├── artifacts/                   # Scaler and other ML metadata (e.g. scaler.joblib)
├── venv/                        # Python virtual environment
├── requirements.txt             # Backend dependencies
└── README.md
```

## How to Run

### Backend API

The backend is a FastAPI application that serves predictions and ECG data to the frontend.

1. **Activate the virtual environment**:
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Start the server** (from the project root):
   ```powershell
   python -m uvicorn backend.api.main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.

### Frontend UI

The frontend is a React application built with Vite.

1. **Navigate to the frontend directory**:
   ```powershell
   cd frontend
   ```
2. **Install dependencies and start the dev server**:
   ```powershell
   npm install
   npm run dev
   ```
   The UI will be available at `http://localhost:5173`.

### Backend CLI (ML Pipeline)

Orchestrate ML stages via the CLI. Run these from the project root:

- **Preprocess data**:
  ```powershell
  python -m backend.src.cli preprocess
  ```
- **Train the model**:
  ```powershell
  python -m backend.src.cli train
  ```
- **Generate predictions/plots for a record**:
  ```powershell
  python -m backend.src.cli predict a01
  ```
- **Run the evaluator agent**:
  ```powershell
  python -m backend.src.cli agent a01
  ```
  This requires a `GEMINI_API_KEY` environment variable to be set (see `.env`).

## Core Components

- **`backend/api/main.py`**: Handles HTTP requests, loads the trained model, and provides prediction, Grad-CAM, and agent endpoints for the frontend.
- **`backend/src/config.py`**: The single source of truth for all file paths, sampling frequencies, and model parameters.
- **`backend/src/model.py`**: Defines the hybrid CNN-Transformer architecture tailored for physiological time-series.
- **`backend/src/database.py`**: Manages the SQLite database of patients and ECG records used by the API and UI.
- **`backend/src/agent.py`**: Gemini-powered agent that generates triage reports from model predictions and Grad-CAM visualizations.
- **`backend/src/utilities/gradcam.py`**: Generates Grad-CAM heatmaps highlighting the ECG regions driving each prediction.
- **`backend/src/utilities/preprocess.py`**: Inference-time signal preprocessing used by both the API and CLI.
- **`data/`**: Centralized storage for raw signals and processed features, ensuring data consistency across API and CLI tools.
