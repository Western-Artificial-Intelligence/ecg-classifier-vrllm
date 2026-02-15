# NeuralApnea Triage - ECG Sleep Apnea Triage Tool

This project implements a CNN-Transformer model for detecting sleep apnea from ECG signals. The codebase is organized into a modular, industry-standard structure with a clear separation between the frontend UI and the backend ML API.

## Project Structure

```
.
├── backend/                # Backend Python application
│   ├── api/                # FastAPI application logic
│   │   └── main.py         # Main API entry point
│   ├── src/                # Core ML logic and processing
│   │   ├── cli.py          # Command-line interface for ML pipeline
│   │   ├── config.py       # Centralized configuration (paths, FS, etc.)
│   │   ├── data_loader.py  # Data loading and preparation
│   │   ├── model.py        # CNN-Transformer architecture
│   │   ├── train.py        # Model training logic
│   │   ├── evaluate.py     # Evaluation and prediction utilities
│   │   ├── preprocessing.py# Batch processing pipeline
│   │   └── utilities/      # Helper functions (Grad-CAM, HRV, etc.)
│   └── __init__.py         # Marks backend as a Python package
├── frontend/               # React + Vite frontend application
│   ├── src/                # Frontend source code
│   └── index.html          # Entry point for the web UI
├── data/                   # Centralized data directory
│   ├── raw/                # Original ECG data (.dat files)
│   └── processed/          # Preprocessed features and cache
├── models/                 # Stored trained model files
├── results/                # Output plots, logs, and artifacts
├── artifacts/              # Scalers and other ML metadata
├── notebooks/              # Exploratory analysis notebooks
├── venv/                   # Python virtual environment
├── requirements.txt        # Backend dependencies
└── README.md
```

## How to Run

### Backend API
The backend is a FastAPI application that serves predictions and ECG data to the frontend.

1.  **Activate Virtual Environment**:
    ```powershell
    .\venv\Scripts\activate
    ```
2.  **Start the Server**:
    From the project root:
    ```powershell
    python -m uvicorn backend.api.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

### Frontend UI
The frontend is a React application built with Vite.

1.  **Navigate to Frontend**:
    ```powershell
    cd frontend
    ```
2.  **Start Development Server**:
    ```powershell
    npm run dev
    ```
    The UI will be available at `http://localhost:5173`.

### Backend CLI (ML Pipeline)
You can orchestrate ML stages (preprocessing, training, evaluation) via the CLI. Run these from the project root:

*   **Preprocess Data**:
    ```powershell
    python -m backend.src.cli preprocess
    ```
*   **Train Model**:
    ```powershell
    python -m backend.src.cli train
    ```
*   **Generate Predictions/Plots**:
    ```powershell
    python -m backend.src.cli predict a01
    ```

## Core Components

*   **`backend/api/main.py`**: Handles HTTP requests, loads the trained model, and provides endpoints for the frontend.
*   **`backend/src/config.py`**: The single source of truth for all file paths, sampling frequencies, and model parameters.
*   **`backend/src/model.py`**: Defines the hybrid CNN-Transformer architecture tailored for physiological time-series.
*   **`data/`**: Centralized storage for raw signals and processed features, ensuring data consistency across API and CLI tools.
