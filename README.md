# ECG Apnea Classifier

This project implements a CNN-Transformer model for detecting sleep apnea from ECG signals. The codebase is organized to promote modularity and clarity, facilitating development, testing, and deployment.

## Project Structure

```
.
├── artifacts/              # Stores generated artifacts like scalers.
├── data/                   # Contains all data files.
│   ├── raw/                # Original, immutable raw ECG data.
│   └── processed/          # Preprocessed data (e.g., `apnea-ecg.pkl`).
├── models/                 # Stores trained model files (`.keras`, `.h5`).
├── notebooks/              # Jupyter notebooks for exploratory analysis.
├── results/                # Stores output files like plots (`.png`) and logs (`.csv`).
├── src/                    # All source code for the application.
│   ├── __init__.py         # Marks `src` as a Python package.
│   ├── config.py           # Centralized configuration parameters and file paths.
│   ├── data_loader.py      # Handles loading and preparing processed data for the model.
│   ├── model.py            # Defines the CNN-Transformer model architecture.
│   ├── train.py            # Orchestrates the model training process.
│   ├── evaluate.py         # Provides functions for model evaluation and single-record prediction.
│   ├── preprocessing.py    # Batch processes raw ECG data into the `apnea-ecg.pkl` file.
│   ├── utilities/          # General-purpose helper functions.
│   │   ├── __init__.py     # Marks `utilities` as a Python package.
│   │   └── preprocess.py   # Utility function for on-demand preprocessing of single ECG records.
│   └── main.py             # Command-line interface (CLI) to run pipeline stages.
├── tests/                  # Unit and integration tests.
├── Dockerfile              # Dockerfile for containerizing the application (future).
├── LICENSE
├── README.md
└── requirements.txt        # Project dependencies.
```

## How Files Connect & Their Purpose

The project workflow generally follows these steps, orchestrated by `src/main.py`:

1.  **Preprocessing (`src/preprocessing.py` -> `data/processed/apnea-ecg.pkl`)**:
    *   `src/preprocessing.py` reads raw ECG data from `data/raw/ecgdata` (paths defined in `src/config.py`).
    *   It processes this data to extract features and creates `apnea-ecg.pkl` in `data/processed/`.
2.  **Data Loading (`src/data_loader.py` -> `src/train.py`, `src/evaluate.py`)**:
    *   `src/data_loader.py` loads the `apnea-ecg.pkl` file from `data/processed/`.
    *   It prepares this data (interpolation, normalization) for training and testing, and is used by `src/train.py` and `src/evaluate.py`.
3.  **Model Definition (`src/model.py`)**:
    *   `src/model.py` defines the CNN-Transformer neural network architecture.
    *   `src/train.py` imports and uses this definition to build the model for training.
4.  **Training (`src/train.py`)**:
    *   `src/train.py` takes prepared data from `src/data_loader.py` and the model definition from `src/model.py`.
    *   It trains the model, saving checkpoints and the final model to the `models/` directory (paths from `src/config.py`).
    *   Training logs and plots (accuracy/loss history) are saved to `results/`.
5.  **Evaluation & Prediction (`src/evaluate.py`)**:
    *   `src/evaluate.py` loads trained models from `models/`.
    *   It uses `src/data_loader.py` for test data and generates evaluation metrics, confusion matrices, and ROC curves, saving them to `results/`.
    *   For single-record prediction, it uses `src/utilities/preprocess.py` to prepare individual raw ECG files (from `data/raw/ecgdata`) for the loaded model.
6.  **Utility (`src/utilities/preprocess.py`)**:
    *   `src/utilities/preprocess.py` provides a standalone function to preprocess a *single* raw ECG record, ensuring consistency with the main preprocessing pipeline. It's used by `src/evaluate.py` for real-time inference.
7.  **Configuration (`src/config.py`)**:
    *   `src/config.py` is imported by most other `src` files, providing central access to all necessary parameters and file paths, ensuring consistency across the project.
8.  **Main CLI (`src/main.py`)**:
    *   `src/main.py` is the central command-line interface. It uses `argparse` to allow users to trigger different stages of this pipeline (e.g., `python -m src.main train`, `python -m src.main predict a01`).