"""
Configuration file for the ECG Apnea project.

This file contains all the configuration variables for the project,
such as file paths, model parameters, and preprocessing settings.
"""

import os
from multiprocessing import cpu_count
from dotenv import load_dotenv

load_dotenv()

# -- File Paths --
# Using os.path.join for platform compatibility
# Note: These paths are relative to the project root directory

# Path to the raw ECG data
RAW_DATA_DIR = os.path.join("data", "raw")

# Path to the processed data
PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Path to the trained models
MODELS_DIR = os.path.join("models")

# Path to the results
RESULTS_DIR = os.path.join("results")

# Canonical model artifact selected by validation policy.
ACTIVE_MODEL_FILENAME = os.getenv("ACTIVE_MODEL_FILENAME", "model.keras")
ACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, ACTIVE_MODEL_FILENAME)
# Backward-compatible legacy model path.
COMPAT_FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "model.final.keras")

# Metadata outputs for reproducibility.
TRAINING_METADATA_PATH = os.path.join(RESULTS_DIR, "training_run_metadata.json")
EVALUATION_METADATA_PATH = os.path.join(RESULTS_DIR, "evaluation_run_metadata.json")

# Path to the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini model name used by the Evaluator Agent
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# -- Reproducibility Settings --
# Global random seed for Python/NumPy/TensorFlow training/evaluation paths.
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "7"))
# Enable deterministic TensorFlow ops where supported.
ENABLE_TF_DETERMINISM = os.getenv("ENABLE_TF_DETERMINISM", "1") == "1"


def resolve_model_path() -> str:
    """
    Return canonical model path when available, otherwise use legacy fallback.
    """
    return ACTIVE_MODEL_PATH if os.path.exists(ACTIVE_MODEL_PATH) else COMPAT_FINAL_MODEL_PATH

# -- Preprocessing Settings --

# Sampling frequency of the ECG signals
FS = 100

# Number of sample points in one minute
SAMPLE = FS * 60

# Time window settings for creating signal segments
# Minutes to include before the current minute
BEFORE = 2
# Minutes to include after the current minute
AFTER = 2

# Heart rate limits for filtering out physiologically impossible values
HR_MIN = 20
HR_MAX = 300

# Number of worker processes for parallel processing
# Uses one less than the number of available CPU cores, or 35 if more than 35 cores are available
NUM_WORKER = 35 if cpu_count() > 35 else cpu_count() - 1

# Interpolation rate for resampling the signal
IR = 3


# -- Model Training Settings --

# Model-specific parameters can be added here, for example:
# BATCH_SIZE = 128
# EPOCHS = 20
# LEARNING_RATE = 0.001
