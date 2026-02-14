"""
Configuration file for the ECG Apnea project.

This file contains all the configuration variables for the project,
such as file paths, model parameters, and preprocessing settings.
"""

import os
from multiprocessing import cpu_count

# -- File Paths --
# Using os.path.join for platform compatibility
# Note: These paths are relative to the project root directory

# Path to the raw ECG data
RAW_DATA_DIR = os.path.join("data", "raw", "ecgdata")

# Path to the processed data
PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Path to the trained models
MODELS_DIR = os.path.join("models")

# Path to the results
RESULTS_DIR = os.path.join("results")


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
