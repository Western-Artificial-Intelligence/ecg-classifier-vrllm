import os  # Provides utilities for working with file paths and directories
import pickle  # Used to load the preprocessed apnea-ecg.pkl (Python serialized object)
from typing import Dict, List, Tuple, Literal, Any  # Type hints for better readability and tooling support

import numpy as np  # Numerical computing library for arrays and math
import tensorflow as tf  # Deep learning library used to load and run the CNN-Transformer model

# A type alias indicating that the 'split' argument can only be "train" or "test"
SplitType = Literal["train", "test"]

class value_extractor:
    
    def __init__(
        self,
        base_dir: str = "dataset",        # Directory where the pickle file and related artifacts live
        pkl_name: str = "apnea-ecg.pkl",  # Filename of the preprocessed data produced by preprocessing.py
        model_path: str = "model.final.h5",  # Path to the trained CNN-Transformer model on disk
        ir: int = 3,                      # Interpolation rate used in training (samples per second)
        before: int = 2,                  # Minutes of context before the labeled minute (must match training)
        after: int = 2,                   # Minutes of context after the labeled minute (must match training)
    ) -> None:
        # Store the base directory path
        self.base_dir = base_dir
        # Build the full path to the pickle file (base_dir + pkl_name)
        self.pkl_path = os.path.join(base_dir, pkl_name)
        # Store the path to the trained model file
        self.model_path = model_path

        # Store interpolation-related settings; these must match the training script
        self.ir = ir
        self.before = before
        self.after = after

        # Dictionary to hold the loaded preprocessed apnea-ecg data (o_train, y_train, etc.)
        self.apnea_ecg: Dict[str, Any] = {}
        # Numpy array that will hold the interpolation time axis once initialized
        self.tm: np.ndarray | None = None
        # Placeholder for the loaded TensorFlow model; initially None until load_model() is called
        self.model: tf.keras.Model | None = None
        