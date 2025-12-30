"""
This module is responsible for loading the preprocessed ECG data.
It performs further preparation steps like interpolation and normalization
to ensure the data is in the correct format and scale for the model.

Functions:
    - load_data(): Loads data from the apnea-ecg.pkl file and prepares
                   training and testing datasets.
"""

import os
import pickle

# Scientific computing libraries
import numpy as np
from scipy.interpolate import splev, splrep

# Import project-specific configuration
from src import config

# Define a scaler function for min-max normalization.
# This scales an array's values to a range between 0 and 1.
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def load_data() -> tuple:
    """
    Loads the processed ECG data from a pickle file, interpolates and normalizes
    the RRI and amplitude signals, and prepares them into NumPy arrays
    suitable for a Keras model.

    The data is loaded from the file specified by `config.PROCESSED_DATA_DIR`.
    Signals are interpolated to a fixed time grid defined by `config.BEFORE`,
    `config.AFTER`, and `config.IR`.

    Returns:
        tuple: A tuple containing the prepared datasets:
            - x_train (np.ndarray): Training features.
            - y_train (np.ndarray): Training labels.
            - groups_train (list): Original record names for training data segments.
            - x_test (np.ndarray): Testing features.
            - y_test (np.ndarray): Testing labels.
            - groups_test (list): Original record names for testing data segments.
    """
    # Create a uniform time array (time_mesh) for curve interpolation.
    # The length of this array is determined by the total segment duration
    # (config.BEFORE + 1 + config.AFTER minutes) and the interpolation rate (config.IR).
    tm = np.arange(0, (config.BEFORE + 1 + config.AFTER) * 60, step=1 / float(config.IR))

    # Construct the full path to the preprocessed data file.
    # config.PROCESSED_DATA_DIR specifies the directory where apnea-ecg.pkl is stored.
    apnea_ecg_path = os.path.join(config.PROCESSED_DATA_DIR, "apnea-ecg.pkl")
    
    # Load the preprocessed data, which contains RRI and amplitude signals
    # along with their timestamps and corresponding labels.
    with open(apnea_ecg_path, 'rb') as f:
        apnea_ecg = pickle.load(f)

    # --- Prepare Training Data ---
    x_train = []
    # Extract original signals (o_train), labels (y_train), and group identifiers (groups_train).
    o_train, y_train, groups_train = apnea_ecg["o_train"], apnea_ecg["y_train"], apnea_ecg["groups_train"]
    
    # Iterate through each signal segment in the training set.
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
        
        # Perform cubic spline interpolation on RRI and amplitude signals.
        # This resamples the signals onto the uniform time mesh 'tm'.
        # Signals are scaled (normalized) before interpolation using the 'scaler' lambda.
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])

    # Convert the list of features into a NumPy array.
    # The transpose operation (0, 2, 1) reshapes the array from
    # (num_samples, num_features, sequence_length) to (num_samples, sequence_length, num_features),
    # which is the expected input shape for the Keras model (batch_size, timesteps, features).
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))
    y_train = np.array(y_train, dtype="float32")

    # --- Prepare Testing Data ---
    x_test = []
    # Extract original signals (o_test), labels (y_test), and group identifiers (groups_test).
    o_test, y_test, groups_test = apnea_ecg["o_test"], apnea_ecg["y_test"], apnea_ecg["groups_test"]
    
    # Iterate through each signal segment in the testing set, similar to training data.
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
        
        # Interpolate and normalize RRI and amplitude signals for testing data.
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])

    # Convert test features into a NumPy array with the correct shape for the model.
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return x_train, y_train, groups_train, x_test, y_test, groups_test
