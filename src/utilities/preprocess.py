"""
This utility module provides a function to preprocess a single ECG record
into a format suitable for model inference. It mirrors the preprocessing
steps used for the full dataset but is designed for on-demand use.

Functions:
    - _normalize(): Helper function for min-max normalization.
    - preprocess(): Main function to preprocess a single ECG record.
"""

import os

# Scientific computing and data analysis libraries
import numpy as np
import wfdb
import biosppy.signals.tools as st
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
from scipy.interpolate import splev, splrep

# Import project-specific configuration
from src import config


def _normalize(arr: np.ndarray) -> np.ndarray:
    """
    Performs min-max normalization on a NumPy array.
    Scales the values of the array to a range between 0 and 1.

    Args:
        arr (np.ndarray): The input NumPy array to normalize.

    Returns:
        np.ndarray: The normalized NumPy array. Returns an array of zeros
                    if the input array has zero range (max - min < 1e-8).
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr # Return empty array if input is empty
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-8: # Avoid division by zero for constant arrays
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def preprocess(record_path_or_name: str) -> dict:
    """
    Preprocesses a single PhysioNet Apnea-ECG record into model-ready tensors.

    This function applies the same signal processing and feature extraction steps
    as `src/preprocessing.py` but for an individual record. It is typically
    used for making predictions on new, single ECG files.

    Args:
        record_path_or_name (str): The path or base name of the ECG record to preprocess.
                                   Examples: 'a01', 'ecgdata/a01', 'data/raw/ecgdata/a01'.

    Returns:
        dict: A dictionary containing the processed data and metadata:
            - 'record' (str): The base name of the processed record (e.g., 'a01').
            - 'tensors' (np.ndarray): The 3D NumPy array of preprocessed features
                                      (RRI and amplitude) ready for model input.
                                      Shape: (num_segments, sequence_length, num_features).
            - 'minutes' (list): A list of central minute indices for which valid segments
                                were extracted.
            - 'skipped' (list): A list of minute indices that were skipped due to
                                insufficient data, noise, or abnormal heart rates.
    """
    record_path_or_name = str(record_path_or_name)
    
    # Determine the base directory for raw ECG files if not explicitly provided in the path.
    # It defaults to config.RAW_DATA_DIR if the path only contains the record name.
    base_dir = os.path.dirname(record_path_or_name) or config.RAW_DATA_DIR
    # Extract the base name of the record (e.g., 'a01' from 'data/raw/ecgdata/a01.dat').
    base_record_name = os.path.splitext(os.path.basename(record_path_or_name))[0]

    # Load the single-channel ECG signal for the specified record.
    # wfdb.rdrecord reads header and signal files.
    rec = wfdb.rdrecord(os.path.join(base_dir, base_record_name), channels=[0])
    signals = rec.p_signal[:, 0] # Extract the first channel's signal data

    # Attempt to load minute-level apnea labels from the .apn annotation file.
    # If the .apn file is missing (e.g., for new, unannotated data),
    # dummy 'N' (Normal) labels are created for all minutes.
    try:
        ann = wfdb.rdann(os.path.join(base_dir, base_record_name), extension="apn")
        labels = ann.symbol # e.g., ['N', 'A', 'N', ...]
    except Exception:
        total_minutes = int(len(signals) / float(config.SAMPLE))
        labels = ["N"] * total_minutes # Create dummy labels

    X = [] # List to store extracted RRI and amplitude features for each segment
    minutes = [] # List to store central minute index of each valid segment
    skipped = [] # List to store central minute index of each skipped segment

    # Iterate through each minute of the signal based on the labels.
    for j in range(len(labels)):
        # Check if the current minute 'j' has enough 'BEFORE' and 'AFTER' context
        # required to form a full segment based on configuration.
        if j < config.BEFORE or \
           (j + 1 + config.AFTER) > len(signals) / float(config.SAMPLE):
            skipped.append(j) # Mark this minute as skipped
            continue

        # Define the start and end sample points for the current signal segment.
        # This segment covers `BEFORE` minutes, the current minute, and `AFTER` minutes.
        start_sample = int((j - config.BEFORE) * config.SAMPLE)
        end_sample = int((j + 1 + config.AFTER) * config.SAMPLE)
        signal_segment = signals[start_sample:end_sample]

        # Apply a bandpass filter to the extracted signal segment.
        # This removes low-frequency baseline wander and high-frequency noise,
        # preparing the signal for R-peak detection. Configuration from src.config.
        signal_filt, _, _ = st.filter_signal(
            signal_segment,
            ftype="FIR",
            band="bandpass",
            order=int(0.3 * config.FS),
            frequency=[3, 45],
            sampling_rate=config.FS,
        )

        # Detect R-peaks using the Hamilton segmenter and then correct their locations.
        rpeaks, = hamilton_segmenter(signal_filt, sampling_rate=config.FS)
        rpeaks, = correct_rpeaks(signal_filt, rpeaks=rpeaks, sampling_rate=config.FS, tol=0.1)

        if len(rpeaks) == 0:
            skipped.append(j) # Skip if no R-peaks could be detected
            continue

        # Filter out signal segments based on physiologically plausible R-peak counts.
        # Segments with too few or too many beats per window are likely noisy or anomalous.
        expected_segment_duration_minutes = (1 + config.AFTER + config.BEFORE)
        beats_per_window = len(rpeaks) / expected_segment_duration_minutes
        if beats_per_window < 40 or beats_per_window > 200:
            skipped.append(j)
            continue

        # Extract R-R Interval (RRI) features.
        rri_tm = rpeaks[1:] / float(config.FS) # Time points for RRI values
        rri_signal = np.diff(rpeaks) / float(config.FS) # RRI values (duration between consecutive R-peaks)
        if rri_signal.size == 0:
            skipped.append(j) # Skip if no RRI could be calculated (e.g., only one R-peak)
            continue
        # Apply a median filter to smooth out RRI signal and reduce artifacts.
        rri_signal = medfilt(rri_signal, kernel_size=3)

        # Extract R-peak Amplitude features.
        ampl_tm = rpeaks / float(config.FS) # Time points for R-peak amplitudes
        # Ensure R-peak indices are within the bounds of the filtered signal.
        rpeaks_clip = np.clip(rpeaks, 0, len(signal_filt) - 1)
        ampl_signal = signal_filt[rpeaks_clip] # Amplitude of the signal at R-peak locations

        # Calculate Heart Rate (HR) and filter based on physiological limits.
        # Avoid division by zero by clipping RRI values.
        hr = 60.0 / np.clip(rri_signal, 1e-6, None)
        if not np.all(np.logical_and(hr >= config.HR_MIN, hr <= config.HR_MAX)):
            skipped.append(j)
            continue

        # If the segment passes all checks, add its features and minute index.
        X.append(((rri_tm, rri_signal), (ampl_tm, ampl_signal)))
        minutes.append(j)

    # If no valid segments were extracted, return empty tensors and metadata.
    if not X:
        # Calculate expected sequence length based on config for an empty tensor.
        seq_len = int((config.BEFORE + 1 + config.AFTER) * 60 * config.IR)
        tensors = np.empty((0, seq_len, 2), dtype=np.float32)
        return {
            "record": base_record_name,
            "tensors": tensors,
            "minutes": [],
            "skipped": skipped,
        }

    # Interpolate extracted RRI and amplitude signals to a fixed time grid.
    # This ensures all segments have a consistent length for model input.
    tm_fixed_grid = np.arange(0, (config.BEFORE + 1 + config.AFTER) * 60, step=1.0 / config.IR)

    x_list = []
    for (rri_tm, rri_signal), (ampl_tm, ampl_signal) in X:
        # Normalize and interpolate both RRI and amplitude signals.
        rri_interp = splev(tm_fixed_grid, splrep(rri_tm, _normalize(rri_signal), k=3), ext=1)
        ampl_interp = splev(tm_fixed_grid, splrep(ampl_tm, _normalize(ampl_signal), k=3), ext=1)
        x_list.append([rri_interp, ampl_interp])

    # Convert the list of processed features into a NumPy array.
    # Transpose to get the shape (num_segments, sequence_length, num_features)
    # as expected by the Keras model.
    x_arr = np.array(x_list, dtype="float32").transpose((0, 2, 1))

    return {
        "record": base_record_name,
        "tensors": x_arr,
        "minutes": minutes,
        "skipped": skipped,
    }
