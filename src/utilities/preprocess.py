"""
This utility module provides a function to preprocess a single ECG record
into a format suitable for model inference. It mirrors the preprocessing
steps used for the full dataset but is designed for on-demand use.

Functions:
    - _normalize(): Helper function for min-max normalization.
    - preprocess(): Main function to preprocess a single ECG record.
    - save_preprocessed_cache(): Save preprocessing results to disk.
    - load_preprocessed_cache(): Load preprocessing results from disk.
    - preprocess_with_cache(): Wrapper that uses caching for better performance.
"""

import os
import time

# Scientific computing and data analysis libraries
import numpy as np
import wfdb
import biosppy.signals.tools as st
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
from scipy.interpolate import splev, splrep

# Import project-specific configuration
from src import config

# Import HRV and EDR computation utilities
from src.utilities.hrv_edr import (
    compute_time_domain_hrv,
    compute_frequency_domain_hrv,
    compute_edr_metrics,
    compute_rpeak_stats,
)


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
            "raw_segments": [],
            "minutes": [],
            "skipped": skipped,
        }

    # Interpolate extracted RRI and amplitude signals to a fixed time grid.
    # This ensures all segments have a consistent length for model input.
    tm_fixed_grid = np.arange(0, (config.BEFORE + 1 + config.AFTER) * 60, step=1.0 / config.IR)

    x_list = []
    raw_segments_list = [] # Store raw signal segments
    
    # We iterate again through the original processing loop logic but now we need 
    # to match the filtered X list. A cleaner way is to store the raw segment in X originally.
    # Let's refactor the loop slightly to store raw_segment in X.
    
    # Re-initialize lists
    X_features = [] 
    X_raw = []
    minutes = []
    skipped = []
    
    # Lists to collect data for stats computation
    all_rpeaks = []
    all_rri = []
    all_amplitudes = []

    for j in range(len(labels)):
        if j < config.BEFORE or \
           (j + 1 + config.AFTER) > len(signals) / float(config.SAMPLE):
            skipped.append(j)
            continue

        start_sample = int((j - config.BEFORE) * config.SAMPLE)
        end_sample = int((j + 1 + config.AFTER) * config.SAMPLE)
        signal_segment = signals[start_sample:end_sample]

        # Filter
        signal_filt, _, _ = st.filter_signal(
            signal_segment,
            ftype="FIR",
            band="bandpass",
            order=int(0.3 * config.FS),
            frequency=[3, 45],
            sampling_rate=config.FS,
        )

        # R-peaks
        rpeaks, = hamilton_segmenter(signal_filt, sampling_rate=config.FS)
        rpeaks, = correct_rpeaks(signal_filt, rpeaks=rpeaks, sampling_rate=config.FS, tol=0.1)

        if len(rpeaks) == 0:
            skipped.append(j)
            continue

        beats_per_window = len(rpeaks) / ((1 + config.AFTER + config.BEFORE))
        if beats_per_window < 40 or beats_per_window > 200:
            skipped.append(j)
            continue

        # Features
        rri_tm = rpeaks[1:] / float(config.FS)
        rri_signal = np.diff(rpeaks) / float(config.FS)
        
        if rri_signal.size == 0:
            skipped.append(j)
            continue
            
        rri_signal = medfilt(rri_signal, kernel_size=3)

        ampl_tm = rpeaks / float(config.FS)
        rpeaks_clip = np.clip(rpeaks, 0, len(signal_filt) - 1)
        ampl_signal = signal_filt[rpeaks_clip]

        hr = 60.0 / np.clip(rri_signal, 1e-6, None)
        if not np.all(np.logical_and(hr >= config.HR_MIN, hr <= config.HR_MAX)):
            skipped.append(j)
            continue

        X_features.append(((rri_tm, rri_signal), (ampl_tm, ampl_signal)))
        X_raw.append(signal_segment) # Store the raw segment
        minutes.append(j)
        
        # Collect data for stats computation
        # Note: rri_signal has length (num_rpeaks - 1), ampl_signal has length num_rpeaks
        # We align them by taking amplitudes at positions 1: to match the RRI intervals
        all_rpeaks.extend(rpeaks.tolist())
        all_rri.extend(rri_signal.tolist())
        all_amplitudes.extend(ampl_signal[1:].tolist())  # Skip first amplitude to align with RRI

    if not X_features:
        seq_len = int((config.BEFORE + 1 + config.AFTER) * 60 * config.IR)
        tensors = np.empty((0, seq_len, 2), dtype=np.float32)
        return {
            "record": base_record_name,
            "tensors": tensors,
            "raw_segments": [],
            "minutes": [],
            "skipped": skipped,
        }

    x_list = []
    for (rri_tm, rri_signal), (ampl_tm, ampl_signal) in X_features:
        rri_interp = splev(tm_fixed_grid, splrep(rri_tm, _normalize(rri_signal), k=3), ext=1)
        ampl_interp = splev(tm_fixed_grid, splrep(ampl_tm, _normalize(ampl_signal), k=3), ext=1)
        x_list.append([rri_interp, ampl_interp])

    x_arr = np.array(x_list, dtype="float32").transpose((0, 2, 1))

    # Compute physiological stats if we have sufficient data
    stats = {}
    if len(all_rri) > 10 and len(all_amplitudes) > 10:  # Minimum data requirement
        try:
            stats = {
                "hrv_time": compute_time_domain_hrv(np.array(all_rri)),
                "hrv_freq": compute_frequency_domain_hrv(np.array(all_rri)),
                "edr": compute_edr_metrics(
                    np.array(all_amplitudes), 
                    np.array(all_rri), 
                    fs=config.FS
                ),
                "rpeak": compute_rpeak_stats(
                    np.array(all_rpeaks), 
                    len(signals), 
                    fs=config.FS
                ),
            }
        except Exception as e:
            print(f"Warning: Could not compute stats for {base_record_name}: {e}")
            stats = {}

    return {
        "record": base_record_name,
        "tensors": x_arr,
        "raw_segments": X_raw, # Return list of raw numpy arrays
        "minutes": minutes,
        "skipped": skipped,
        "stats": stats,  # NEW - physiological statistics
    }


def save_preprocessed_cache(record_name: str, preprocessed_data: dict) -> None:
    """
    Save preprocessed ECG data to disk cache.
    
    Args:
        record_name: Base name of the record (e.g., 'a01')
        preprocessed_data: Dictionary returned by preprocess() function
    """
    cache_dir = os.path.join(config.PROCESSED_DATA_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{record_name}_preprocessed.npz")
    
    # Prepare data for saving
    # Convert raw_segments list to a single array for easier storage
    raw_segments_array = np.array(preprocessed_data["raw_segments"], dtype=object)
    
    # Save with compression
    np.savez_compressed(
        cache_file,
        record=record_name,
        tensors=preprocessed_data["tensors"],
        raw_segments=raw_segments_array,
        minutes=np.array(preprocessed_data["minutes"]),
        skipped=np.array(preprocessed_data["skipped"]),
        stats=preprocessed_data.get("stats", {}),
        timestamp=time.time()
    )
    
    print(f"Cached preprocessing results for {record_name} at {cache_file}")


def load_preprocessed_cache(record_name: str) -> dict:
    """
    Load preprocessed ECG data from disk cache.
    
    Args:
        record_name: Base name of the record (e.g., 'a01')
        
    Returns:
        Dictionary with preprocessed data, or None if cache doesn't exist
    """
    cache_dir = os.path.join(config.PROCESSED_DATA_DIR, "cache")
    cache_file = os.path.join(cache_dir, f"{record_name}_preprocessed.npz")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        data = np.load(cache_file, allow_pickle=True)
        
        # Reconstruct the dictionary
        result = {
            "record": str(data["record"]),
            "tensors": data["tensors"],
            "raw_segments": list(data["raw_segments"]),
            "minutes": list(data["minutes"]),
            "skipped": list(data["skipped"]),
            "stats": data["stats"].item() if "stats" in data else {},
        }
        
        print(f"Loaded cached preprocessing results for {record_name}")
        return result
        
    except Exception as e:
        print(f"Error loading cache for {record_name}: {e}")
        return None


def preprocess_with_cache(record_path_or_name: str, force_recompute: bool = False) -> dict:
    """
    Preprocess a single ECG record with disk caching for better performance.
    
    This function checks if preprocessing results are cached. If they exist and are
    valid, it loads them from disk. Otherwise, it runs preprocessing and caches the results.
    
    Args:
        record_path_or_name: The path or base name of the ECG record to preprocess.
        force_recompute: If True, bypass cache and recompute preprocessing.
        
    Returns:
        dict: Same as preprocess() function - contains tensors, raw_segments, minutes, etc.
    """
    record_path_or_name = str(record_path_or_name)
    
    # Extract the base name of the record
    base_record_name = os.path.splitext(os.path.basename(record_path_or_name))[0]
    
    # Try to load from cache first (unless forced to recompute)
    if not force_recompute:
        cached_data = load_preprocessed_cache(base_record_name)
        if cached_data is not None:
            return cached_data
    
    # Cache miss or forced recompute - run preprocessing
    print(f"Running preprocessing for {base_record_name}...")
    preprocessed_data = preprocess(record_path_or_name)
    
    # Save to cache for future use
    save_preprocessed_cache(base_record_name, preprocessed_data)
    
    return preprocessed_data
