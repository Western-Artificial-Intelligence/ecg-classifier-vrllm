
"""
This script handles the preprocessing of raw ECG data from the PhysioNet Apnea-ECG dataset.
It extracts features like R-R intervals (RRI) and R-peak amplitudes from ECG signals,
filters out noisy or physiologically implausible data, and prepares a consolidated
dataset (apnea-ecg.pkl) for model training and testing.

The preprocessing pipeline involves:
1. Loading raw ECG signals and annotations.
2. Filtering signals to remove noise.
3. Detecting R-peaks.
4. Extracting RRI and amplitude features.
5. Filtering out abnormal segments.
6. Saving the processed data into a pickle file.
"""

import pickle
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Scientific computing and signal processing libraries
import numpy as np
import wfdb
import biosppy.signals.tools as st
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
from tqdm import tqdm # Progress bar for loops

# Import project-specific configuration
from src import config

def worker(name: str, labels: list) -> tuple:
    """
    Processes a single ECG record, extracts features, and filters data.

    This function is designed to be run in parallel for multiple records.
    It performs signal filtering, R-peak detection, RRI and amplitude extraction,
    and then applies physiological plausibility checks.

    Args:
        name (str): The name of the ECG record (e.g., 'a01').
        labels (list): A list of apnea labels for each minute of the record.

    Returns:
        tuple: A tuple containing:
            - X (list): List of extracted features (RRI and amplitude signals).
            - y (list): List of corresponding binary labels (0.0 for normal, 1.0 for apnea).
            - groups (list): List of record names, used for grouping data.
            
    Raises:
        Exception: Catches and re-raises any exception that occurs during processing
                   of a single record, providing context about the failed file.
    """
    try:
        X = []
        y = []
        groups = []
        
        # Load the single-channel ECG signal from the raw data directory
        # config.RAW_DATA_DIR specifies the base path for raw ECG files.
        signals = wfdb.rdrecord(os.path.join(config.RAW_DATA_DIR, name), channels=[0]).p_signal[:, 0]
        
        # Iterate through each minute of the signal based on provided labels
        # tqdm is used to display a progress bar in the console.
        for j in tqdm(range(len(labels)), desc=name, file=sys.stdout):
            # Ensure enough 'before' and 'after' context is available for the current minute 'j'
            if j < config.BEFORE or \
               (j + 1 + config.AFTER) > len(signals) / float(config.SAMPLE):
                continue # Skip if the segment is too short

            # Extract a segment of the signal covering 'before' minutes, the current minute, and 'after' minutes
            start_sample = int((j - config.BEFORE) * config.SAMPLE)
            end_sample = int((j + 1 + config.AFTER) * config.SAMPLE)
            signal_segment = signals[start_sample:end_sample]
            
            # Apply a bandpass filter to the signal segment to remove noise
            # Filter parameters (type, band, order, frequencies, sampling rate) are from src.config
            signal_filtered, _, _ = st.filter_signal(signal_segment, ftype='FIR', band='bandpass',
                                                     order=int(0.3 * config.FS), frequency=[3, 45],
                                                     sampling_rate=config.FS)
            
            # Find R-peaks (QRS complex) using the Hamilton segmenter
            rpeaks, = hamilton_segmenter(signal_filtered, sampling_rate=config.FS)
            # Correct R-peak locations based on signal morphology
            rpeaks, = correct_rpeaks(signal_filtered, rpeaks=rpeaks, sampling_rate=config.FS, tol=0.1)
            
            # Filter out segments with an abnormal number of R-peaks
            # This helps in removing noisy or poor-quality signal segments
            expected_segment_duration_minutes = (1 + config.AFTER + config.BEFORE)
            beats_per_minute = len(rpeaks) / expected_segment_duration_minutes
            if beats_per_minute < 40 or beats_per_minute > 200:
                continue # Skip abnormal segments
            
            # Extract R-R Interval (RRI) and R-peak Amplitude features
            # RRI is the time difference between consecutive R-peaks.
            rri_tm = rpeaks[1:] / float(config.FS) # Time instances of RRI
            rri_signal = np.diff(rpeaks) / float(config.FS) # RRI values
            if rri_signal.size == 0:
                continue # Skip if no RRI could be calculated

            # Apply a median filter to RRI to smooth out outliers
            rri_signal = medfilt(rri_signal, kernel_size=3)
            
            # R-peak amplitude: amplitude of the signal at each R-peak
            ampl_tm = rpeaks / float(config.FS) # Time instances of R-peak amplitudes
            ampl_siganl = signal_filtered[rpeaks] # Amplitude values
            
            # Calculate Heart Rate (HR) from RRI
            hr = 60 / rri_signal
            
            # Remove segments with physiologically impossible heart rates
            if np.all(np.logical_and(hr >= config.HR_MIN, hr <= config.HR_MAX)):
                # If segment passes all checks, save extracted features and label
                X.append([(rri_tm, rri_signal), (ampl_tm, ampl_siganl)])
                # Convert 'N' (Normal) to 0.0 and others (Apnea) to 1.0
                y.append(0. if labels[j] == 'N' else 1.)
                groups.append(name) # Keep track of the original record name
        
        return X, y, groups
    except Exception as e:
        # Print an error message if processing of a specific file fails
        print(f"Error processing file {name}: {e}", file=sys.stderr)
        raise # Re-raise the exception to indicate failure in the worker process

def run_preprocessing():
    """
    Orchestrates the entire data preprocessing pipeline.

    This function defines the lists of records for training and testing,
    parallelizes the feature extraction using a process pool, and saves
    the resulting processed data into a pickle file for later use by the model.
    """
    apnea_ecg = {} # Dictionary to store all processed data

    # Define the list of ECG record names to be used for the training set.
    # These correspond to the 'a', 'b', and 'c' series in the PhysioNet Apnea-ECG dataset.
    train_record_names = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "b01", "b02", "b03", "b04", "b05",
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    ]

    o_train = [] # Stores extracted RRI and amplitude signals for training
    y_train = [] # Stores labels for training
    groups_train = [] # Stores original record names for training data points
    
    print('Preprocessing training data...')
    # Use ProcessPoolExecutor for parallel processing of ECG records.
    # The number of worker processes is configured in src.config.
    with ProcessPoolExecutor(max_workers=config.NUM_WORKER) as executor:
        task_list = []
        # Submit a 'worker' task for each training record
        for record_name in train_record_names:
            # Load apnea annotations (.apn file) for the current record
            labels = wfdb.rdann(os.path.join(config.RAW_DATA_DIR, record_name), extension="apn").symbol
            task_list.append(executor.submit(worker, record_name, labels)) # Add task to the pool

        # Collect results as tasks complete
        for task in as_completed(task_list):
            X, y, groups = task.result()
            o_train.extend(X)
            y_train.extend(y)
            groups_train.extend(groups)

    print() # Newline for better output formatting

    # Load "event-2-answers" which contains labels for the 'x' series test records.
    # This file is expected to be in the PROCESSED_DATA_DIR.
    answers = {}
    event_2_answers_path = os.path.join(config.PROCESSED_DATA_DIR, "event-2-answers")
    with open(event_2_answers_path, "r") as f:
        for answer in f.read().split("\n\n"):
            # Parse the answer file to create a dictionary of record names to labels
            answers[answer[:3]] = list("".join(answer.split()[2::2]))

    # Define the list of ECG record names to be used for the testing set.
    # These correspond to the 'x' series in the PhysioNet Apnea-ECG dataset.
    test_record_names = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]

    o_test = [] # Stores extracted RRI and amplitude signals for testing
    y_test = [] # Stores labels for testing
    groups_test = [] # Stores original record names for testing data points
    
    print("Preprocessing testing data...")
    # Use ProcessPoolExecutor for parallel processing of test ECG records.
    with ProcessPoolExecutor(max_workers=config.NUM_WORKER) as executor:
        task_list = []
        # Submit a 'worker' task for each testing record
        for record_name in test_record_names:
            # Get labels from the 'answers' dictionary loaded previously
            labels = answers[record_name]
            task_list.append(executor.submit(worker, record_name, labels))

        # Collect results as tasks complete
        for task in as_completed(task_list):
            X, y, groups = task.result()
            o_test.extend(X)
            y_test.extend(y)
            groups_test.extend(groups)

    # Consolidate all processed data into a single dictionary
    apnea_ecg = dict(o_train=o_train, y_train=y_train, groups_train=groups_train,
                     o_test=o_test, y_test=y_test, groups_test=groups_test)
    
    # Save the consolidated data to a pickle file in the processed data directory.
    # This file (apnea-ecg.pkl) will be loaded by src/data_loader.py for model training.
    apnea_ecg_path = os.path.join(config.PROCESSED_DATA_DIR, "apnea-ecg.pkl")
    with open(apnea_ecg_path, "wb") as f:
        pickle.dump(apnea_ecg, f, protocol=2)

    print("\nPreprocessing complete!")

if __name__ == "__main__":
    # If this script is run directly, execute the preprocessing pipeline.
    run_preprocessing()
