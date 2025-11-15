import os
import numpy as np
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
import biosppy.signals.tools as st
from scipy.interpolate import splev, splrep

# Configuration (mirrors preprocessing.py and CNN-transformer-LTSM.py)
FS = 100
SAMPLE = FS * 60  # samples per minute
BEFORE = 2        # minutes before
AFTER = 2         # minutes after
HR_MIN = 20
HR_MAX = 300
IR = 3            # interpolation rate (samples per second)


def _normalize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def preprocess(record):
    """
    Preprocess a single Apnea-ECG record into model-ready tensors.

    Parameters
    ----------
    record : str
        Can be any of:
          - 'a01'
          - 'ecgdata/a01'
          - 'ecgdata/a01.dat' / '.hea' / '.apn'

    Returns
    -------
    dict with keys:
        'record'  : record name (e.g. 'a01')
        'tensors' : np.ndarray shape (N, T, 2) where T = 900
        'minutes' : list of central minute indices kept
        'skipped' : list of minute indices skipped
    """
    record = str(record)
    # If no directory is given, assume 'ecgdata'
    base_dir = os.path.dirname(record) or "ecgdata"
    base = os.path.splitext(os.path.basename(record))[0]

    # Load single-channel ECG signal
    rec = wfdb.rdrecord(os.path.join(base_dir, base), channels=[0])
    signals = rec.p_signal[:, 0]

    # Try to load minute-level labels from .apn; if missing, create dummy labels
    try:
        ann = wfdb.rdann(os.path.join(base_dir, base), extension="apn")
        labels = ann.symbol
    except Exception:
        total_minutes = int(len(signals) / float(SAMPLE))
        labels = ["N"] * total_minutes

    X = []
    minutes = []
    skipped = []

    for j in range(len(labels)):
        # Need full context window available
        if j < BEFORE or (j + 1 + AFTER) > len(signals) / float(SAMPLE):
            skipped.append(j)
            continue

        start = int((j - BEFORE) * SAMPLE)
        end = int((j + 1 + AFTER) * SAMPLE)
        signal = signals[start:end]

        # Band-pass filter (same as preprocessing.py)
        signal_filt, _, _ = st.filter_signal(
            signal,
            ftype="FIR",
            band="bandpass",
            order=int(0.3 * FS),
            frequency=[3, 45],
            sampling_rate=FS,
        )

        # Find R peaks
        rpeaks, = hamilton_segmenter(signal_filt, sampling_rate=FS)
        rpeaks, = correct_rpeaks(signal_filt, rpeaks=rpeaks, sampling_rate=FS, tol=0.1)

        if len(rpeaks) == 0:
            skipped.append(j)
            continue

        # Remove abnormal R-peak counts per window (as in preprocessing.py)
        beats_per_window = len(rpeaks) / float(1 + AFTER + BEFORE)
        if beats_per_window < 40 or beats_per_window > 200:
            skipped.append(j)
            continue

        # Extract RRI and amplitude features
        rri_tm = rpeaks[1:] / float(FS)
        rri_signal = np.diff(rpeaks) / float(FS)
        if rri_signal.size == 0:
            skipped.append(j)
            continue

        rri_signal = medfilt(rri_signal, kernel_size=3)

        ampl_tm = rpeaks / float(FS)
        # Clip indices just in case
        rpeaks_clip = np.clip(rpeaks, 0, len(signal_filt) - 1)
        ampl_signal = signal_filt[rpeaks_clip]

        hr = 60.0 / np.clip(rri_signal, 1e-6, None)
        if not np.all(np.logical_and(hr >= HR_MIN, hr <= HR_MAX)):
            skipped.append(j)
            continue

        X.append(((rri_tm, rri_signal), (ampl_tm, ampl_signal)))
        minutes.append(j)

    # If nothing kept, return empty tensors
    if not X:
        seq_len = int((BEFORE + 1 + AFTER) * 60 * IR)
        tensors = np.empty((0, seq_len, 2), dtype=np.float32)
        return {
            "record": base,
            "tensors": tensors,
            "minutes": [],
            "skipped": skipped,
        }

    # Interpolate to fixed grid (same as in CNN-transformer-LTSM.load_data)
    tm = np.arange(0, (BEFORE + 1 + AFTER) * 60, step=1.0 / IR)

    x_list = []
    for (rri_tm, rri_signal), (ampl_tm, ampl_signal) in X:
        rri_interp = splev(tm, splrep(rri_tm, _normalize(rri_signal), k=3), ext=1)
        ampl_interp = splev(tm, splrep(ampl_tm, _normalize(ampl_signal), k=3), ext=1)
        x_list.append([rri_interp, ampl_interp])

    x_arr = np.array(x_list, dtype="float32").transpose((0, 2, 1))  # (N, seq_len, 2)

    return {
        "record": base,
        "tensors": x_arr,
        "minutes": minutes,
        "skipped": skipped,
    }

