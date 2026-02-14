"""
Utilities for computing HRV and EDR metrics from preprocessed ECG data.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import iqr


def compute_time_domain_hrv(rri: np.ndarray) -> dict:
    """
    Compute time-domain HRV metrics from RR intervals.
    
    Args:
        rri: RR intervals in seconds
        
    Returns:
        dict with metrics:
        - mean_rri: Mean RR interval (ms)
        - sdnn: Standard deviation of NN intervals (ms)
        - rmssd: Root mean square of successive differences (ms)
        - pnn50: Percentage of successive RR intervals differing by >50ms
        - cv: Coefficient of variation (SDNN/mean)
    """
    rri_ms = rri * 1000  # Convert to milliseconds
    
    # Basic stats
    mean_rri = np.mean(rri_ms)
    sdnn = np.std(rri_ms, ddof=1)
    
    # Successive differences
    diff_rri = np.diff(rri_ms)
    rmssd = np.sqrt(np.mean(diff_rri ** 2))
    
    # pNN50
    nn50_count = np.sum(np.abs(diff_rri) > 50)
    pnn50 = (nn50_count / len(diff_rri)) * 100 if len(diff_rri) > 0 else 0.0
    
    # Coefficient of variation
    cv = (sdnn / mean_rri) * 100 if mean_rri > 0 else 0.0
    
    return {
        "mean_rri_ms": float(mean_rri),
        "sdnn_ms": float(sdnn),
        "rmssd_ms": float(rmssd),
        "pnn50_percent": float(pnn50),
        "cv_percent": float(cv),
    }


def compute_frequency_domain_hrv(rri: np.ndarray, fs_resample: float = 4.0) -> dict:
    """
    Compute frequency-domain HRV metrics using Welch's method.
    
    Args:
        rri: RR intervals in seconds
        fs_resample: Resampling frequency for spectral analysis (Hz)
        
    Returns:
        dict with metrics:
        - vlf_power: Very low frequency power (0.003-0.04 Hz)
        - lf_power: Low frequency power (0.04-0.15 Hz)
        - hf_power: High frequency power (0.15-0.4 Hz)
        - lf_hf_ratio: LF/HF ratio
        - total_power: Total spectral power
    """
    # Resample RRI to uniform time grid
    rri_time = np.cumsum(rri)
    rri_time = np.insert(rri_time, 0, 0)  # Add t=0
    
    # Create uniform time grid
    time_uniform = np.arange(0, rri_time[-1], 1.0 / fs_resample)
    
    # Interpolate RRI onto uniform grid
    rri_uniform = np.interp(time_uniform, rri_time[1:], rri * 1000)
    
    # Detrend
    rri_detrend = sp_signal.detrend(rri_uniform)
    
    # Compute power spectral density using Welch's method
    freqs, psd = sp_signal.welch(
        rri_detrend,
        fs=fs_resample,
        nperseg=min(256, len(rri_detrend)),
        scaling='density'
    )
    
    # Define frequency bands
    vlf_band = (freqs >= 0.003) & (freqs < 0.04)
    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.4)
    
    # Compute band powers
    vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band])
    lf_power = np.trapz(psd[lf_band], freqs[lf_band])
    hf_power = np.trapz(psd[hf_band], freqs[hf_band])
    total_power = vlf_power + lf_power + hf_power
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
    
    return {
        "vlf_power_ms2": float(vlf_power),
        "lf_power_ms2": float(lf_power),
        "hf_power_ms2": float(hf_power),
        "total_power_ms2": float(total_power),
        "lf_hf_ratio": float(lf_hf_ratio),
    }


def compute_edr_metrics(amplitudes: np.ndarray, rri: np.ndarray, fs: float = 100) -> dict:
    """
    Compute ECG-Derived Respiration metrics from R-peak amplitudes.
    
    Uses amplitude variation method (simplest EDR technique).
    
    Args:
        amplitudes: R-peak amplitudes
        rri: RR intervals in seconds
        fs: Sampling frequency (Hz)
        
    Returns:
        dict with metrics:
        - resp_rate_bpm: Estimated respiratory rate (breaths/min)
        - edr_amplitude_range: Range of EDR signal
        - edr_variability: Standard deviation of EDR
    """
    # Defensive check: ensure amplitudes and rri have compatible lengths
    if len(amplitudes) != len(rri):
        min_len = min(len(amplitudes), len(rri))
        amplitudes = amplitudes[:min_len]
        rri = rri[:min_len]
    
    # Normalize amplitudes
    amp_norm = (amplitudes - np.mean(amplitudes)) / np.std(amplitudes)
    
    # Low-pass filter to extract respiratory component (0.1-0.5 Hz)
    nyquist = fs / 2.0
    low_cutoff = 0.1 / nyquist
    high_cutoff = 0.5 / nyquist
    
    # Design bandpass filter for respiratory frequencies
    b, a = sp_signal.butter(3, [low_cutoff, high_cutoff], btype='band')
    
    # Apply filter (zero-phase to avoid distortion)
    try:
        edr_signal = sp_signal.filtfilt(b, a, amp_norm)
    except Exception:
        # If filtering fails, use raw amplitudes
        edr_signal = amp_norm
    
    # Estimate respiratory rate from spectral peak
    # Resample to uniform time grid
    rri_time = np.cumsum(rri)
    time_uniform = np.arange(0, rri_time[-1], 1.0 / 4.0)  # 4 Hz resampling
    
    edr_uniform = np.interp(time_uniform, rri_time, edr_signal)
    
    # Compute PSD
    freqs, psd = sp_signal.welch(edr_uniform, fs=4.0, nperseg=min(256, len(edr_uniform)))
    
    # Find peak in respiratory band (0.1-0.5 Hz = 6-30 breaths/min)
    resp_band = (freqs >= 0.1) & (freqs <= 0.5)
    if np.any(resp_band):
        peak_idx = np.argmax(psd[resp_band])
        peak_freq = freqs[resp_band][peak_idx]
        resp_rate_bpm = peak_freq * 60
    else:
        resp_rate_bpm = 0.0
    
    # EDR variability
    edr_range = np.ptp(edr_signal)
    edr_std = np.std(edr_signal)
    
    return {
        "resp_rate_bpm": float(resp_rate_bpm),
        "edr_amplitude_range": float(edr_range),
        "edr_variability": float(edr_std),
    }


def compute_rpeak_stats(rpeaks: np.ndarray, signal_length: int, fs: float = 100) -> dict:
    """
    Compute R-peak detection statistics.
    
    Args:
        rpeaks: R-peak indices (samples)
        signal_length: Total signal length (samples)
        fs: Sampling frequency (Hz)
        
    Returns:
        dict with metrics:
        - num_rpeaks: Total number of R-peaks detected
        - mean_hr_bpm: Mean heart rate
        - hr_std_bpm: Standard deviation of heart rate
        - recording_duration_min: Duration of recording
    """
    duration_sec = signal_length / fs
    duration_min = duration_sec / 60.0
    
    rri = np.diff(rpeaks) / fs
    hr = 60.0 / rri if len(rri) > 0 else np.array([])
    
    return {
        "num_rpeaks": int(len(rpeaks)),
        "mean_hr_bpm": float(np.mean(hr)) if len(hr) > 0 else 0.0,
        "hr_std_bpm": float(np.std(hr)) if len(hr) > 0 else 0.0,
        "recording_duration_min": float(duration_min),
    }
