
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Apnea-ECG Baseline (simple & well-documented)
===================================================
Purpose
-------
Get you from zero → baseline on PhysioNet **Apnea-ECG** fast, with minimal moving parts
and clear comments so you can tweak confidently.

What it does
------------
- Auto-downloads the dataset into --dl_dir if no .hea files are found.
- Builds 60s windows aligned to minute-level labels (A=1, N=0) from .apn.
- Features (minimal on purpose):
    * HRV (time-domain): hr_mean, sdnn, rmssd, pnn50
      - Prefer **QRS annotations** (.qrs) for R-peaks
      - Fallback to a very simple peak detector if .qrs is missing
    * Morph/Energy: signal std, log-energy, peak count, peak prominence stats
- Trains a small **XGBoost** classifier (subject-wise split).
- Picks an operating threshold aiming for **~85% specificity** on TRAIN.
- Prints AUROC/AUPRC and writes:
    * outputs/apnea_features.csv
    * outputs/confusion_matrix.png
    * outputs/calibration.png

Usage
-----
python quick_apnea_baseline_simple.py --dl_dir data/apnea-ecg --subset a01 a02 a03 --max_records 3
# or let it auto-pick the first N available records
python quick_apnea_baseline_simple.py --dl_dir data/apnea-ecg --max_records 10

Dependencies
------------
pip install wfdb numpy scipy pandas scikit-learn xgboost matplotlib tqdm
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import wfdb
from scipy.signal import butter, filtfilt, find_peaks, hilbert, resample
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# ----------------------------
# Small logging helpers
# ----------------------------
def log(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


# ----------------------------
# Dataset discovery / download
# ----------------------------
def discover_records(dl_dir: str) -> List[str]:
    """Return stems of .hea files at this folder level (e.g., ['a01','a01r','a01er'])."""
    p = Path(dl_dir)
    return sorted({f.stem for f in p.glob("*.hea")})

def ensure_dataset(dl_dir: str) -> None:
    """If no .hea files are found, download the PhysioNet 'apnea-ecg' database here."""
    if discover_records(dl_dir):
        return
    log(f"No .hea found under '{dl_dir}'. Downloading PhysioNet 'apnea-ecg'...")
    Path(dl_dir).mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("apnea-ecg", dl_dir=dl_dir)
    log("Download complete.")


# ----------------------------
# Basic signal utilities
# ----------------------------
def bandpass(sig, fs, lo=5.0, hi=15.0, order=3):
    ny = 0.5 * fs
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, sig)

def detect_rpeaks_simple(sig, fs):
    """Very simple R-peak proxy (only for fallback; QRS annotations are preferred)."""
    bp = bandpass(sig, fs, 5, 15, 3)
    min_rr = 0.25  # 240 bpm upper bound
    peaks, _ = find_peaks(bp, distance=int(min_rr*fs), prominence=np.std(bp)*0.5 if np.std(bp)>0 else 0.01)
    return peaks


# ----------------------------
# Features (minimal & clear)
# ----------------------------
def hrv_features_from_peaks(peaks_samples, fs) -> Dict[str, float]:
    """Time-domain HRV from peak sample indices (seconds)."""
    feats = {"hr_mean": np.nan, "sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan}
    if peaks_samples is None or len(peaks_samples) < 2:
        return feats
    rr = np.diff(peaks_samples) / fs  # RR in seconds
    if rr.size == 0:
        return feats
    hr = 60.0 / rr
    feats["hr_mean"] = float(np.nanmean(hr))
    feats["sdnn"]    = float(np.nanstd(rr) * 1000.0)  # ms
    drr = np.diff(rr)
    if drr.size:
        feats["rmssd"] = float(np.sqrt(np.nanmean(drr**2)) * 1000.0)  # ms
        feats["pnn50"] = float(100.0 * np.mean(np.abs(drr) > 0.05))
    return feats

def morph_energy_features(sig):
    """Simple morphology/energy proxies. Pass in a z-normalized window for stability."""
    feats = {
        "sig_std": float(np.std(sig)),
        "log_energy": float(np.log(np.sum(sig**2) + 1e-8)),
    }
    peaks, props = find_peaks(sig, distance=max(1, len(sig)//600), prominence=np.std(sig)*0.5 if np.std(sig)>0 else 0.01)
    prom = props.get('prominences', np.array([]))
    feats["peak_count"] = int(len(peaks))
    feats["peak_prom_mean"] = float(np.mean(prom)) if prom.size else 0.0
    feats["peak_prom_std"]  = float(np.std(prom)) if prom.size else 0.0
    return feats


# ----------------------------
# Record loading & windowing
# ----------------------------
def load_record_signal_and_labels(dl_dir, rec_id):
    """
    Reads:
      - Signal via rdsamp (first lead)
      - Labels via 'apn' annotation (A/N per minute)
      - QRS peaks via 'qrs' annotation (if present)
    Returns (fs, signal, labels, qrs_samples)
    """
    rec_path = os.path.join(dl_dir, rec_id)

    # ECG signal
    sig, fields = wfdb.rdsamp(rec_path)
    fs = float(fields['fs'])
    sig = sig[:,0] if sig.ndim > 1 else sig

    # Apnea annotations (minute labels)
    try:
        ann_apn = wfdb.rdann(rec_path, 'apn')
    except Exception:
        # Windows path quirk fallback
        ann_apn = wfdb.rdann(rec_id, 'apn', pn_dir=dl_dir.replace("\\","/"))
    labels = np.array([1 if s=='A' else 0 for s in ann_apn.symbol], dtype=int)

    # QRS peaks (optional)
    try:
        ann_qrs = wfdb.rdann(rec_path, 'qrs')
        qrs_samples = np.asarray(ann_qrs.sample, dtype=int)
    except Exception:
        qrs_samples = np.array([], dtype=int)

    return fs, sig, labels, qrs_samples

def minute_windows(sig, fs, n_minutes):
    """Return list of 60s windows aligned to labels; pads/shortens last window if needed."""
    win = int(60*fs)
    windows = []
    for i in range(n_minutes):
        s = i*win; e = (i+1)*win
        if e <= len(sig):
            windows.append(sig[s:e])
        else:
            # tail: keep if >=30s data; otherwise drop
            if len(sig) - s >= win//2:
                w = np.zeros(win, dtype=sig.dtype)
                w[:len(sig)-s] = sig[s:]
                windows.append(w)
            else:
                break
    return windows


# ----------------------------
# Metrics & plotting
# ----------------------------
def eval_at_fixed_specificity(y_true, y_prob, spec_target=0.85):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    idx = int(np.argmin(np.abs(spec - spec_target)))
    return float(thr[idx]), float(tpr[idx]), float(spec[idx])

def plot_confusion(cm, out_path):
    plt.figure()
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Test)")
    plt.colorbar(im)
    plt.xticks([0,1], ["Normal","Apnea"])
    plt.yticks([0,1], ["Normal","Apnea"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_calibration(y_true, y_prob, out_path):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration (Test)")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Quick, simple Apnea-ECG baseline.")
    ap.add_argument("--dl_dir", type=str, required=True, help="Folder with Apnea-ECG files (will auto-download if empty).")
    ap.add_argument("--subset", nargs="*", default=None, help="Optional list of records to use (e.g., a01 a01r a01er).")
    ap.add_argument("--max_records", type=int, default=10, help="Cap number of records to use.")
    args = ap.parse_args()

    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Ensure data is present (auto-download if needed)
    ensure_dataset(args.dl_dir)

    # 1) Pick records
    recs_avail = discover_records(args.dl_dir)
    log(f"Available (first 20): {recs_avail[:20]}{' ...' if len(recs_avail)>20 else ''}")
    if args.subset:
        recs = [r for r in args.subset if r in recs_avail][:args.max_records]
        if not recs:
            raise SystemExit("[ERROR] None of the --subset names exist at --dl_dir.")
    else:
        recs = recs_avail[:args.max_records]
    if not recs:
        raise SystemExit("[ERROR] No records found. Check --dl_dir.")

    # Subject-wise split: last 2 for test (simple & explicit)
    train_ids, test_ids = (recs[:-2], recs[-2:]) if len(recs) > 2 else (recs[:-1], recs[-1:])
    log(f"Train IDs: {train_ids}")
    log(f"Test  IDs: {test_ids}")

    # 2) Feature extraction
    rows = []
    for rec_id in tqdm(recs, desc="Featureizing"):
        try:
            fs, sig, labels, qrs_samples = load_record_signal_and_labels(args.dl_dir, rec_id)
        except Exception as e:
            warn(f"Skipping {rec_id}: {e}"); continue

        wins = minute_windows(sig, fs, len(labels))
        if not wins:
            warn(f"{rec_id}: no windows built."); continue

        for i, (w, y) in enumerate(zip(wins, labels[:len(wins)])):
            start = i*int(60*fs); end = start + len(w)
            # Prefer QRS-based HRV if available; else fallback to simple detector
            if qrs_samples.size >= 2:
                mask  = (qrs_samples >= start) & (qrs_samples < end)
                peaks = qrs_samples[mask]
            else:
                peaks = detect_rpeaks_simple(w, fs)

            feats = {}
            feats.update(hrv_features_from_peaks(peaks, fs))
            # Morph/Energy on z-normalized window
            w_z = (w - np.mean(w)) / (np.std(w) + 1e-8)
            feats.update(morph_energy_features(w_z))

            feats.update({"record": rec_id, "minute_idx": i, "y": int(y)})
            rows.append(feats)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[ERROR] No feature rows created. Check annotations and folder level.")

    # Save features CSV
    tmp_path = out_dir / "apnea_features.tmp.csv"
    final_path = out_dir / "apnea_features.csv"
    df.to_csv(tmp_path, index=False); os.replace(tmp_path, final_path)
    log(f"Saved features to {final_path} (shape: {df.shape})")

    # 3) Train/test split by record
    is_train = df["record"].isin(train_ids)
    train_df = df[is_train].dropna()
    test_df  = df[~is_train].dropna()

    X_train = train_df.drop(columns=["record","minute_idx","y"]).values
    y_train = train_df["y"].values
    X_test  = test_df.drop(columns=["record","minute_idx","y"]).values
    y_test  = test_df["y"].values

    # 4) XGBoost baseline (tiny, fast)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=4,
        random_state=42
    )
    clf.fit(X_train, y_train)
    p_train = clf.predict_proba(X_train)[:,1]
    p_test  = clf.predict_proba(X_test)[:,1]

    # 5) Metrics & threshold (train-chosen)
    auroc = roc_auc_score(y_test, p_test)
    auprc = average_precision_score(y_test, p_test)
    log(f"[RESULT] AUROC={auroc:.3f}, AUPRC={auprc:.3f} (test)")

    thr, se_tr, sp_tr = eval_at_fixed_specificity(y_train, p_train, spec_target=0.85)
    y_hat = (p_test >= thr).astype(int)
    cm = confusion_matrix(y_test, y_hat)
    log(f"[RESULT] Train OP ~85% Sp: thr={thr:.3f}, Train SE={se_tr:.3f}, SP={sp_tr:.3f}")
    log(f"[RESULT] Confusion (test):\n{cm}")

    # 6) Plots
    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_calibration(y_test, p_test, out_dir / "calibration.png")
    log("[INFO] Saved confusion_matrix.png and calibration.png in outputs/.")

if __name__ == "__main__":
    main()
