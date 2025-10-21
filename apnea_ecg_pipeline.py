# apnea_ecg_pipeline.py
# =============================================================================
# End-to-end, single-file, highly-commented Apnea-ECG baseline
#
# Big picture (why this approach):
# - The PhysioNet Apnea-ECG dataset labels each MINUTE as apnea (A) or normal (N).
# - Apnea affects breathing and autonomic tone → both influence the ECG:
#     • HRV (heart-rate variability) shifts with sympathetic/parasympathetic balance.
#     • EDR (ECG-Derived Respiration) captures breathing rhythm from ECG amplitude changes.
# - We therefore extract small, interpretable minute-level features (HRV + EDR power),
#   then train a tree model (LightGBM preferred; GradientBoosting fallback).
# - We split by record (night) to avoid leakage and calibrate probabilities so 0.5 ≈ “50% chance”.
#
# Artifacts produced:
#   ./outputs/features.csv          minute-level feature table + labels
#   ./outputs/model.joblib          trained sklearn pipeline (scaler+calibrator+model)
#   ./outputs/figs/*.png            quick EDA plots for teammates
# =============================================================================

# --- Standard library imports
from pathlib import Path               # portable filesystem paths (Windows/Linux/Mac)
import time                            # timestamps for pretty logging
import sys                             # exit cleanly on Ctrl+C
import warnings                        # suppress noisy warnings for a cleaner console
warnings.filterwarnings("ignore")      # we intentionally hide non-critical warnings

# --- Scientific Python stack
import numpy as np                     # fast numerical arrays and math
import pandas as pd                    # tables (DataFrame) for features/labels
from scipy import signal, interpolate  # signal processing (Welch PSD, detrend) + splines
import matplotlib.pyplot as plt        # quick plots for EDA/teammate onboarding

# --- PhysioNet tools (WFDB)
import wfdb                             # read WFDB records (.dat/.hea)
from wfdb import processing             # R-peak detection (XQRS) and corrections

# --- Scikit-learn (ML)
from sklearn.model_selection import GroupShuffleSplit   # split by record (group)
from sklearn.metrics import (                           # standard metrics
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV  # probability calibration
from sklearn.preprocessing import StandardScaler         # included for robust pipeline
from sklearn.pipeline import make_pipeline               # chain steps together
from sklearn.ensemble import GradientBoostingClassifier  # fallback model if LGBM missing
from joblib import dump                                  # save trained model pipeline

# =========================
# Configuration (edit here)
# =========================
DATA_DIR = Path("data")                 # where WFDB will download apnea-ecg
OUT_DIR  = Path("outputs")              # where we write features, model, figures
FIG_DIR  = OUT_DIR / "figs"             # subfolder for EDA plots
OUT_DIR.mkdir(parents=True, exist_ok=True)  # ensure outputs exist
FIG_DIR.mkdir(parents=True, exist_ok=True)  # ensure figs folder exists

RANDOM_STATE  = 42                      # seed for reproducibility (splits/models)
TEST_FRACTION = 0.20                    # 20% of records reserved for held-out test

MAX_RECORDS = None                      # set to an int (e.g., 10) for quick demo runs

# Physiologically meaningful frequency bands (Hz):
EDR_BAND   = (0.10, 0.40)               # ~6–24 breaths/min (typical adult sleep breathing)
RR_LF_BAND = (0.04, 0.15)               # LF HRV (mixed symp/parasymp oscillations)
RR_HF_BAND = (0.15, 0.40)               # HF HRV (respiratory sinus arrhythmia; vagal)

# =========================
# Utility helpers
# =========================
def log(msg: str) -> None:
    """Print a message with a HH:MM:SS timestamp so multi-person runs are readable."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def bandpower_psd(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """
    Compute power in a frequency band using Welch’s method.
    Why Welch? It’s a robust PSD estimator for short/noisy signals.
    - x:    1D time series
    - fs:   sampling rate (Hz)
    - fmin, fmax: frequency band of interest
    Returns: scalar band power (area under PSD between fmin and fmax).
    """
    if len(x) < 8:                                      # too few samples → unreliable
        return 0.0
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(len(x), 256))  # PSD estimate
    band = (f >= fmin) & (f <= fmax)                    # boolean mask for band
    return float(np.trapz(Pxx[band], f[band])) if band.any() else 0.0

def minute_windows(n_samples: int, fs: float):
    """
    Yield non-overlapping [start, end) sample indices for 60-second windows.
    We do this because labels in Apnea-ECG are per MINUTE.
    """
    step = int(60 * fs)                # number of samples in one minute
    # range stops at the last full minute; leftover tail is ignored (no label)
    for start in range(0, n_samples - step + 1, step):
        yield start, start + step

def rr_time_features(rr_ms: np.ndarray) -> dict:
    """
    Time-domain HRV features from RR intervals in milliseconds for ONE minute:
    - mean_rr: average beat interval (ms) → inverse of HR (slower HR → larger RR)
    - sdnn:    overall variability (ms)   → broad autonomic influences
    - rmssd:   short-term variability     → vagal (parasympathetic) activity proxy
    - pnn50:   fraction of |ΔRR| > 50 ms  → another vagal-linked marker
    If <2 beats in the minute, we can’t compute diffs → return NaNs (transparent fail).
    """
    if rr_ms.size < 2:                                       # need ≥ 2 beats for diffs
        return dict(mean_rr=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan)
    diffs = np.diff(rr_ms)                                   # successive differences
    return dict(
        mean_rr=float(np.mean(rr_ms)),                       # central tendency
        sdnn=float(np.std(rr_ms, ddof=1)),                   # unbiased std
        rmssd=float(np.sqrt(np.mean(diffs**2))),             # RMS of diffs
        pnn50=float(np.mean(np.abs(diffs) > 50.0)),          # proportion > 50 ms
    )

def rr_freq_features_from_context(r_locs: np.ndarray, fs: float) -> dict:
    """
    Frequency-domain HRV (LF/HF) computed on a multi-minute CONTEXT window.
    Rationale: estimating LF/HF needs several cycles; 60 s is tight.
    Steps:
      1) Convert R-peak indices (samples) → times (s) and RR intervals (s).
      2) Build tachogram (RR vs time) and resample it to a UNIFORM 4 Hz grid via splines.
      3) Detrend (remove slow drift) then compute power in LF and HF bands.
    Returns NaNs when unstable (too few beats or spline fails).
    """
    if len(r_locs) < 5:                                     # too few beats overall
        return dict(rr_lf=np.nan, rr_hf=np.nan, rr_lf_hf=np.nan)

    beat_t = r_locs / fs                                    # R-peak times (s)
    rr_s   = np.diff(beat_t)                                # RR intervals (s)
    if len(rr_s) < 3:                                       # need multiple intervals
        return dict(rr_lf=np.nan, rr_hf=np.nan, rr_lf_hf=np.nan)

    rr_t = beat_t[1:]                                       # timestamps for each RR
    fs_rr = 4.0                                             # resample tachogram at 4 Hz
    try:
        t_grid   = np.arange(rr_t[0], rr_t[-1], 1 / fs_rr)  # uniform time grid
        spline   = interpolate.CubicSpline(rr_t, rr_s)      # smooth interpolation
        rr_interp = signal.detrend(spline(t_grid))          # remove slow drift/baseline
        lf = bandpower_psd(rr_interp, fs_rr, *RR_LF_BAND)   # LF band power
        hf = bandpower_psd(rr_interp, fs_rr, *RR_HF_BAND)   # HF band power
        return dict(rr_lf=lf, rr_hf=hf, rr_lf_hf=(lf / hf if hf > 0 else np.nan))
    except Exception:                                       # e.g., spline errors
        return dict(rr_lf=np.nan, rr_hf=np.nan, rr_lf_hf=np.nan)

def edr_power_from_r_heights(r_locs: np.ndarray, r_heights: np.ndarray, fs: float) -> float:
    """
    EDR proxy: “ECG-Derived Respiration power”.
    Idea: inspiration/expiration subtly change chest impedance & electrode geometry,
          modulating R-wave HEIGHT over time at the breathing frequency.
    Implementation:
      - Take R-peak heights at their timestamps.
      - Interpolate heights to a uniform 1 Hz grid (enough for 0.10–0.40 Hz band).
      - Detrend and compute power in the respiratory band.
    """
    if len(r_locs) < 3:                                     # need a few peaks at least
        return np.nan
    t = r_locs / fs                                         # times of R-peaks (s)
    fs_edr = 1.0                                            # 1 sample/sec is sufficient
    try:
        t_grid = np.arange(t[0], t[-1], 1 / fs_edr)         # uniform time grid
        spline = interpolate.CubicSpline(t, r_heights)      # interpolate peak heights
        edr    = signal.detrend(spline(t_grid))             # remove slow drift
        return bandpower_psd(edr, fs_edr, *EDR_BAND)        # respiratory band power
    except Exception:
        return np.nan                                       # fail transparent

def get_model():
    """
    Return a tree-based classifier suitable for tabular features.
    Preference: LightGBM (handles class_weight, fast, strong).
    Fallback:   GradientBoosting (pure sklearn; fine for a baseline).
    """
    try:
        from lightgbm import LGBMClassifier                 # import here to keep dependency optional
        return LGBMClassifier(
            n_estimators=300,                               # # of boosting rounds
            learning_rate=0.05,                             # step size per round
            subsample=0.9,                                  # row subsampling (stochasticity)
            colsample_bytree=0.8,                           # feature subsampling
            class_weight="balanced",                        # mitigate class imbalance
            random_state=RANDOM_STATE,                      # reproducibility
        )
    except Exception:                                       # if LightGBM not installed
        return GradientBoostingClassifier(random_state=RANDOM_STATE)

# =========================
# Step 1 — Download dataset
# =========================
def download_dataset() -> None:
    """
    Use WFDB to download the 'apnea-ecg' database into ./data.
    We detect presence by .apn files (minute labels). If found → skip download.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)             # ensure data/ exists
    if list(DATA_DIR.glob("*.apn")):                        # any label files present?
        log("Found existing .apn labels — skipping download.")
        return
    log("Downloading PhysioNet 'apnea-ecg' into ./data …")  # friendly status
    wfdb.dl_database("apnea-ecg", str(DATA_DIR))            # pulls all records
    log("Download complete.")                               # done

# ============================
# Step 2 — Feature extraction
# ============================
def process_record(rec_name: str) -> pd.DataFrame:
    """
    Convert one record (e.g., 'a01') into minute-level feature rows.
    Output columns:
      mean_rr, sdnn, rmssd, pnn50, rr_lf, rr_hf, rr_lf_hf, edr_pwr_010_040, record, minute, y
    """
    rec_base = DATA_DIR / rec_name                          # base path (without extension)

    # --- Load ECG waveform and metadata
    rec = wfdb.rdrecord(str(rec_base))                      # read .dat/.hea pair
    fs  = rec.fs                                            # sampling frequency (Hz)
    # Some records have 2 leads; we consistently use the FIRST lead for simplicity.
    sig = rec.p_signal[:, 0] if rec.p_signal.ndim == 2 else rec.p_signal
    n_samples = len(sig)                                    # total number of samples

    # --- Detect R-peaks using XQRS (robust default detector in WFDB)
    xqrs = processing.XQRS(sig=sig, fs=fs)                  # initialize detector
    xqrs.detect()                                           # run detection
    r_locs = np.asarray(xqrs.qrs_inds, dtype=int)           # indices of detected peaks

    # --- Optional peak correction to local up-slope maxima (small alignment tweak)
    try:
        r_locs = processing.correct_peaks(
            sig, r_locs, search_radius=int(0.05 * fs), peak_dir="up"
        )
    except Exception:
        pass                                                # if correction fails, keep original

    r_heights = sig[r_locs]                                 # R-peak amplitudes (for EDR)

    # --- Load per-minute labels from .apn (lines like: "0 N", "1 A", ...)
    #     We map N→0 (normal) and A→1 (apnea).
    with open(rec_base.with_suffix(".apn"), "r") as f:
        labels = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0].isdigit():      # guard against blank/odd lines
                labels.append(1 if parts[1].upper() == "A" else 0)
    labels = np.array(labels, dtype=int)                    # convert to numpy array

    rows = []                                               # will hold one dict per minute
    minute_idx = 0                                          # label index increments per window
    for start, end in minute_windows(n_samples, fs):        # iterate over 60-second windows
        if minute_idx >= len(labels):                       # if dataset ended (partial last minute)
            break

        # --- R-peaks within THIS minute (for time-domain HRV)
        mask_min = (r_locs >= start) & (r_locs < end)       # boolean mask of peaks inside window
        r_min    = r_locs[mask_min]                         # R-peak indices for this minute
        # RR intervals for this minute in milliseconds (need ≥2 peaks)
        rr_ms    = (np.diff(r_min) / fs * 1000.0) if r_min.size >= 2 else np.array([])

        feats = {}                                          # feature dict for the row
        feats.update(rr_time_features(rr_ms))               # mean_rr, sdnn, rmssd, pnn50

        # --- Build a ~5-min CONTEXT around the minute for LF/HF and EDR
        ctx_start = max(0, start - int(120 * fs))           # include 2 minutes BEFORE
        ctx_end   = min(n_samples, end   + int(120 * fs))   # include 2 minutes AFTER
        mask_ctx  = (r_locs >= ctx_start) & (r_locs < ctx_end)

        feats.update(rr_freq_features_from_context(r_locs[mask_ctx], fs))  # rr_lf, rr_hf, rr_lf_hf
        feats["edr_pwr_010_040"] = edr_power_from_r_heights(               # respiratory band power
            r_locs[mask_ctx], r_heights[mask_ctx], fs
        )

        # --- Identifiers and label for this minute
        feats["record"] = rec_name                          # record id (e.g., 'a01')
        feats["minute"] = minute_idx                        # minute index within the record
        feats["y"]      = int(labels[minute_idx])           # 0 = normal, 1 = apnea

        rows.append(feats)                                  # add row to list
        minute_idx += 1                                     # move to next label/minute

    # Convert list of dicts to a DataFrame (one row per minute)
    return pd.DataFrame(rows)

def build_features() -> pd.DataFrame:
    """
    Iterate through all records (based on .apn files), extract features,
    concatenate, write to CSV, and return the combined DataFrame.
    """
    apn_files = sorted(DATA_DIR.glob("*.apn"))              # discover label files
    if not apn_files:                                       # sanity check
        raise SystemExit("No .apn files found. Did the download complete?")

    # Allow a quick-run mode for new teammates (subset of records)
    if MAX_RECORDS is not None:
        apn_files = apn_files[:MAX_RECORDS]

    all_dfs = []                                            # accumulate per-record DataFrames
    for apn in apn_files:
        rec_name = apn.stem                                 # strip ".apn" → 'a01'
        try:
            log(f"Processing {rec_name} …")
            df = process_record(rec_name)                   # extract features from this record
            if df.empty:                                    # warn if minute rows couldn’t be made
                log(f"[WARN] {rec_name}: produced 0 rows (few beats or label mismatch).")
            else:
                all_dfs.append(df)                          # keep non-empty results
        except Exception as e:
            # Robust baseline philosophy: skip problematic records but explain why.
            log(f"[WARN] Skipping {rec_name}: {e}")

    if not all_dfs:                                         # if everything failed, stop early
        raise SystemExit("No features extracted from any record. Check warnings above.")

    feats = pd.concat(all_dfs, ignore_index=True)           # stack all records vertically
    feats.to_csv(OUT_DIR / "features.csv", index=False)     # persist for transparency/reuse
    log(f"Saved features → {OUT_DIR/'features.csv'}  shape={feats.shape}")
    return feats

# ============================
# Step 3 — Quick EDA & plots
# ============================
def plot_label_balance(df: pd.DataFrame) -> None:
    """Bar chart of Normal vs Apnea minutes to see class imbalance at a glance."""
    fig, ax = plt.subplots(figsize=(6, 4))                  # create figure/axes
    counts = df["y"].value_counts().sort_index()            # counts for 0 then 1
    bars = ax.bar(["Normal (0)", "Apnea (1)"], counts.values)  # draw bars
    ax.set_title("Label Balance")                           # title
    ax.set_ylabel("Minutes")                                # y-axis label
    # annotate bar tops with exact counts for clarity
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()                                      # trim whitespace
    fig.savefig(FIG_DIR / "label_balance.png")              # write to disk
    plt.close(fig)                                          # free memory

def plot_apnea_rate_by_record(df: pd.DataFrame) -> None:
    """Bar chart of per-record apnea fraction (heterogeneity across nights/patients)."""
    rates = df.groupby("record")["y"].mean().sort_values(ascending=False)  # fraction of y==1 per record
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(rates.index, rates.values)                        # bar per record
    ax.set_title("Apnea Rate by Record")
    ax.set_ylabel("Fraction of minutes with apnea")
    ax.set_xticks(range(len(rates.index)))                   # tick per bar
    ax.set_xticklabels(rates.index, rotation=90, fontsize=8) # rotate to fit
    fig.tight_layout()
    fig.savefig(FIG_DIR / "apnea_rate_by_record.png")
    plt.close(fig)

def plot_example_ecg_with_rpeaks(example_record: str) -> None:
    """
    10-second ECG snippet with detected R-peaks (sanity check for detector quality).
    If you see many missed peaks or false ones, consider light filtering or a different detector.
    """
    try:
        rec = wfdb.rdrecord(str(DATA_DIR / example_record))  # load same record by id
        fs  = rec.fs
        sig = rec.p_signal[:, 0] if rec.p_signal.ndim == 2 else rec.p_signal
        n   = len(sig)

        dur   = int(10 * fs)                                 # 10 seconds → samples
        start = max(0, n // 2 - dur // 2)                    # center the window in the record
        end   = start + dur                                  # end index

        xqrs = processing.XQRS(sig=sig, fs=fs); xqrs.detect()
        r    = np.asarray(xqrs.qrs_inds, dtype=int)          # all peaks in record
        r10  = r[(r >= start) & (r < end)]                   # peaks inside 10-s window

        t = np.arange(start, end) / fs                       # time axis in seconds
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, sig[start:end], linewidth=1, label="ECG (lead 1)")  # ECG trace
        ax.plot(r10 / fs, sig[r10], "o", markersize=4, label="R-peaks")# mark peaks
        ax.set_title(f"Example ECG (10 s) with R-peaks — {example_record}")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("mV")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{example_record}_ecg_rpeaks.png")
        plt.close(fig)
    except Exception as e:
        log(f"[WARN] Could not plot example ECG for {example_record}: {e}")

def plot_feature_histograms(df: pd.DataFrame, feature_cols: list) -> None:
    """Histograms for each feature to develop intuition (skew, tails, bimodality)."""
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(7, 2.2 * len(feature_cols)))
    if len(feature_cols) == 1:                               # normalize axes to list
        axes = [axes]
    for ax, col in zip(axes, feature_cols):
        vals = df[col].dropna().values                       # ignore NaNs for plotting
        ax.hist(vals, bins=40)                               # draw histogram
        ax.set_title(col)                                    # title = feature name
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_histograms.png")
    plt.close(fig)

def plot_correlation(df: pd.DataFrame, feature_cols: list) -> None:
    """Pearson correlation heatmap (trees tolerate collinearity; this is for intuition)."""
    sub = df[feature_cols].dropna()                          # remove rows with missing values
    if sub.shape[0] < 2:                                     # need at least 2 rows
        return
    C = np.corrcoef(sub.values.T)                            # correlation matrix (features x features)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(C, vmin=-1, vmax=1)                       # heatmap in [-1,1]
    ax.set_xticks(range(len(feature_cols))); ax.set_xticklabels(feature_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feature_cols))); ax.set_yticklabels(feature_cols, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")   # add color bar legend
    ax.set_title("Feature Correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_correlation.png")
    plt.close(fig)

# ============================
# Step 4 — Train & evaluate
# ============================
def train_and_evaluate(df: pd.DataFrame) -> None:
    """
    Train a calibrated tree model with a RECORD-WISE split:
    - GroupShuffleSplit ensures all minutes from a record stay together.
    - CalibratedClassifierCV (isotonic) makes predicted probabilities reliable.
    - We report AUROC/AUPRC, confusion matrix, and a classification report.
    """
    # Explicit feature list (kept small & interpretable on purpose)
    feature_cols = [
        "mean_rr", "sdnn", "rmssd", "pnn50",                 # HRV (time)
        "rr_lf", "rr_hf", "rr_lf_hf",                        # HRV (freq)
        "edr_pwr_010_040",                                   # EDR respiratory band power
    ]

    df = df.dropna(subset=feature_cols)                     # simple, honest baseline (no imputation)
    X  = df[feature_cols].values                             # feature matrix (N x F)
    y  = df["y"].values                                      # labels (0/1)
    groups = df["record"].values                             # group id = record name

    # Group-wise split: ~80% of records for training, 20% for testing
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_tr, X_te = X[train_idx], X[test_idx]                   # split features
    y_tr, y_te = y[train_idx], y[test_idx]                   # split labels

    base = get_model()                                       # LightGBM or GBDT
    # Pipeline:
    #  - StandardScaler(with_mean=False): inert for trees, but keeps pipeline robust
    #    if you later swap in a linear model; we keep it for consistency.
    #  - CalibratedClassifierCV: wrap base model to calibrate probabilities (isotonic).
    model = make_pipeline(
        StandardScaler(with_mean=False),
        CalibratedClassifierCV(base, method="isotonic", cv=3),
    )

    log("Training model with isotonic calibration …")
    model.fit(X_tr, y_tr)                                    # fit base model + calibrator

    p_te  = model.predict_proba(X_te)[:, 1]                  # calibrated P(apnea)
    y_hat = (p_te >= 0.5).astype(int)                        # default threshold at 0.5

    auroc = roc_auc_score(y_te, p_te)                        # threshold-free ranking
    auprc = average_precision_score(y_te, p_te)              # robust under imbalance
    cm    = confusion_matrix(y_te, y_hat)                    # [[TN, FP],[FN, TP]]
    rep   = classification_report(y_te, y_hat, digits=3)     # precision/recall/F1 by class

    log(f"[RESULT] AUROC={auroc:.3f}  AUPRC={auprc:.3f}")    # print scalar metrics
    log("[CONFUSION]\n" + str(cm))                           # print confusion matrix
    log("[REPORT]\n" + rep)                                  # print detailed report

    dump(model, OUT_DIR / "model.joblib")                    # persist trained pipeline
    log(f"Saved model → {OUT_DIR/'model.joblib'}")           # path confirmation

# ============================
# Main orchestration function
# ============================
def main() -> None:
    """
    Orchestrate the full pipeline:
      1) Download data (idempotent).
      2) Build features (save CSV).
      3) EDA visuals (save PNGs).
      4) Train & evaluate model (save joblib).
    """
    np.random.seed(RANDOM_STATE)                             # make NumPy ops reproducible

    download_dataset()                                       # pull apnea-ecg if needed
    feats = build_features()                                 # compute minute-level features

    log("Running quick EDA and saving figures …")            # status
    plot_label_balance(feats)                                # class balance bar plot
    plot_apnea_rate_by_record(feats)                         # per-record apnea fraction
    example_record = feats["record"].value_counts().index[0] # pick a common record for ECG demo
    plot_example_ecg_with_rpeaks(example_record)             # 10 s ECG + R-peaks plot

    feat_cols = [                                            # list features once to avoid typos
        "mean_rr","sdnn","rmssd","pnn50",
        "rr_lf","rr_hf","rr_lf_hf","edr_pwr_010_040"
    ]
    plot_feature_histograms(feats, feat_cols)                # univariate distributions
    plot_correlation(feats, feat_cols)                       # feature correlation heatmap
    log(f"Figures saved in: {FIG_DIR.resolve()}")            # show absolute path

    train_and_evaluate(feats)                                # fit & evaluate model

    log("All done. Artifacts are in ./outputs")          # friendly finish line

# Entry point guard so this can be imported without running
if __name__ == "__main__":
    try:
        main()                                               # run the pipeline
    except KeyboardInterrupt:                                # allow Ctrl+C to stop gracefully
        log("Interrupted by user.")                          # inform user
        sys.exit(1)                                          # exit with non-zero code
