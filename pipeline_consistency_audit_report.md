# Pipeline Consistency Audit Report

Date: 2026-02-19

## Scope
Audit of training, evaluation, single-record prediction, API, and frontend display paths to identify mismatches between ground truth files, CLI outputs, and UI behavior.

## Findings

### 1) Test-set leakage in model selection/evaluation (Critical)
- `backend/src/train.py:119` uses `x_test, y_test` as `validation_data` during training.
- `backend/src/train.py:77` selects checkpoint by `val_loss` on that same set.
- `backend/src/evaluate.py:95` evaluates on the same `x_test, y_test`.
- Effect: inflated/optimistic test metrics because the test split is used for model selection.

### 2) Frontend stale prediction cache can mask model changes (High)
- `frontend/src/utils/storage.ts:38` stores predictions keyed only by `filename`.
- `frontend/src/components/PatientAnalysis.tsx:319` and `frontend/src/components/PatientAnalysis.tsx:767` prefer cached predictions and may skip API refresh.
- Effect: UI may show old predictions after retraining or changing model file (`model.keras` vs `model.final.keras`).

### 3) API and CLI can diverge due to preprocessing cache path differences (High)
- API prediction path uses cached preprocessing: `backend/api/main.py:125` (`preprocess_with_cache`).
- Cache key is record name only: `backend/src/utilities/preprocess.py:413`, cache load at `backend/src/utilities/preprocess.py:417`.
- CLI `predict_on_record` uses non-cached preprocessing: `backend/src/evaluate.py:201` (`preprocess`).
- Effect: API and CLI can disagree if cached tensors are stale after config/preprocessing/raw-data changes.

### 4) Grad-CAM "probability" is confidence of predicted class, not always P(apnea) (Medium)
- Class chosen by argmax: `backend/api/main.py:209`.
- Saved/returned confidence value is `prediction_prob[predicted_class]`: `backend/api/main.py:211`, `backend/api/main.py:257`, `backend/api/main.py:268`.
- Effect: when predicted class is Non-Apnea, the exposed probability is P(non-apnea), which can be misread as P(apnea).

### 5) Training preprocessing and inference preprocessing are not identical (Medium)
- Training worker amplitude indexing uses un-clipped R-peaks: `backend/src/preprocessing.py:108`.
- Inference preprocessing clips R-peaks and clips denominator for HR: `backend/src/utilities/preprocess.py:164`, `backend/src/utilities/preprocess.py:260`.
- Effect: subtle train/infer feature mismatch risk in edge cases.

### 6) Model-file usage differs across paths (Medium)
- API and record prediction load `model.final.keras`: `backend/api/main.py:34`, `backend/src/evaluate.py:187`.
- Evaluation path can be edited to `model.keras` at `backend/src/evaluate.py:85`.
- Effect: comparisons across CLI eval/API/UI can accidentally mix model artifacts.

## Solutions

### 1) Fix test-set leakage (Critical)
- Quick fix:
  - Keep `x01..x35` for final evaluation only.
  - Split training set (`a/b/c`) into train and validation before `model.fit`.
- Robust fix:
  - Add subject-level split (GroupShuffleSplit/GroupKFold) using record IDs to avoid minute-level leakage across the same subject.
  - Use validation split for checkpoint/early stopping and never for final reporting.
- Verification:
  - Confirm `train.py` no longer passes `x_test, y_test` to `validation_data`.
  - Confirm final `evaluate_model()` still uses untouched `x_test, y_test` once.

### 2) Fix stale frontend cache (High)
- Quick fix:
  - Add a "Refresh predictions" action that bypasses IndexedDB and always calls `/api/predict`.
- Robust fix:
  - Include a model/version key in cache key (for example: `filename + model_hash + preprocess_version`).
  - Expire cached predictions by TTL.
- Verification:
  - Retrain model, reload UI, verify predictions update without manual cache clearing.

### 3) Fix API/CLI preprocess divergence (High)
- Quick fix:
  - Make CLI prediction use `preprocess_with_cache(..., force_recompute=True)` or API use forced recompute for debugging mode.
- Robust fix:
  - Version the preprocess cache with config signature (`FS`, `BEFORE`, `AFTER`, `IR`, code version).
  - Invalidate cache automatically when signature changes.
- Verification:
  - Compare API and CLI prediction outputs for same record and model; minute/probabilities should match.

### 4) Clarify probability semantics (Medium)
- Quick fix:
  - Rename fields in Grad-CAM responses: `class_confidence` and `predicted_class`.
- Robust fix:
  - Also return `p_apnea` and `p_non_apnea` everywhere (`/api/predict`, `/api/gradcam`, CSV exports).
  - Keep thresholding based on explicit `p_apnea`.
- Verification:
  - Ensure UI labels and backend response docs consistently refer to the same probability.

### 5) Align training and inference preprocessing (Medium)
- Quick fix:
  - Apply the same R-peak clipping and HR safety behavior in both preprocessing paths.
- Robust fix:
  - Consolidate to one shared feature-extraction function used by both `preprocessing.py` and `utilities/preprocess.py`.
- Verification:
  - Unit-test identical input segment through both paths and assert same extracted features.

### 6) Unify model artifact usage (Medium)
- Quick fix:
  - Introduce one config value (for example `ACTIVE_MODEL_PATH`) and load it everywhere.
- Robust fix:
  - Save model metadata (`name`, `epoch`, `val_loss`, timestamp) and expose in API/UI.
  - Add explicit mode switch (`best` vs `final`) in CLI and API.
- Verification:
  - Print active model path/hash at startup in CLI and API; confirm they match during comparison runs.

## Clarifications
- Apnea classification is not "only at 100% confidence".
- Frontend apnea filtering uses `>= 0.5`: `frontend/src/components/PatientAnalysis.tsx:953`.
- Other class decisions use argmax (equivalent two-class boundary near 0.5): `backend/api/main.py:209`.

## Recommended Remediation Order
1. Remove test leakage: create train/validation split from training records only; keep `x01..x35` as untouched test.
2. Unify model source across all paths (single configurable model path).
3. Add cache invalidation/versioning for preprocessing and frontend prediction cache.
4. Standardize probability semantics (explicit `p_apnea` vs `class_confidence`).
5. Align preprocessing implementations to avoid train/infer drift.

## Ground-Truth Sources (for reference)
- Training records (`a/b/c`): per-record `.apn` annotations (e.g., `data/raw/ecgdata/a01.apn`).
- Test records (`x`): `data/processed/event-2-answers` parsed in `backend/src/preprocessing.py:173-199`.
