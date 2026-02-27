# Paper Results Lock Plan

## Goal
Produce final, reproducible, leakage-free results for the paper with:
- Train/validation on `a/b/c` using `GroupKFold(k=5)` (grouped by record)
- Final untouched test on `x01-x35`
- One consistent model artifact and one consistent evaluation protocol

## Locked Experimental Decisions
1. Data protocol:
   - Development set: `a01-a20`, `b01-b05`, `c01-c10`
   - Final holdout test: `x01-x35`
2. Validation protocol:
   - `GroupKFold(n_splits=5)` over development set, grouped by record
3. Model selection protocol:
   - Choose the best model by validation performance (not the final epoch state)
   - Final reported test metrics must come from the validation-selected model policy
4. Positive class:
   - Apnea = class `1`
5. Final report metrics:
   - Accuracy, Sensitivity, Specificity, F1, AUC
6. Interpretability:
   - Grad-CAM and Agent outputs are qualitative support, not primary performance metrics

## Implementation Plan

### Phase 1: Remove Leakage and Lock Split Logic
1. Add grouped CV split utility for `a/b/c`.
   - Create/update split module to generate fold indices by record ID.
2. Ensure training never uses `x` for validation.
   - Update training pipeline so `validation_data` is fold-specific from `a/b/c` only.
3. Reserve `x01-x35` strictly for final evaluation.
   - Ensure no callback/model selection logic consumes `x`.

Acceptance checks:
- No code path uses `x_test` during `model.fit`.
- Fold logs show only `a/b/c` records in train/val.

### Phase 2: Standardize Model Selection and Artifact Usage
1. Define one active model path setting (for training/eval/API).
2. Use the same artifact consistently across:
   - CLI evaluation
   - API prediction
   - Grad-CAM generation
3. Document model selection rule:
   - Best validation model policy (explicitly not last-epoch/final-state model).

Acceptance checks:
- One config-controlled model path is referenced everywhere.
- No mixed `model.keras` vs `model.final.keras` behavior during final runs.

### Phase 3: Freeze Evaluation Semantics
1. Lock probability/classification rule used for metrics.
   - Keep consistent threshold/argmax definition.
2. Ensure confusion matrix and derived metrics use same label convention.
3. Export per-record summary on `x`:
   - apnea-minute ratio per record
   - highest and lowest predicted apnea prevalence records

Acceptance checks:
- Metric script and output CSV use one consistent decision rule.
- Highest/lowest record stats can be reproduced from saved outputs.

### Phase 4: Reproducibility Controls
1. Set and log random seeds (Python/NumPy/TensorFlow).
2. Log run metadata:
   - commit SHA
   - model path/name
   - fold ID
   - epoch stop point
3. Disable/clear stale caches before final experimental runs.

Acceptance checks:
- Re-running final script with same seed reproduces near-identical metrics.
- Cached artifacts do not alter final reported outputs.

### Phase 5: Final Paper Artifacts
1. Generate and save:
   - training curves
   - confusion matrix
   - ROC curve
   - final metrics table
2. Report model size:
   - parameter count
   - serialized model file size
3. Add qualitative section:
   - selected Grad-CAM examples
   - selected Agent report excerpts

Acceptance checks:
- All figures are regenerated from the final locked pipeline.
- Reported numbers match saved CSV/plots.

## Concrete Deliverables
1. Updated training/evaluation code implementing GroupKFold on `a/b/c`.
2. Final test-only evaluation script for `x01-x35`.
3. Results package:
   - metrics CSV
   - per-record summary CSV
   - confusion matrix/ROC/training-curve images
4. Methods section notes:
   - split policy
   - CV policy
   - final test policy
   - model selection rule

## Execution Order (Do Not Reorder)
1. Phase 1 (split/leakage)
2. Phase 2 (artifact consistency)
3. Phase 3 (metric semantics)
4. Phase 4 (reproducibility)
5. Phase 5 (final artifact generation)

## Out of Scope for Final Paper Lock
- Large UI refactors
- New agent feature work unrelated to model metrics
- Non-essential backend API redesign
