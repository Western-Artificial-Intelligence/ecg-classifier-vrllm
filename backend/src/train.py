"""
This module is responsible for training the CNN-Transformer model
defined in `src.model` using the preprocessed data loaded via `src.data_loader`.
It orchestrates the training loop, applies callbacks for monitoring and
optimization, and saves the trained model.

Functions:
    - train_model(): Loads data, builds and compiles the model,
                     runs the training process, and saves the final model.
"""

import glob
import json
import os
import random
import shutil
import subprocess
from datetime import datetime, timezone

# Deep learning framework and libraries
import tensorflow as tf
import keras
import numpy as np

# Import project-specific modules
from backend.src import config # For accessing configuration parameters (e.g., file paths, model params)
from backend.src.data_loader import load_data # For loading prepared training and testing data
from backend.src.model import create_model # For instantiating the CNN-Transformer model architecture
from backend.src.utilities.splits import iter_group_kfold_indices, summarize_fold_groups


def _get_git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _configure_reproducibility() -> None:
    os.environ["PYTHONHASHSEED"] = str(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    if config.ENABLE_TF_DETERMINISM:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:
            print(f"Warning: Could not enable TF op determinism: {exc}")


def _clear_stale_fold_artifacts() -> None:
    stale_patterns = [
        os.path.join(config.MODELS_DIR, "model.fold*.keras"),
        os.path.join(config.RESULTS_DIR, "log.fold*.csv"),
    ]
    for pattern in stale_patterns:
        for stale_file in glob.glob(pattern):
            try:
                os.remove(stale_file)
                print(f"Removed stale artifact: {stale_file}")
            except OSError as exc:
                print(f"Warning: Failed to remove stale artifact {stale_file}: {exc}")


def train_model() -> keras.callbacks.History:
    """
    Orchestrates the model training process.

    This function performs the following steps:
    1. Loads development (`a/b/c`) and test (`x`) data via `load_data`.
    2. Uses GroupKFold (k=5) on development groups (record IDs) to build
       train/validation folds without record leakage.
    3. Trains one model per fold and tracks best fold by minimum validation loss.
    4. Saves the validation-selected best model as both `model.keras` and
       `model.final.keras` for downstream compatibility.

    Returns:
        keras.callbacks.History: History object from the selected best fold.
    """
    print("--- Starting Model Training ---")
    _configure_reproducibility()
    print(
        "Reproducibility config: "
        f"seed={config.RANDOM_SEED}, tf_determinism={config.ENABLE_TF_DETERMINISM}"
    )

    # 1. Load data. We train/validate only on development cohort (`a/b/c`).
    # `x_test/y_test` remain untouched for final evaluation in evaluate_model().
    x_train, y_train, groups_train, _, _, _ = load_data()

    # 2. Convert labels to categorical format (one-hot encoding).
    y_train = keras.utils.to_categorical(y_train, num_classes=2)

    print(f"Loaded development data shapes: features={x_train.shape}, labels={y_train.shape}")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    _clear_stale_fold_artifacts()

    best_fold_idx = None
    best_fold_epoch = None
    best_fold_val_loss = float("inf")
    best_fold_model_path = None
    best_history = None
    fold_summaries = []

    # 3. GroupKFold on development cohort only (no `x` usage here).
    for fold_idx, (train_idx, val_idx) in enumerate(
        iter_group_kfold_indices(groups_train, n_splits=5),
        start=1,
    ):
        train_groups, val_groups = summarize_fold_groups(groups_train, train_idx, val_idx)
        print(f"\n--- Fold {fold_idx}/5 ---")
        print(f"Train groups ({len(train_groups)}): {train_groups}")
        print(f"Val groups ({len(val_groups)}): {val_groups}")

        model = create_model(input_shape=x_train.shape[1:])
        if fold_idx == 1:
            print("\nModel Summary:")
            model.summary()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        fold_model_path = os.path.join(config.MODELS_DIR, f"model.fold{fold_idx}.keras")
        fold_log_path = os.path.join(config.RESULTS_DIR, f"log.fold{fold_idx}.csv")

        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=fold_model_path,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=3,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                fold_log_path,
                separator=",",
                append=False,
            ),
        ]

        history = model.fit(
            x_train[train_idx],
            y_train[train_idx],
            batch_size=128,
            epochs=20,  # config.EPOCHS could be added later if desired
            validation_data=(x_train[val_idx], y_train[val_idx]),
            callbacks=callbacks_list,
        )

        fold_best_val_loss = min(history.history["val_loss"])
        fold_best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        print(f"Fold {fold_idx} best val_loss: {fold_best_val_loss:.6f} at epoch {fold_best_epoch}")

        fold_summaries.append({
            "fold": fold_idx,
            "train_group_count": len(train_groups),
            "val_group_count": len(val_groups),
            "train_groups": train_groups,
            "val_groups": val_groups,
            "best_val_loss": float(fold_best_val_loss),
            "best_epoch": fold_best_epoch,
            "fold_model_path": fold_model_path,
            "fold_log_path": fold_log_path,
        })

        if fold_best_val_loss < best_fold_val_loss:
            best_fold_val_loss = fold_best_val_loss
            best_fold_idx = fold_idx
            best_fold_epoch = fold_best_epoch
            best_fold_model_path = fold_model_path
            best_history = history

    if best_fold_model_path is None or best_history is None:
        raise RuntimeError("Training failed: no best fold model selected.")

    # 4. Save validation-selected best model under canonical names.
    model_best_path = config.ACTIVE_MODEL_PATH
    model_final_path = config.COMPAT_FINAL_MODEL_PATH
    shutil.copy2(best_fold_model_path, model_best_path)
    if os.path.abspath(model_final_path) != os.path.abspath(model_best_path):
        shutil.copy2(best_fold_model_path, model_final_path)

    print(f"\nSelected best fold: {best_fold_idx} (val_loss={best_fold_val_loss:.6f})")
    print(f"Saved validation-selected model to: {model_best_path}")
    print(f"Saved compatibility model to: {model_final_path}")

    selected_model = tf.keras.models.load_model(model_best_path)
    model_param_count = int(selected_model.count_params())
    model_file_size_bytes = int(os.path.getsize(model_best_path))
    model_file_size_mb = float(model_file_size_bytes / (1024 * 1024))
    print(
        "Selected model size: "
        f"{model_param_count:,} params, {model_file_size_mb:.2f} MB"
    )

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _get_git_commit_sha(),
        "random_seed": config.RANDOM_SEED,
        "tf_determinism_enabled": config.ENABLE_TF_DETERMINISM,
        "n_folds": 5,
        "selection_metric": "val_loss",
        "selected_fold": best_fold_idx,
        "selected_fold_best_val_loss": float(best_fold_val_loss),
        "selected_fold_best_epoch": best_fold_epoch,
        "selected_model_path": model_best_path,
        "compat_model_path": model_final_path,
        "model_size": {
            "parameter_count": model_param_count,
            "file_size_bytes": model_file_size_bytes,
            "file_size_mb": model_file_size_mb,
        },
        "folds": fold_summaries,
    }
    with open(config.TRAINING_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Training metadata saved to: {config.TRAINING_METADATA_PATH}")

    print("\n--- Model Training Complete ---")
    return best_history

if __name__ == '__main__':
    # If this script is executed directly, it will start the model training process.
    train_model()
