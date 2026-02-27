"""
This module provides functionalities for evaluating the trained model
and for making predictions on new data within the ECG Apnea project.

Functions:
    - plot_training_history(): Visualizes the training and validation metrics.
    - evaluate_model(): Loads the trained model, evaluates it on the test set,
                        and saves evaluation plots and metrics.
    - predict_on_record(): Processes a single ECG record and outputs predictions.
"""

import os
import json
import subprocess
from datetime import datetime, timezone

# Scientific computing and data analysis libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Deep learning framework and metrics
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# Import project-specific modules
from backend.src import config # For accessing configuration parameters and file paths
from backend.src.utilities.preprocess import preprocess # For preprocessing single ECG records for prediction
from backend.src.data_loader import load_data # For loading the main processed dataset
from backend.src.utilities.gradcam import make_gradcam_heatmap, save_gradcam_visualization # Import Grad-CAM utilities

# Locked evaluation threshold for paper reporting.
APNEA_THRESHOLD = 0.5


def _get_git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def plot_training_history(history: tf.keras.callbacks.History):
    """
    Plots the training and validation loss and accuracy over epochs.
    The generated plots are saved to the results directory defined in src.config.

    Args:
        history (tf.keras.callbacks.History): A Keras History object returned from model.fit(),
                                           containing training and validation metrics.
    """
    # Create a figure with two subplots side-by-side: one for loss, one for accuracy.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot training and validation loss
    axes[0].plot(history.history["loss"], "r-", label="Training Loss", linewidth=0.5)
    axes[0].plot(history.history["val_loss"], "b-", label="Validation Loss", linewidth=0.5)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend() # Display legend with labels

    # Plot training and validation accuracy
    axes[1].plot(history.history["accuracy"], "r-", label="Training Accuracy", linewidth=0.5)
    axes[1].plot(history.history["val_accuracy"], "b-", label="Validation Accuracy", linewidth=0.5)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend() # Display legend with labels

    # Adjust subplot params for a tight layout
    fig.tight_layout()
    
    # Save the plot to the RESULTS_DIR specified in src.config.
    # The plot is saved without displaying it interactively.
    plt.savefig(os.path.join(config.RESULTS_DIR, 'hist.png'))


def evaluate_model():
    """
    Evaluates the final trained model on the test dataset.

    This function performs the following steps:
    1. Loads the best trained model from the MODELS_DIR using its saved format.
    2. Loads the prepared test data using `load_data` from `src.data_loader`.
    3. Makes predictions on the test data.
    4. Saves the prediction scores to a CSV file in the RESULTS_DIR.
    5. Calculates and prints various classification metrics (Accuracy, Sensitivity, Specificity, F1-score, AUC).
    6. Generates and saves a Confusion Matrix plot to the RESULTS_DIR.
    7. Generates and saves a Receiver Operating Characteristic (ROC) Curve plot to the RESULTS_DIR.
    """
    print("\n--- Starting Model Evaluation ---")

    # Load the best trained model.
    # The model is expected to be saved in the Keras native format (.keras)
    # in the directory specified by config.MODELS_DIR.
    model_path = config.resolve_model_path()
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please ensure training was successful.")
        return

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    model_param_count = int(model.count_params())
    model_file_size_bytes = int(os.path.getsize(model_path))
    model_file_size_mb = float(model_file_size_bytes / (1024 * 1024))

    # Load the test data.
    # We only need x_test, y_test, and groups_test for evaluation.
    _, _, _, x_test, y_test, groups_test = load_data()
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

    # Make predictions (probability scores for each class) on the test data.
    y_score = model.predict(x_test)
    y_prob_non_apnea = y_score[:, 0]
    y_prob_apnea = y_score[:, 1]

    # Save prediction scores to a CSV file.
    # Includes true labels, per-class probabilities, binary predictions, and record IDs.
    y_true = y_test.astype(int)
    y_pred = (y_prob_apnea >= APNEA_THRESHOLD).astype(int)
    output_df = pd.DataFrame({
        "y_true": y_true,
        "y_score": y_prob_apnea,  # Backward-compatible column name.
        "p_apnea": y_prob_apnea,
        "p_non_apnea": y_prob_non_apnea,
        "y_pred": y_pred,
        "subject": groups_test,
    })
    output_csv_path = os.path.join(config.RESULTS_DIR, "CNN-Transformer.csv")
    output_df.to_csv(output_csv_path, index=False)
    print(f"Prediction scores saved to: {output_csv_path}")

    # Save per-record predicted prevalence summary for ranking highest/lowest files.
    subject_summary_df = (
        output_df.groupby("subject", as_index=False)
        .agg(
            total_minutes=("subject", "size"),
            predicted_apnea_minutes=("y_pred", "sum"),
            true_apnea_minutes=("y_true", "sum"),
            mean_p_apnea=("p_apnea", "mean"),
        )
    )
    subject_summary_df["predicted_apnea_ratio"] = (
        subject_summary_df["predicted_apnea_minutes"] / subject_summary_df["total_minutes"]
    )
    subject_summary_df["true_apnea_ratio"] = (
        subject_summary_df["true_apnea_minutes"] / subject_summary_df["total_minutes"]
    )
    summary_csv_path = os.path.join(config.RESULTS_DIR, "record_apnea_prevalence_summary.csv")
    subject_summary_df.sort_values("predicted_apnea_ratio", ascending=False).to_csv(summary_csv_path, index=False)
    print(f"Per-record apnea prevalence summary saved to: {summary_csv_path}")

    # Calculate various classification metrics.
    # Confusion Matrix: Helps understand classification performance (True Positives, False Positives, etc.).
    C = confusion_matrix(y_true, y_pred, labels=(1, 0)) # Labels are ordered [Positive, Negative]
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    
    # Accuracy: Overall correctness of the model.
    acc = 1. * (TP + TN) / (TP + TN + FP + FN)
    # Sensitivity (Recall): Proportion of actual positive cases that are correctly identified.
    sn = 1. * TP / (TP + FN) if (TP + FN) != 0 else 0.
    # Specificity: Proportion of actual negative cases that are correctly identified.
    sp = 1. * TN / (TN + FP) if (TN + FP) != 0 else 0.
    # F1-score: Harmonic mean of precision and recall, balancing both.
    f1 = f1_score(y_true, y_pred, average='binary')
    # AUC (Area Under the Receiver Operating Characteristic Curve): Measures the model's ability
    # to distinguish between classes across various threshold settings.
    fpr, tpr, _ = roc_curve(y_true, y_prob_apnea) # False Positive Rate, True Positive Rate
    roc_auc = auc(fpr, tpr)
    
    # Print the calculated metrics.
    print(f"\n--- Evaluation Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity (Recall): {sn:.4f}")
    print(f"Specificity: {sp:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Decision rule: predicted apnea if P(apnea) >= {APNEA_THRESHOLD:.2f}")
    print(
        "Model size: "
        f"{model_param_count:,} params, {model_file_size_mb:.2f} MB"
    )

    highest_row = subject_summary_df.loc[subject_summary_df["predicted_apnea_ratio"].idxmax()]
    lowest_row = subject_summary_df.loc[subject_summary_df["predicted_apnea_ratio"].idxmin()]
    print(
        "Highest predicted apnea prevalence record: "
        f"{highest_row['subject']} ({highest_row['predicted_apnea_ratio'] * 100:.2f}%)"
    )
    print(
        "Lowest predicted apnea prevalence record: "
        f"{lowest_row['subject']} ({lowest_row['predicted_apnea_ratio'] * 100:.2f}%)"
    )

    eval_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _get_git_commit_sha(),
        "model_path": model_path,
        "apnea_threshold": APNEA_THRESHOLD,
        "model_size": {
            "parameter_count": model_param_count,
            "file_size_bytes": model_file_size_bytes,
            "file_size_mb": model_file_size_mb,
        },
        "num_samples": int(len(y_true)),
        "num_unique_records": int(output_df["subject"].nunique()),
        "metrics": {
            "accuracy": float(acc),
            "sensitivity": float(sn),
            "specificity": float(sp),
            "f1": float(f1),
            "auc": float(roc_auc),
        },
        "highest_predicted_prevalence_record": {
            "subject": str(highest_row["subject"]),
            "predicted_apnea_ratio": float(highest_row["predicted_apnea_ratio"]),
            "true_apnea_ratio": float(highest_row["true_apnea_ratio"]),
        },
        "lowest_predicted_prevalence_record": {
            "subject": str(lowest_row["subject"]),
            "predicted_apnea_ratio": float(lowest_row["predicted_apnea_ratio"]),
            "true_apnea_ratio": float(lowest_row["true_apnea_ratio"]),
        },
    }
    with open(config.EVALUATION_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_metadata, f, indent=2)
    print(f"Evaluation metadata saved to: {config.EVALUATION_METADATA_PATH}")

    # --- Plotting ---
    # Plot and save the Confusion Matrix.
    labels_plot = ['Apnea', 'Non-Apnea'] # Labels for plot
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2) # Adjust font size for better readability
    sns.heatmap(C, annot=True, cmap='Reds', fmt='g', xticklabels=labels_plot, yticklabels=labels_plot)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    # Save the plot to the RESULTS_DIR specified in src.config.
    plt.savefig(os.path.join(config.RESULTS_DIR, 'Confusion_Matrix.png'), bbox_inches='tight', dpi=300)
    
    # Plot and save the ROC Curve.
    plt.figure()
    lw = 2 # Line width
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') # Random guess line
    plt.xlim([0.0, 1.0]) # X-axis limits
    plt.ylim([0.0, 1.05]) # Y-axis limits
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right") # Legend position
    # Save the plot to the RESULTS_DIR specified in src.config.
    plt.savefig(os.path.join(config.RESULTS_DIR, 'ROC_Curve.png'), bbox_inches='tight', dpi=300)
    
    print(f"Evaluation plots saved to: {config.RESULTS_DIR}")
    print("\n--- Model Evaluation Complete ---")


def predict_on_record(record_name: str):
    """
    Makes predictions on a single ECG record using a pre-trained model.

    This function simulates a real-time prediction scenario:
    1. Loads the trained model.
    2. Uses the `preprocess` utility to prepare the single raw ECG record.
    3. Feeds the processed data to the model for prediction.
    4. Prints the predicted probability of apnea for each minute of the record.

    Args:
        record_name (str): The base name of the record to process (e.g., 'a01', 'x05').
                           The function will look for this record in `config.RAW_DATA_DIR`.
    """
    print(f"\n--- Starting Prediction for Record: {record_name} ---")

    # Load the trained model from the MODELS_DIR.
    model_path = config.resolve_model_path()
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please ensure a model has been trained and saved.")
        return

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    model.summary() # Print model summary for confirmation

    # Preprocess the single raw ECG record using the utility function.
    # The record path is constructed using config.RAW_DATA_DIR.
    full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
    # The 'preprocess' function returns a dict with 'tensors' (model input),
    # 'minutes' (indices of evaluated minutes), and 'skipped' (indices of skipped minutes).
    out = preprocess(full_record_path)
    
    tensors = out["tensors"]
    minutes = out["minutes"]

    # Check if any valid segments were extracted from the record.
    if tensors.shape[0] == 0:
        print(f"No valid signal segments found for record {record_name} after preprocessing. "
              "Prediction cannot be made.")
    else:
        # Make predictions using the loaded model.
        # The model outputs probability scores for the two classes (non-apnea, apnea).
        preds = model.predict(tensors)

        print(f"\nPrediction Results for Record: {record_name}")
        print(f"Input tensors shape for prediction: {tensors.shape}")
        print(f"Number of minutes evaluated: {len(minutes)}, skipped segments: {len(out['skipped'])}")

        # Print the predicted probability of apnea for each evaluated minute.
        for m, p in zip(minutes, preds):
            p_flat = np.asarray(p).ravel() # Flatten prediction array
            # Extract probability for the 'apnea' class (index 1).
            # Handle cases where model might output a single value (e.g., if sigmoid output).
            prob_apnea = float(p_flat[1]) if p_flat.size == 2 else \
                         (float(p_flat[0]) if p_flat.size == 1 else float(p_flat.mean()))
            print(f"Minute {m}: P(apnea)={prob_apnea:.4f}")

    print(f"--- Prediction Complete for Record: {record_name} ---")


def generate_gradcam_for_minute(record_name: str, minute: int, model=None):
    """
    Generate Grad-CAM for a specific minute of a record.
    
    Args:
        record_name (str): The base name of the record to process (e.g., 'a01').
        minute (int): Specific minute index to visualize.
        model: Pre-loaded model (optional, loads if None).
        
    Returns:
        dict: Dictionary containing:
            - 'heatmap': The Grad-CAM heatmap array
            - 'raw_signal': The raw ECG signal for that minute
            - 'prediction': Prediction probabilities for that minute
            - 'minute': The minute index
            - 'predicted_class': 0 or 1 (Non-Apnea or Apnea)
            - 'confidence': Confidence of the prediction
    """
    print(f"\n--- Generating Grad-CAM for Record: {record_name}, Minute: {minute} ---")
    
    # Load model if not provided
    if model is None:
        model_path = config.resolve_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    
    # Preprocess the record
    full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
    out = preprocess(full_record_path)
    
    tensors = out["tensors"]
    minutes = out["minutes"]
    raw_segments = out.get("raw_segments", [])
    
    if tensors.shape[0] == 0:
        raise ValueError(f"No valid signal segments found for record {record_name}")
    
    # Find the index of the requested minute
    try:
        target_idx = minutes.index(minute)
    except ValueError:
        raise ValueError(f"Minute {minute} not found in processed segments. Available minutes: {minutes}")
    
    # Make prediction for this minute
    input_tensor = np.expand_dims(tensors[target_idx], axis=0)
    prediction = model.predict(input_tensor)[0]
    predicted_class = int(np.argmax(prediction))
    confidence = float(prediction[predicted_class])
    
    # Generate Grad-CAM heatmap
    try:
        heatmap = make_gradcam_heatmap(input_tensor, model, "last_conv_layer")
    except Exception as e:
        # Fallback: try to find the last Conv1D layer
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        if not conv_layers:
            raise ValueError("No Conv1D layers found in the model for Grad-CAM")
        last_conv_name = conv_layers[-1].name
        print(f"Using Conv1D layer: {last_conv_name}")
        heatmap = make_gradcam_heatmap(input_tensor, model, last_conv_name)
    
    # Get raw signal
    if target_idx >= len(raw_segments):
        raise ValueError(f"Raw signal not available for minute {minute}")
    raw_signal = raw_segments[target_idx]
    
    print(f"--- Grad-CAM Generation Complete for Minute: {minute} ---")
    
    return {
        "heatmap": heatmap,
        "raw_signal": raw_signal,
        "prediction": prediction,
        "minute": minute,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_label": "Apnea" if predicted_class == 1 else "Non-Apnea"
    }


def generate_gradcam_for_record(record_name: str, visualize_count: int = 3):
    """
    Generates Grad-CAM visualizations for the top confident predictions of a record.

    Args:
        record_name (str): The base name of the record to process.
        visualize_count (int): Number of top confident segments to visualize.
    """
    print(f"\n--- Starting Grad-CAM Generation for Record: {record_name} ---")

    # Load Model
    model_path = config.resolve_model_path()
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")

    # Preprocess
    full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
    out = preprocess(full_record_path)
    tensors = out["tensors"]
    minutes = out["minutes"]
    raw_segments = out["raw_segments"] # New list of raw segments

    if tensors.shape[0] == 0:
        print(f"No valid signal segments found for record {record_name}.")
        return

    # Predict
    preds = model.predict(tensors)

    # --- Grad-CAM Visualization ---
    print("\n--- Generating Grad-CAM Visualizations ---")
    gradcam_dir = os.path.join(config.RESULTS_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    # Sort by confidence
    max_probs = np.max(preds, axis=1)
    top_indices = np.argsort(max_probs)[-visualize_count:][::-1]

    for i in top_indices:
        minute = minutes[i]
        prediction_prob = preds[i]
        predicted_class = np.argmax(prediction_prob)
        class_label = "Apnea" if predicted_class == 1 else "Non-Apnea"
        confidence = prediction_prob[predicted_class]

        print(f"Generating Grad-CAM for Minute {minute} (Pred: {class_label}, Conf: {confidence:.4f})...")

        input_tensor = np.expand_dims(tensors[i], axis=0) 
        
        # Get the corresponding raw segment
        raw_signal = raw_segments[i]

        try:
            # Note: The model should have a layer named 'last_conv_layer' or equivalent Conv1D
            heatmap = make_gradcam_heatmap(input_tensor, model, "last_conv_layer")
            
            save_path = os.path.join(gradcam_dir, f"{record_name}_min{minute}_{class_label}_{confidence:.2f}.png")
            # Pass the RAW signal to the visualization function
            save_gradcam_visualization(raw_signal, heatmap, save_path)
            print(f"Saved visualization to: {save_path}")
        except Exception as e:
            print(f"Failed to generate Grad-CAM for Minute {minute}: {e}")

    print(f"--- Grad-CAM Generation Complete for Record: {record_name} ---")



if __name__ == '__main__':
    # This block allows the script to be run directly for quick testing.
    # In a typical workflow, these functions would be invoked via src/main.py.

    # Example of running full evaluation (requires a trained model and processed data)
    # evaluate_model()

    # Example of running prediction on a single record
    predict_on_record('a01')
    
