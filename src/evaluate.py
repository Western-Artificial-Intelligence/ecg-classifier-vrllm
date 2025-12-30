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

# Scientific computing and data analysis libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Deep learning framework and metrics
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# Import project-specific modules
from src import config # For accessing configuration parameters and file paths
from src.utilities.preprocess import preprocess # For preprocessing single ECG records for prediction
from src.data_loader import load_data # For loading the main processed dataset


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
    model_path = os.path.join(config.MODELS_DIR, 'model.final.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please ensure training was successful.")
        return

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")

    # Load the test data.
    # We only need x_test, y_test, and groups_test for evaluation.
    _, _, _, x_test, y_test, groups_test = load_data()
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

    # Make predictions (probability scores for each class) on the test data.
    y_score = model.predict(x_test)

    # Save prediction scores to a CSV file.
    # This includes true labels, predicted scores for the positive class (apnea),
    # and the original subject (record) name for each segment.
    output_df = pd.DataFrame({"y_true": y_test, "y_score": y_score[:, 1], "subject": groups_test})
    output_csv_path = os.path.join(config.RESULTS_DIR, "CNN-Transformer-LSTM.csv")
    output_df.to_csv(output_csv_path, index=False)
    print(f"Prediction scores saved to: {output_csv_path}")

    # Convert probability scores to binary predictions (0 or 1) by taking the class with higher probability.
    y_true = y_test # True binary labels
    y_pred = np.argmax(y_score, axis=-1) # Predicted binary labels

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
    fpr, tpr, _ = roc_curve(y_true, y_score[:, 1]) # False Positive Rate, True Positive Rate
    roc_auc = auc(fpr, tpr)
    
    # Print the calculated metrics.
    print(f"\n--- Evaluation Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity (Recall): {sn:.4f}")
    print(f"Specificity: {sp:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")

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
    # It expects the model to be saved as 'model.final.keras'.
    model_path = os.path.join(config.MODELS_DIR, "model.final.keras")
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


if __name__ == '__main__':
    # This block allows the script to be run directly for quick testing.
    # In a typical workflow, these functions would be invoked via src/main.py.

    # Example of running full evaluation (requires a trained model and processed data)
    # evaluate_model()

    # Example of running prediction on a single record
    predict_on_record('a01')
