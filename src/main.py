"""
This module serves as the main command-line interface (CLI) entry point
for the ECG Apnea project. It allows users to orchestrate and run different
stages of the machine learning pipeline (preprocessing, training, evaluation,
and prediction) using simple command-line arguments.

Commands:
    - `preprocess`: Runs the data preprocessing pipeline.
    - `train`: Trains the CNN-Transformer model.
    - `evaluate`: Evaluates a trained model on the test set.
    - `predict <record_name>`: Makes predictions on a single specified ECG record.
"""

import argparse # Standard library for parsing command-line arguments

# Import core modules of the project.
# These modules contain the functions that implement each stage of the pipeline.
from src import train        # Module for model training functionalities
from src import evaluate     # Module for model evaluation and prediction functionalities
from src import preprocessing # Module for the main data preprocessing pipeline


def main():
    """
    The main function that sets up the command-line interface (CLI)
    and dispatches commands to the respective modules.

    It parses arguments from the command line and calls the appropriate
    functions to execute the requested action.
    """
    # Create the top-level parser for the application's CLI.
    parser = argparse.ArgumentParser(
        description="ECG Apnea Project CLI - Manage and run the ML pipeline stages."
    )
    
    # Create subparsers for each distinct command (preprocess, train, evaluate, predict).
    # 'dest="command"' stores the name of the subcommand chosen by the user.
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Available commands to run the ECG Apnea pipeline stages."
    )

    # --- 'preprocess' command parser ---
    # This command triggers the full data preprocessing pipeline.
    preprocess_parser = subparsers.add_parser(
        "preprocess", 
        help="Run the comprehensive data preprocessing pipeline to generate apnea-ecg.pkl."
    )
    
    # --- 'train' command parser ---
    # This command initiates the model training process.
    train_parser = subparsers.add_parser(
        "train", 
        help="Train the CNN-Transformer model on the preprocessed data."
    )

    # --- 'evaluate' command parser ---
    # This command performs a full evaluation of the trained model on the test set.
    evaluate_parser = subparsers.add_parser(
        "evaluate", 
        help="Evaluate a trained model's performance on the test dataset, including metrics and plots."
    )

    # --- 'predict' command parser ---
    # This command allows making predictions on a single, specified ECG record.
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Make predictions on a single ECG record using a trained model."
    )
    # Add an argument for the record name, which is required for the 'predict' command.
    predict_parser.add_argument(
        "record_name", 
        type=str, 
        help="The base name of the ECG record to process (e.g., 'a01'). It expects the file "
             "to be located in the raw data directory defined in config.RAW_DATA_DIR."
    )

    # Parse the command-line arguments provided by the user.
    args = parser.parse_args()

    # --- Command Dispatcher ---
    # Based on the 'command' argument, call the appropriate function from the
    # respective module.

    if args.command == "preprocess":
        print("\n--- Running Data Preprocessing Pipeline ---")
        # Invokes the main preprocessing function from the `preprocessing` module.
        preprocessing.run_preprocessing()
        print("--- Data Preprocessing Complete ---")

    elif args.command == "train":
        print("\n--- Starting Model Training ---")
        # Invokes the model training function from the `train` module.
        # It returns a history object which contains training/validation metrics per epoch.
        history = train.train_model()
        print("\n--- Training Complete. Now initiating evaluation of the trained model ---")
        # After training, automatically calls evaluation functions to show results.
        if history: # Check if training produced a history object
            evaluate.plot_training_history(history) # Plot training loss/accuracy curves
        evaluate.evaluate_model() # Run full evaluation on the test set

    elif args.command == "evaluate":
        print("\n--- Starting Model Evaluation ---")
        # Invokes the model evaluation function from the `evaluate` module.
        evaluate.evaluate_model()
        print("--- Model Evaluation Complete ---")

    elif args.command == "predict":
        print(f"\n--- Starting Prediction for Record: {args.record_name} ---")
        # Invokes the prediction function for a single record from the `evaluate` module.
        evaluate.predict_on_record(args.record_name)
        print(f"--- Prediction Complete for Record: {args.record_name} ---")

    else:
        # If no command or an unknown command is provided, print the help message.
        parser.print_help()

if __name__ == "__main__":
    # This block ensures that the 'main()' function is called only when the
    # script is executed directly (e.g., `python src/main.py train`),
    # not when it's imported as a module into another script.
    main()
