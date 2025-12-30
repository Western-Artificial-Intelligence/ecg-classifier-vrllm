"""
This module is responsible for training the CNN-Transformer model
defined in `src.model` using the preprocessed data loaded via `src.data_loader`.
It orchestrates the training loop, applies callbacks for monitoring and
optimization, and saves the trained model.

Functions:
    - train_model(): Loads data, builds and compiles the model,
                     runs the training process, and saves the final model.
"""

import os

# Deep learning framework and libraries
import tensorflow as tf
import keras

# Import project-specific modules
from src import config # For accessing configuration parameters (e.g., file paths, model params)
from src.data_loader import load_data # For loading prepared training and testing data
from src.model import create_model # For instantiating the CNN-Transformer model architecture


def train_model() -> keras.callbacks.History:
    """
    Orchestrates the model training process.

    This function performs the following steps:
    1. Loads training and testing data using `load_data` from `src.data_loader`.
    2. Converts target labels to a categorical format required by the model.
    3. Initializes the CNN-Transformer model using `create_model` from `src.model`.
    4. Compiles the model with a specified optimizer, loss function, and metrics.
    5. Defines and applies Keras callbacks for model checkpointing, early stopping,
       learning rate reduction, and logging.
    6. Executes the training loop using `model.fit()`.
    7. Saves the final trained model.

    Returns:
        keras.callbacks.History: A History object containing records of training loss
                                 values and metrics over successive epochs.
    """
    print("--- Starting Model Training ---")

    # 1. Load the data using the data_loader module.
    # We only need x_train, y_train, x_test, y_test for training.
    x_train, y_train, _, x_test, y_test, _ = load_data()

    # 2. Convert labels to categorical format (one-hot encoding).
    # This is necessary because the model uses 'softmax' activation in the output layer
    # and 'binary_crossentropy' loss, which expects categorical labels for binary classification.
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    print(f"Loaded data shapes: Training (features:{x_train.shape}, labels:{y_train.shape}), "
          f"Testing (features:{x_test.shape}, labels:{y_test.shape})")

    # 3. Create the model instance using the architecture defined in src.model.
    # input_shape is derived from the loaded training data.
    model = create_model(input_shape=x_train.shape[1:])
    print("\nModel Summary:")
    model.summary() # Prints a summary of the model's layers and parameters.

    # 4. Compile the model.
    # 'adam' optimizer is a popular choice for deep learning.
    # 'binary_crossentropy' is used for binary classification problems with categorical labels.
    # 'accuracy' is included as a metric to monitor performance during training.
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # 5. Define callbacks for training control and logging.
    # Callbacks are functions that run at certain stages of the training process.
    
    # ModelCheckpoint: Saves the model (weights and architecture) periodically.
    # 'filepath' uses config.MODELS_DIR to store the best model based on validation loss.
    # 'save_best_only=True' ensures only the model with the best validation loss is kept.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.MODELS_DIR, 'model.keras'),
        monitor='val_loss', # Metric to monitor
        verbose=1,          # Log when a new best model is saved
        save_best_only=True # Only save when validation loss improves
    )
    
    # EarlyStopping: Stops training if the monitored metric (validation loss)
    # does not improve for a specified number of epochs ('patience').
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Metric to monitor
        patience=30,        # Number of epochs with no improvement after which training will be stopped
        verbose=1           # Log when early stopping occurs
    )
    
    # ReduceLROnPlateau: Reduces the learning rate when the monitored metric
    # (validation loss) has stopped improving. This can help the model converge better.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', # Metric to monitor
        patience=3,         # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1           # Log when learning rate is reduced
    )
    
    # CSVLogger: Streams epoch results to a CSV file, useful for plotting training history later.
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(config.RESULTS_DIR, 'log.csv'), # Path to save the log file
        separator=',',  # Separator for CSV values
        append=True     # Append results if the file already exists
    )

    # Combine all callbacks into a list.
    callbacks_list = [checkpoint, early_stopping, reduce_lr, csv_logger]

    # 6. Train the model.
    # 'x_train', 'y_train': Training data and labels.
    # 'batch_size': Number of samples per gradient update.
    # 'epochs': Number of complete passes through the training dataset.
    # 'validation_data': Data on which to evaluate the loss and any model metrics at the end of each epoch.
    # 'callbacks': The list of callbacks to apply during training.
    history = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=20, # config.EPOCHS could be used here if defined in config.py
        validation_data=(x_test, y_test),
        callbacks=callbacks_list
    )

    # 7. Save the final trained model.
    # The model is saved in the Keras native format (.keras) for better compatibility and features.
    # The model.keras file is saved periodically by the ModelCheckpoint callback,
    # and this line ensures the final state of the model (after all epochs) is also saved.
    model.save(os.path.join(config.MODELS_DIR, "model.final.keras"))

    print("\n--- Model Training Complete ---")
    return history

if __name__ == '__main__':
    # If this script is executed directly, it will start the model training process.
    train_model()
