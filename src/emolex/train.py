# src/emolex/train.py

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np # For type hinting X_train_pad, X_test
import pandas as pd # For type hinting y_train, y_test (if they are Series)

# Define callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',    
    patience=3,               
    restore_best_weights=True,
    verbose=1
)

def train_model(
    model: tf.keras.Model, 
    X_train_pad: np.ndarray, 
    y_train: np.ndarray, # Or pd.Series if you prefer
    X_test_pad: np.ndarray, 
    y_test: np.ndarray, # Or pd.Series if you prefer
    epochs: int = 10, 
    batch_size: int = 32, 
    callbacks: list[Callback] = None,
    random_seed: int = 42
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains a Keras deep learning model and returns the trained model and its training history.

    Args:
        model (tf.keras.Model): The compiled Keras model to be trained.
        X_train_pad (np.ndarray): Padded and tokenized training features.
        y_train (np.ndarray): Training labels (encoded).
        X_test_pad (np.ndarray): Padded and tokenized validation features.
        y_test (np.ndarray): Validation labels (encoded).
        epochs (int): Number of training epochs. Defaults to 10.
        batch_size (int): Number of samples per gradient update. Defaults to 32.
        callbacks (list[tf.keras.callbacks.Callback]): List of Keras callbacks to apply during training.
                                                       Defaults to None.

    Returns:
        tuple[tf.keras.Model, tf.keras.callbacks.History]: A tuple containing:
            - The trained Keras model.
            - The History object containing training loss and metrics.
    """
    # Set tensorflow random seed
    tf.random.set_seed(42)

    # Ensure callbacks is an empty list if None
    if callbacks is None:
        callbacks = [] 

    print(f"Starting model training for {epochs} epochs with batch size {batch_size}...")
    
    history = model.fit(
        X_train_pad, 
        y_train,
        validation_data=(X_test_pad, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1 # Show progress bar during training
    )
    
    print("Model training complete.")
    return history