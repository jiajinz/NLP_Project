# src/emolex/dl_models.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd


def lstm_model(num_classes: int, vocab_size: int, max_len: int, embedding_dim: int = 128) -> tf.keras.Model:
    """
    Builds, compiles, and summarizes a Sequential LSTM model for text classification.

    Args:
        num_classes (int): The number of output classes for the classification task.
                           This determines the size of the final Dense layer.
        vocab_size (int): The size of the vocabulary (number of unique words + 1 for padding/OOV).
                          This is the `input_dim` for the Embedding layer.
        max_len (int): The maximum sequence length for input texts.
                       This is the `input_length` for the Embedding layer.
        embedding_dim (int): The dimension of the word embeddings. Defaults to 128.

    Returns:
        tf.keras.Model: The compiled Keras Sequential model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
    
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=optimizer, 
        metrics=['accuracy']
    )

    # Building the model ensures input_shape is set and allows summary to show correct shapes
    model.build(input_shape=(None, max_len)) # (None for batch size, max_len for sequence length)
    model.summary()
    
    return model


def bilstm_model(num_classes: int, vocab_size: int, max_len: int, embedding_dim: int = 128) -> tf.keras.Model:
    """
    Builds, compiles, and summarizes a Sequential Bidirectional LSTM model for text classification.

    Args:
        num_classes (int): The number of output classes for the classification task.
                           This determines the size of the final Dense layer.
        vocab_size (int): The size of the vocabulary (number of unique words + 1 for padding/OOV).
                          This is the `input_dim` for the Embedding layer.
        max_len (int): The maximum sequence length for input texts.
                       This is the `input_length` for the Embedding layer.
        embedding_dim (int): The dimension of the word embeddings. Defaults to 128.

    Returns:
        tf.keras.Model: The compiled Keras Sequential model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        Dropout(0.3),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
    
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=optimizer, 
        metrics=['accuracy']
    )

    # Building the model ensures input_shape is set and allows summary to show correct shapes
    model.build(input_shape=(None, max_len)) # (None for batch size, max_len for sequence length)
    model.summary()
    
    return model


def train_dl_model(
    model: tf.keras.Model, 
    X_train_pad: np.ndarray, 
    y_train: [np.ndarray | pd.Series],
    X_test_pad: np.ndarray, 
    y_test: [np.ndarray | pd.Series],
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
        y_train (np.ndarray | pd.Series): Training labels (encoded).
        X_test_pad (np.ndarray): Padded and tokenized validation features.
        y_test (np.ndarray | pd.Series): Validation labels (encoded).
        epochs (int): Number of training epochs. Defaults to 10.
        batch_size (int): Number of samples per gradient update. Defaults to 32.
        callbacks (list[tf.keras.callbacks.Callback]): List of Keras callbacks to apply during training.
                                                       Defaults to None.
        random_seed (int): Seed for TensorFlow's random operations for reproducibility. Defaults to 42.

    Returns:
        tuple[tf.keras.Model, tf.keras.callbacks.History]: A tuple containing:
            - The trained Keras model.
            - The History object containing training loss and metrics.
    """
    # Set tensorflow random seed for reproducibility
    tf.random.set_seed(random_seed)

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
        verbose=1
    )
    
    print("Model training complete.")
    return model, history 
