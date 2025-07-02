# src/emolex/dl_models.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate


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
