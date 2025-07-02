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


def train_dl_model(
    model: tf.keras.Model, 
    X_train_pad: np.ndarray, 
    y_train: Union[np.ndarray | pd.Series],
    X_test_pad: np.ndarray, 
    y_test: Union[np.ndarray | pd.Series],
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

    
def train_bert_model(
    num_classes: int, 
    train_dataset: Dataset, 
    eval_dataset: Dataset,
    output_dir: str = "./bert_output",
    num_train_epochs: int = 2,
    per_device_batch_size: int = 16,
    logging_dir: str = "./logs",
    random_seed: int = 42
) -> tuple[Trainer, dict]:
    """
    Trains/Fine-tunes a Hugging Face BertForSequenceClassification model.

    Args:
        num_classes (int): The number of output classes for the classification task.
        train_dataset (datasets.Dataset): The tokenized training dataset (Hugging Face Dataset object).
        eval_dataset (datasets.Dataset): The tokenized evaluation dataset (Hugging Face Dataset object).
        output_dir (str): Directory to save model checkpoints and outputs. Defaults to "./bert_output".
        num_train_epochs (int): Total number of training epochs to perform. Defaults to 2.
        per_device_batch_size (int): Batch size per GPU/CPU for training and evaluation. Defaults to 16.
        logging_dir (str): Directory for storing logs. Defaults to "./logs".
        random_seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[Trainer, dict]: A tuple containing:
            - The Hugging Face Trainer object after training.
            - A dictionary of evaluation results.
    """
    # Set seed for reproducibility
    tf.random.set_seed(random_seed) # For TensorFlow operations within Hugging Face
    np.random.seed(random_seed)
    # If using PyTorch backend for Hugging Face (default), also set torch seeds:
    
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print("Loading pre-trained BERT model...")
    # Load pre-trained model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

    print("Defining training arguments...")
    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        logging_dir=logging_dir,
        save_strategy="epoch",
        report_to="none", 
        seed=random_seed, 
        evaluation_strategy="epoch", 
        load_best_model_at_end=True, 
        # Changed metric_for_best_model to match one of the keys returned by compute_metrics
        # "macro_f1" or "accuracy" are good choices. Let's use "macro_f1" as it's a balanced metric.
        metric_for_best_model="macro_f1", 
        greater_is_better=True, 
    )

    print("Loading evaluation metrics...")
    # Evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Ensure labels are numpy arrays for metrics computation if they are not already
        labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels_np)["accuracy"]
        # For multi-class, 'macro' average is common
        precision = precision_metric.compute(predictions=predictions, references=labels_np, average="macro")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels_np, average="macro")["recall"]
        # Corrected: Use the variable 'f1' which holds the computed f1_metric result
        f1 = f1_metric.compute(predictions=predictions, references=labels_np, average='macro')["f1"]
        
        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1 
        }

    print("Setting up Hugging Face Trainer...")
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting BERT model training...")
    # Train
    trainer.train()
    print("BERT model training complete.")

    print("Evaluating BERT model...")
    # Evaluate
    results = trainer.evaluate()
    print("BERT model evaluation results:", results)

    return trainer, results
