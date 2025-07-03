# src/emolex/train.py 

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, List
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping
# from tensorflow.keras.optimizers import Adam
from tf_keras.optimizers import Adam
import torch

# --- Hugging Face Imports ---
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TFDistilBertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate
# ----------------------------


def train_dl_model(
    model: tf.keras.Model, 
    X_train_pad: np.ndarray, 
    y_train: Union[np.ndarray, pd.Series],
    X_test_pad: np.ndarray, 
    y_test: Union[np.ndarray, pd.Series],
    epochs: int = 10, 
    batch_size: int = 32, 
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
        random_seed (int): Seed for TensorFlow's random operations for reproducibility. Defaults to 42.

    Returns:
        tuple[tf.keras.Model, tf.keras.callbacks.History]: A tuple containing:
            - The trained Keras model.
            - The History object containing training loss and metrics.
    """
    # Set tensorflow random seed for reproducibility
    tf.random.set_seed(random_seed)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',    
        patience=3,               
        restore_best_weights=True,
        verbose=1
    )
    callbacks = [early_stopping]

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
    X_train: Union[np.ndarray, pd.Series, list[str]],
    y_train: Union[np.ndarray, pd.Series, list[int]],
    X_test: Union[np.ndarray, pd.Series, list[str]],
    y_test: Union[np.ndarray, pd.Series, list[int]],
    num_classes: int,
    max_len: int = 128,
    output_dir: str = "./bert_output",
    num_train_epochs: int = 3,
    per_device_batch_size: int = 16,
    logging_dir: str = "./logs",
    random_seed: int = 42
) -> Tuple[Trainer, dict]: 
    """
    Trains/Fine-tunes a Hugging Face BertForSequenceClassification model.

    This function handles tokenization, conversion to Hugging Face Datasets,
    model loading, and training using the Hugging Face Trainer API.

    Args:
        X_train (np.ndarray | pd.Series | list[str]): Training text data.
        y_train (np.ndarray | pd.Series | list[int]): Training labels (integer-encoded).
        X_test (np.ndarray | pd.Series | list[str]): Evaluation text data.
        y_test (np.ndarray | pd.Series | list[int]): Evaluation labels (integer-encoded).
        num_classes (int): The number of output classes for the classification task.
        max_len (int): The maximum sequence length for tokenization. Defaults to 128.
        output_dir (str): Directory to save model checkpoints and outputs. Defaults to "./bert_output".
        num_train_epochs (int): Total number of training epochs to perform. Defaults to 3.
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

    # ----------------------------------------------------------------
    # TODO: Move dataset creation and tokenization to preprocessing.py
    print("Creating Hugging Face Datasets from input data...")
    train_hf = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
    test_hf = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})
    
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenization function (uses max_len from function parameters)
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_len)
    
    print("Tokenizing datasets...")
    train_hf_tokenized = train_hf.map(tokenize_function, batched=True, remove_columns=["text"])
    test_hf_tokenized = test_hf.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Ensure labels are native integers for the model (good practice, though Trainer often handles this)
    train_hf_tokenized = train_hf_tokenized.map(lambda x: {"label": int(x["label"])})
    test_hf_tokenized = test_hf_tokenized.map(lambda x: {"label": int(x["label"])})
    
    # Set format to PyTorch tensors for Trainer (Hugging Face Trainer's default backend)
    train_hf_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_hf_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    # ----------------------------------------------------------------

    print("Loading pre-trained BERT model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

    print("Defining training arguments...")
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
        metric_for_best_model="accuracy", # Ensure this matches a key in compute_metrics
        greater_is_better=True,
    )

    print("Loading evaluation metrics...")
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
        f1 = f1_metric.compute(predictions=predictions, references=labels_np, average='macro')["f1"]

        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        }

    print("Setting up Hugging Face Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_tokenized,
        eval_dataset=test_hf_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Starting BERT model training...")
    trainer.train()
    print("BERT model training complete.")

    print("Evaluating BERT model...")
    results = trainer.evaluate()
    print("BERT model evaluation results:", results)

    return trainer, results


def train_distilbert_model(
    X_train: Union[np.ndarray, pd.Series, list[str]],
    y_train: Union[np.ndarray, pd.Series, list[int]],
    X_test: Union[np.ndarray, pd.Series, list[str]],
    y_test: Union[np.ndarray, pd.Series, list[int]],
    num_classes: int,
    max_len: int = 128,
    output_dir: str = "./bert_output",
    num_train_epochs: int = 3,
    per_device_batch_size: int = 16,
    logging_dir: str = "./logs",
    random_seed: int = 42
) -> Tuple[Trainer, dict]: 
    """
    Trains/Fine-tunes a Hugging Face TFDistilBertForSequenceClassification model.

    This function handles tokenization, conversion to Hugging Face Datasets,
    model loading, and training using the Hugging Face Trainer API.

    Args:
        X_train (np.ndarray | pd.Series | list[str]): Training text data.
        y_train (np.ndarray | pd.Series | list[int]): Training labels (integer-encoded).
        X_test (np.ndarray | pd.Series | list[str]): Evaluation text data.
        y_test (np.ndarray | pd.Series | list[int]): Evaluation labels (integer-encoded).
        num_classes (int): The number of output classes for the classification task.
        max_len (int): The maximum sequence length for tokenization. Defaults to 128.
        output_dir (str): Directory to save model checkpoints and outputs. Defaults to "./bert_output".
        num_train_epochs (int): Total number of training epochs to perform. Defaults to 3.
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

    # ----------------------------------------------------------------
    # TODO: Move dataset creation and tokenization to preprocessing.py
    print("Creating Hugging Face Datasets from input data...")
    train_hf = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
    test_hf = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})
    
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Tokenization function (uses max_len from function parameters)
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_len)
    
    print("Tokenizing datasets...")
    train_hf_tokenized = train_hf.map(tokenize_function, batched=True, remove_columns=["text"])
    test_hf_tokenized = test_hf.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Ensure labels are native integers for the model (good practice, though Trainer often handles this)
    train_hf_tokenized = train_hf_tokenized.map(lambda x: {"label": int(x["label"])})
    test_hf_tokenized = test_hf_tokenized.map(lambda x: {"label": int(x["label"])})
    
    # Set format to PyTorch tensors for Trainer (Hugging Face Trainer's default backend)
    train_hf_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_hf_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    # ----------------------------------------------------------------

    print("Loading pre-trained DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)

    print("Defining training arguments...")
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
        metric_for_best_model="accuracy", # Ensure this matches a key in compute_metrics
        greater_is_better=True,
    )

    print("Loading evaluation metrics...")
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
        f1 = f1_metric.compute(predictions=predictions, references=labels_np, average='macro')["f1"]

        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        }

    print("Setting up Hugging Face Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_tokenized,
        eval_dataset=test_hf_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Starting DistilBERT model training...")
    trainer.train()
    print("DistilBERT model training complete.")

    print("Evaluating DistilBERT model...")
    results = trainer.evaluate()
    print("DistilBERT model evaluation results:", results)

    return trainer, results


def train_distilbert_model_v2(
    train_dataset: Dataset, # Updated type hint: Expects Hugging Face Dataset
    eval_dataset: Dataset,  # Updated type hint: Expects Hugging Face Dataset
    num_classes: int,
    output_dir: str = "./distilbert_output", # Changed default output dir
    num_train_epochs: int = 3,
    per_device_batch_size: int = 16,
    logging_dir: str = "./logs",
    random_seed: int = 42
) -> Tuple[Trainer, dict]:
    """
    Trains/Fine-tunes a Hugging Face DistilBertForSequenceClassification model (PyTorch backend).

    This function expects pre-tokenized Hugging Face Dataset objects for training
    and evaluation, loads the DistilBERT model, and trains it using the
    Hugging Face Trainer API.

    Args:
        train_dataset (datasets.Dataset): The tokenized training dataset (Hugging Face Dataset object).
        eval_dataset (datasets.Dataset): The tokenized evaluation dataset (Hugging Face Dataset object).
        num_classes (int): The number of output classes for the classification task.
        output_dir (str): Directory to save model checkpoints and outputs. Defaults to "./distilbert_output".
        num_train_epochs (int): Total number of training epochs to perform. Defaults to 3.
        per_device_batch_size (int): Batch size per GPU/CPU for training and evaluation. Defaults to 16.
        logging_dir (str): Directory for storing logs. Defaults to "./logs".
        random_seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[Trainer, dict]: A tuple containing:
            - The Hugging Face Trainer object after training.
            - A dictionary of evaluation results.
    """
    # Set seed for reproducibility
    tf.random.set_seed(random_seed) # For TensorFlow operations if any are used in the broader environment
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print("Loading pre-trained DistilBERT model...")
    # Using PyTorch model class as implemented in the original function
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)

    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        logging_dir=logging_dir,
        save_strategy="epoch",
        report_to="none", # Keep as 'none' if you don't want external logging tools (e.g., TensorBoard)
        seed=random_seed,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", # Changed to macro_f1 (consistent with compute_metrics)
        greater_is_better=True,
    )

    print("Loading evaluation metrics...")
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
        f1 = f1_metric.compute(predictions=predictions, references=labels_np, average='macro')["f1"]

        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1 # This is the metric used for best model selection
        }

    print("Setting up Hugging Face Trainer...")
    # Instantiate the EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,   
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback] # Pass the instance to the callbacks list
    )

    print("Starting DistilBERT model training...")
    trainer.train()
    print("DistilBERT model training complete.")

    print("Evaluating DistilBERT model...")
    results = trainer.evaluate()
    print("DistilBERT model evaluation results:", results)

    return trainer, results


def train_distilbert_model_tf(
    X_train: Union[np.ndarray, pd.Series, list[str]],
    y_train: Union[np.ndarray, pd.Series, list[int]],
    X_test: Union[np.ndarray, pd.Series, list[str]],
    y_test: Union[np.ndarray, pd.Series, list[int]],
    num_classes: int,
    max_len: int = 128,
    batch_size: int = 64,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    random_seed: int = 42
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains/Fine-tunes a TFDistilBertForSequenceClassification model using TensorFlow's Keras API.

    This function handles tokenization, conversion to Hugging Face Datasets,
    conversion to TensorFlow Datasets, model loading, compilation, and training.

    Args:
        X_train (np.ndarray | pd.Series | list[str]): Training text data.
        y_train (np.ndarray | pd.Series | list[int]): Training labels (integer-encoded).
        X_test (np.ndarray | pd.Series | list[str]): Validation text data.
        y_test (np.ndarray | pd.Series | list[int]): Validation labels (integer-encoded).
        num_classes (int): The number of output classes for the classification task.
        max_len (int): The maximum sequence length for tokenization. Defaults to 128.
        batch_size (int): Batch size for training and evaluation. Defaults to 64.
        epochs (int): Total number of training epochs to perform. Defaults to 3.
        learning_rate (float): Learning rate for the Adam optimizer. Defaults to 5e-5.
        random_seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[tf.keras.Model, tf.keras.callbacks.History]: A tuple containing:
            - The trained TensorFlow DistilBERT model.
            - The History object containing training loss and metrics.
    """
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # Compute class weights
    # This function expects y_train to be an array-like object (e.g., Series, list, numpy array)
    # np.unique will get the unique class labels from y_train
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # Convert to a dictionary mapping class index to weight
    class_weight_dict = dict(enumerate(class_weights))

    print("Initializing DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("Creating Hugging Face Datasets from input data...")
    # Convert input data to lists for Hugging Face Dataset creation
    train_dataset_hf = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
    test_dataset_hf = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

    print("Tokenizing datasets...")
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_dataset_tokenized = train_dataset_hf.map(tokenize, batched=True, remove_columns=["text"])
    test_dataset_tokenized = test_dataset_hf.map(tokenize, batched=True, remove_columns=["text"])

    print("Ensuring labels are native integers...")
    # Labels must be native ints for SparseCategoricalCrossentropy
    train_dataset_tokenized = train_dataset_tokenized.map(lambda x: {"label": int(x["label"])})
    test_dataset_tokenized = test_dataset_tokenized.map(lambda x: {"label": int(x["label"])})

    print("Setting dataset format to TensorFlow...")
    # Set format to TensorFlow tensors
    train_dataset_tokenized.set_format("tensorflow", columns=["input_ids", "attention_mask", "label"])
    test_dataset_tokenized.set_format("tensorflow", columns=["input_ids", "attention_mask", "label"])

    print("Creating data collator...")
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    print("Converting Hugging Face Datasets to TensorFlow Datasets...")
    # Convert to TensorFlow Datasets
    # Note: `collate_fn` is used with `tf_train.batch()` or `tf_test.batch()` in a custom loop,
    # but when using `to_tf_dataset` directly, the `collate_fn` applies during conversion.
    # The `label_cols` argument handles mapping for Keras.
    tf_train = train_dataset_tokenized.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],  # Must be a list for `to_tf_dataset`
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator, # Apply collator during conversion to TF dataset
    )

    tf_test = test_dataset_tokenized.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],  # Must be a list
        shuffle=False, # No need to shuffle test set
        batch_size=batch_size,
        collate_fn=data_collator, # Apply collator during conversion to TF dataset
    )

    print(f"Loading TFDistilBertForSequenceClassification model with {num_classes} labels...")
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=num_classes
    )

    print("Compiling model...")
    model.compile(
        # Change this line:
        optimizer="adam",
        # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print(f"Starting DistilBERT model training for {epochs} epochs...")
    history = model.fit(
        tf_train,
        validation_data=tf_test,
        epochs=epochs,
        class_weight=class_weight_dict
    )

    print("DistilBERT model training complete.")
    return model, history
