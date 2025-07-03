# src/emolex/evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Union


def plot_training_history(history: tf.keras.callbacks.History) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the training and validation accuracy from a Keras model's history.

    Args:
        history (tf.keras.callbacks.History): The History object returned by model.fit().

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5)) 
    
    # Check if 'accuracy' and 'val_accuracy' keys exist
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    else:
        # Fallback for models not tracking accuracy (e.g., pure regression)
        print("Accuracy metrics not found in history. Plotting loss instead.")
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_ylabel("Loss")

    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_title("Training and Validation Accuracy (or Loss)")
    fig.tight_layout()
    fig.show()
    return fig, ax


def generate_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series], 
    class_labels: list[str] = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generates and plots a confusion matrix.

    Args:
        y_true (np.ndarray | pd.Series): True labels.
        y_pred (np.ndarray | pd.Series): Predicted labels.
        class_labels (list[str], optional): List of class names to use for x and y ticks.
                                            If None, integer labels will be used. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib Figure and Axes objects.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(
        cm, 
        annot=True, 
        xticklabels=class_labels, 
        yticklabels=class_labels, 
        fmt='d', 
        cmap="Blues", 
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.show()
    return fig, ax

def generate_classification_report(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series], 
    class_labels: list[str] = None
) -> str:
    """
    Generates and prints a classification report.

    Args:
        y_true (np.ndarray | pd.Series): True labels.
        y_pred (np.ndarray | pd.Series): Predicted labels.
        class_labels (list[str], optional): List of string names for the labels.
                                            If None, numerical labels will be used. Defaults to None.

    Returns:
        str: The formatted classification report.
    """
    report = classification_report(y_true, y_pred, target_names=class_labels) 
    print(report)