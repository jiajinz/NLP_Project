# src/emolex/ml_models.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC 
import scipy.sparse

def train_tfidf_lr_model(
    X_train_tfidf: scipy.sparse.csr_matrix,
    y_train: pd.Series, 
    random_seed: int = 42
) -> LogisticRegression:
    """
    Trains a Logistic Regression model using pre-computed TF-IDF features.

    Args:
        X_train_tfidf (scipy.sparse.csr_matrix): TF-IDF features for the training set (sparse matrix).
        y_train (pd.Series): Training labels (e.g., 'label_encoded' column).
        random_seed (int): Seed for the random number generator for reproducibility.
                           Defaults to 42.

    Returns:
        LogisticRegression: The trained LogisticRegression model.
    """
    print("Training Logistic Regression model...")
    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=random_seed)
    model.fit(X_train_tfidf, y_train) 
    print("Logistic Regression model training complete.")

    return model

def train_tfidf_svm_model(
    X_train_tfidf: scipy.sparse.csr_matrix,
    y_train: pd.Series, 
    random_seed: int = 42
) -> LinearSVC:
    """
    Trains a Linear Support Vector Classification (SVC) model using pre-computed TF-IDF features.

    Args:
        X_train_tfidf (scipy.sparse.csr_matrix): TF-IDF features for the training set (sparse matrix).
        y_train (pd.Series): Training labels (e.g., 'label_encoded' column).
        random_seed (int): Seed for the random number generator for reproducibility.
                           Defaults to 42.

    Returns:
        LinearSVC: The trained LinearSVC model.
    """
    print("Training Linear SVM model...")
    model = LinearSVC(random_state=random_seed, max_iter=2000) 
    model.fit(X_train_tfidf, y_train)
    print("Linear SVM model training complete.")

    return model