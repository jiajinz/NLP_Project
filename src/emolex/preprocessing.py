# src/emolex/preprocessing.py

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertTokenizerFast, DistilBertTokenizer, DistilBertTokenizerFast
from datasets import Dataset
from typing import Union


# NLTK Downloads
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4') 
except LookupError:
    nltk.download('omw-1.4')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_mental_health_sentiment_dataset() -> pd.DataFrame: 
    """
    Loads data from "mental_health_sentiment.csv" into a DataFrame.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # Get the directory of the current file (preprocessing.py)
    current_file_dir = Path(__file__).parent # This is 'src/emolex/'

    # Navigate to project root
    project_root = current_file_dir.parent.parent
    data_file_path = project_root / "data" / "mental_health_sentiment.csv"
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_file_path.resolve()}")
    # Read file into df
    df = pd.read_csv(data_file_path, index_col=0)
    # Rename columns, drop Nulls, drop duplicates
    df = df.rename(columns={"statement": "text", "status": "label"}) \
        .dropna() \
        .drop_duplicates() \
        .reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """
    Cleans and preprocesses text: lowercasing, removing punctuation/numbers,
    tokenization, stop word removal, and lemmatization.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = str(text).lower()
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    tokens = text.split()
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def encode_sentiment_labels(df: pd.DataFrame, label_col: str = 'label') -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encodes a categorical column (default 'label') into a numerical 'label_encoded' column
    using LabelEncoder.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_col (str): The name of the column containing categorical labels to encode.
                         Defaults to 'label'.

    Returns:
        tuple[pd.DataFrame, LabelEncoder]: A tuple containing:
            - The DataFrame with a new 'label_encoded' column.
            - The fitted LabelEncoder object.
    """
    encoder = LabelEncoder()
    df['label_encoded'] = encoder.fit_transform(df[label_col])

    # Print encoding map for reference
    # Using zip with sorted classes ensures a consistent order for the map
    label_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("Label Encoding Map:", label_map)
    
    return df, encoder


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing sets based on 'clean_text' (X)
    and 'label_encoded' (y) columns. Ensures stratification by 'y'.

    Args:
        df (pd.DataFrame): The input DataFrame which must contain 'clean_text' and 'label_encoded' columns.
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Controls the shuffling applied to the data before applying the split.
                            Pass an int for reproducible output across multiple function calls.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series, pd.Series]: A tuple containing:
            X_train, X_test, y_train, y_test
    """
    X = df['clean_text']
    y = df['label_encoded']
    
    # Call the imported scikit-learn's train_test_split function (aliased as sk_train_test_split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        stratify=y, # Ensures that the proportion of classes is the same in both splits
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def dl_text_vectorization(
    X_train: pd.Series, 
    X_test: pd.Series, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    vocab_size: int = 10000, 
    max_len: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs text tokenization and padding, and filters out empty sequences.

    Args:
        X_train (pd.Series): Training text data (e.g., cleaned text).
        X_test (pd.Series): Testing text data (e.g., cleaned text).
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        vocab_size (int): The maximum number of words to keep, based on word frequency.
                          Only the most common `vocab_size-1` words will be kept.
        max_len (int): The maximum length of all sequences. Shorter sequences are padded;
                       longer sequences are truncated.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            X_train_pad_filtered (NumPy array), X_test_pad_filtered (NumPy array),
            y_train_filtered (NumPy array), y_test_filtered (NumPy array)
    """ 
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    # Convert y_train and y_test to NumPy arrays if they aren't already,
    # to ensure consistent indexing behavior with boolean masks
    y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
    y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test


    # Filter out empty sequences (sequences that sum to 0 after padding, meaning they were empty or just OOV)
    # Ensure masks are applied consistently to X and y
    non_empty_train = X_train_pad.sum(axis=1) > 0
    X_train_pad_filtered = X_train_pad[non_empty_train]
    y_train_filtered = y_train_np[non_empty_train]

    non_empty_test = X_test_pad.sum(axis=1) > 0
    X_test_pad_filtered = X_test_pad[non_empty_test]
    y_test_filtered = y_test_np[non_empty_test]

    print(f"Original X_train shape: {X_train.shape}, Filtered X_train_pad shape: {X_train_pad_filtered.shape}")
    print(f"Original X_test shape: {X_test.shape}, Filtered X_test_pad shape: {X_test_pad_filtered.shape}")

    # Return the tokenizer object as well, it might be useful for inference
    # or to inspect the word index later. You could add it to the return tuple
    # if you need it outside this function. For now, sticking to original return.
    return X_train_pad_filtered, X_test_pad_filtered, y_train_filtered, y_test_filtered


def tfidf_text_vectorization(
    X_train: pd.Series, 
    X_test: pd.Series, 
    max_features: int = 10000
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Performs TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction
    on training and testing text data.

    Args:
        X_train (pd.Series): Training text data (e.g., 'clean_text' column).
        X_test (pd.Series): Testing text data (e.g., 'clean_text' column).
        max_features (int): The maximum number of features (vocabulary size) for TF-IDF.
                            Only the top `max_features` terms ordered by term frequency
                            across the corpus will be used. Defaults to 10000.

    Returns:
        tuple[np.ndarray, np.ndarray, TfidfVectorizer]: A tuple containing:
            - X_train_tfidf (np.ndarray): TF-IDF features for the training set.
            - X_test_tfidf (np.ndarray): TF-IDF features for the testing set.
            - vectorizer (TfidfVectorizer): The fitted TfidfVectorizer object,
                                            essential for transforming new, unseen text data.
    """
    print(f"Starting TF-IDF feature extraction with max_features={max_features}...")
    # TF-IDF feature extraction
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Fit the vectorizer on X_train and transform X_train
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform X_test using the *fitted* vectorizer from X_train
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF transformed X_train_tfidf shape: {X_train_tfidf.shape}")
    print(f"TF-IDF transformed X_test_tfidf shape: {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer


def hf_vectorization(
    model_type: str,
    X_train: Union[np.ndarray, pd.Series, List[str]],
    X_test: Union[np.ndarray, pd.Series, List[str]],
    y_train: Union[np.ndarray, pd.Series, List[int]],
    y_test: Union[np.ndarray, pd.Series, List[int]],
    max_len: int = 100
) -> Tuple[Dataset, Dataset]:
    """
    Performs tokenization and prepares datasets for Hugging Face Transformer models.

    This function converts raw text and labels into Hugging Face Dataset objects,
    tokenizes them using the specified model's tokenizer, and sets the format
    to PyTorch tensors for use with the Hugging Face Trainer.

    Args:
        model_type (str): The type of the transformer model. Expected values are
                          "bert" or "distilbert".
        X_train (Union[np.ndarray, pd.Series, List[str]]): Training text data.
        X_test (Union[np.ndarray, pd.Series, List[str]]): Test/Evaluation text data.
        y_train (Union[np.ndarray, pd.Series, List[int]]): Training labels (integer-encoded).
        y_test (Union[np.ndarray, pd.Series, List[int]]): Test/Evaluation labels (integer-encoded).
        max_len (int): The maximum sequence length for tokenization. Defaults to 100.

    Returns:
        Tuple[datasets.Dataset, datasets.Dataset]: A tuple containing the tokenized
        training and test Hugging Face Dataset objects, formatted for PyTorch.

    Raises:
        ValueError: If an unsupported `model_type` is provided.
    """
    print("Creating Hugging Face Datasets from input data...")
    train_hf = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
    test_hf = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

    # Load appropriate tokenizer
    if model_type == "bert":
        print("Loading BERT tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_type == "distilbert": # Corrected syntax
        print("Loading DistilBERT tokenizer...")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased") # Using DistilBertTokenizerFast
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'bert' or 'distilbert'.")

    print("Tokenizing datasets...")
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_len)

    train_hf_tokenized = train_hf.map(tokenize_function, batched=True, remove_columns=["text"])
    test_hf_tokenized = test_hf.map(tokenize_function, batched=True, remove_columns=["text"])

    # Ensure labels are native integers for the model (good practice, though Trainer often handles this)
    train_hf_tokenized = train_hf_tokenized.map(lambda x: {"label": int(x["label"])})
    test_hf_tokenized = test_hf_tokenized.map(lambda x: {"label": int(x["label"])})

    # Set format to PyTorch tensors for Trainer (Hugging Face Trainer's default backend)
    # The Trainer expects 'input_ids', 'attention_mask', and 'labels'
    train_hf_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_hf_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_hf_tokenized, test_hf_tokenized