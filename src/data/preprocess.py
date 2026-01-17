import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_aps_data(filepath):
    """Loads a dataset from a CSV file.

    Ignores the license headline and parses 'na' as NaN.

    Args:
        filepath (str): Relative path to the CSV file (e.g., 'aps_failure_training_set.csv'). 
            The path is relative to the current working directory.

    Returns:
        pd.DataFrame: Loaded data.

    """
    df = pd.read_csv(filepath, skiprows=20, na_values='na')
    return df


def encode_target(y_series):
    """Encodes categorical target labels 'neg'/'pos' as integerts 0/1.
    Args:
        y_series (pd.Series): A pandas Series containing strings 'neg' and 'pos'.
    
    Returns:
        pd.Series: A new Serries with strings mapped to 0 for 'neg' and 1 for 'pos'.
    """
    return y_series.map({'neg': 0, 'pos': 1})


def get_cols_with_missing_threshold(df, threshold=0.5):
    """Returns a list of columns to drop that have more missing values than an established threshold (by deafault 50%). 

    This operation should be used on training dataset to identify features with insufficient data.
    
    Args:
        df (pd.DataFrame): A pandas DataFrame to analyze.
        threshold (float): The maximum allowed fraction of missing values (0.0 to 1.0). Names of columns strictly greater than this value are returned in a list. Defaults to 0.5.

    Returns:
        list[str]: A list of columns names to be dropped.
    """
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    return cols_to_drop


def get_imputation_pipeline(strategy='median'):
    """Creates a scikit-learn pipeline for data imputation. 

    The pipeline contains a single step using SimpleImputer with chosen strategy (defaults to median).

    Args:
        strategy (str): The imputation strategy. Supported values: 'median' (default), 'mean', 'most_frequent', 'constant'.

    Returns:
        Pipeline: A scikit-learn Pipeline object containing the 'imputer' step.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy=strategy))
    ])

def get_scaling_pipeline():
    """Creates a scikit-learnb pipeline for data standarization.

    The pipeline contains a single step using StandardScaler to transform features by scaling them to unit variance. 
    
    Returns:
        Pipeline: A scikit-learn Pipeline object containing the 'scaler' step.
    """
    return Pipeline([
        ('scaler', StandardScaler())
    ])


def check_split_integrity(X_train, X_test, y_train, y_test):
    """Checks and prints the integrity and class distribution of train/test datasets.

    This utility function verifies if the feature sets have consistent columns and 
    compares the target class distribution (positive ratio) between the training 
    and testing sets.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.

    Returns:
        None: This function prints the integrity report directly to the console.
    """
    print(f"Kształt X_train: {X_train.shape}")
    print(f"Kształt X_test:  {X_test.shape}")

    # Sprawdzenie czy mamy te same kolumny w tej samej kolejności w obu zbiorach 
    if list(X_train.columns) != list(X_test.columns):
        print("UWAGA: Zbiory mają różne kolumny!")
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        if missing_in_test:
            print(f"Kolumny w train, których brak w test: {missing_in_test}")
    else:
        print("Kolumny w obu zbiorach są zgodne.")

    pos_ratio_train = y_train.mean()
    pos_ratio_test = y_test.mean()

    print(f"Procent klasy pozytywnej (awaria) w Train: {pos_ratio_train:.2%}")
    print(f"Procent klasy pozytywnej (awaria) w Test:  {pos_ratio_test:.2%}")