import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_aps_data(filepath):
    """
    Wczytuje dane, pomija nagłówek licencyjny i parsuje 'na' jako NaN.
    """
    df = pd.read_csv(filepath, skiprows=20, na_values='na')
    return df


def encode_target(y_series):
    """
    Zamienia tekst 'neg'/'pos' na 0/1.
    Zwraca pd.Series.
    """
    return y_series.map({'neg': 0, 'pos': 1})


def get_cols_with_missing_threshold(df, threshold=0.5):
    """
    Zwraca listę kolumn, które mają więcej braków danych niż określony próg (domyślnie 50%).
    Obliczenia powinny być wykonywane na zbiorze treningowym.
    """
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    return cols_to_drop


def get_imputation_pipeline(strategy='median'):
    """
    Tworzy i zwraca obiekt Pipeline ze Scikit-learn do uzupełniania pozostałych braków.
    Używamy SimpleImputer(domyślna strategia to median).
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy=strategy))
    ])

def get_scaling_pipeline():
    """
    Standaryzacja danych (StandardScaler).
    """
    return Pipeline([
        ('scaler', StandardScaler())
    ])


def check_split_integrity(X_train, X_test, y_train, y_test):
    """
    """
    print(f"Kształt X_train: {X_train.shape}")
    print(f"Kształt X_test:  {X_test.shape}")

    # Sprawdzenie czy mamy te same kolumny w obu zbiorach
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