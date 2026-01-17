import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA


def drop_low_variance_features(df, threshold=0.01):
    """Identifies and drops features with a variance below the threshold.
    
    Args:
        df (pd.DataFrame): Input features.
        threshold (float): Correlation threshold (0.0 to 1.0). Defaults to 0.01.

    Returns: 
        pd.DataFrame: DataFrame with redundant features removed.
    """
    selector= VarianceThreshold(threshold=threshold)
    selector.fit(df)
    return df[df.columns[selector.get_support()]]

def drop_high_correlated_features(df, threshold=0.9):
    """Identifies and drops features with a correlation coefficient above the threshold.
    
    Args:
        df (pd.DataFrame): Input features.
        threshold (float): Correlation threshold (0.0 to 1.0). Defaults to 0.9.

    Returns:
        pd.DataFrame: DataFrame with redundant features removed.
    """
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=cols_to_drop)

def select_top_features(X, y, k=50, method='f_classif'):
    return 0