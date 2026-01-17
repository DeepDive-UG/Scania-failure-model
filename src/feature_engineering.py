import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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


def select_features_mutual_info(X, y, k=50):
    """Filter method: Selects features using Mutual Information (captures non-linear).
    Mutual information measures the dependency between variables and can 
    capture non-linear relationships between features and the target.
    
    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series/np.array): The target vector.
        k (int): The number of top features to select. Defaults to 50.
        
    Returns:
        pd.DataFrame: DataFrame containing only the selected k features.
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)

    selected_cols=X.columns[selector.get_support()]
    df_out=pd.DataFrame(X_new, columns=selected_cols)

    return  df_out


def select_features_lasso(X, y, C=0.01):
    """
    Embedded method: Selects features using Logistic Regression with L1 (Lasso) penalty.
    
    The L1 penalty forces the coefficients of less important features to zero.
    
    Args:
        X (pd.DataFrame): The scaled input feature matrix.
        y (pd.Series/np.array): The target vector.
        C (float): Inverse of regularization strength. Smaller values lead to 
            stronger regularization and fewer selected features. Defaults to 0.01.
            
    Returns:
        pd.DataFrame: DataFrame containing the non-zero coefficient features.
    """
    logistic = LogisticRegression(C=C, penalty="l1", solver="liblinear", random_state=42)
    selector = SelectFromModel(logistic)
    X_new = selector.fit_transform(X, y)
    
    selected_cols = X.columns[selector.get_support()]
    df_out = pd.DataFrame(X_new, columns=selected_cols, index=X.index)
    
    return df_out


def select_features_rfe(X, y, n_features=50):
    """
    Wrapper method: Performs Recursive Feature Elimination using a Random Forest.
    
    RFE works by recursively removing features and building a model on the 
    remaining ones to identify which features contribute most to the accuracy.
    
    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series/np.array): The target vector.
        n_features (int): The number of top features to retain. Defaults to 50.
        
    Returns:
        pd.DataFrame: DataFrame containing the retained n_features.
    """
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=5)
    X_new = selector.fit_transform(X, y)

    selected_cols = X.columns[selector.get_support()]
    df_out = pd.DataFrame(X_new, columns=selected_cols, index=X.index)

    return df_out



def benchmark_selection(estimator, X, y, name):
    """Calculates the cross-validation score for a specific feature set using a given estimator.

    Args:
        estimator (object): The machine learning model to use (e.g., RandomForestClassifier).
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        name (str): The name of the selection method for reporting.

    Returns:
        dict: A dictionary containing metrics for comparison.
    """
    scores = cross_val_score(estimator, X, y, cv=5, scoring='f1') # Using F1 score for Scania imbalance
    return {
        "Method": name,
        "Feature Count": X.shape[1],
        "F1 Score (Mean)": scores.mean(),
        "F1 Score (Std)": scores.std()
    }


def apply_pca(X_train, X_test, n_components=0.95):
    """Applies PCA to reduce dimensionality.

    Args:
        X_train (pd.DataFrame): Scaled training features.
        X_test (pd.DataFrame): Scaled test features.
        n_components (float/int): Amount of variance to retain (0.0 to 1.0) 
            or specific number of components. Defaults to 0.95.

    Returns:
        tuple: (X_train_pca, X_test_pca, pca_model)
            - X_train_pca (pd.DataFrame): Transformed training data.
            - X_test_pca (pd.DataFrame): Transformed test data.
            - pca_model (PCA): The fitted PCA object for further analysis.
    """
    # Initialize and fit PCA on training data only
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    # Transform both sets
    X_train_pca_arr = pca.transform(X_train)
    X_test_pca_arr = pca.transform(X_test)

    # Create column names for the new principal components
    column_names = [f'PC{i+1}' for i in range(X_train_pca_arr.shape[1])]

    # Convert back to DataFrames
    X_train_pca = pd.DataFrame(X_train_pca_arr, columns=column_names, index=X_train.index)
    X_test_pca = pd.DataFrame(X_test_pca_arr, columns=column_names, index=X_test.index)

    return X_train_pca, X_test_pca, pca
