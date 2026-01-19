import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from joblib import dump
from src.data.preprocess import (
    load_aps_data,
    encode_target,
    get_cols_with_missing_threshold,
    get_imputation_pipeline,
    get_scaling_pipeline,
    check_split_integrity
)
from src.feature_engineering import (
    select_features_lasso,
    select_features_mutual_info,
    select_features_rfe,
    benchmark_selection
)

df = load_aps_data("data/raw/aps_failure_training_set.csv")

X = df.drop(columns="class")
y = encode_target(df["class"])

'''80/20 Train Test Split 
and integrity check
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
check_split_integrity(X_train, X_test, y_train, y_test)

'''Dropping columns with many missing values'''
cols_to_drop = get_cols_with_missing_threshold(X_train)
X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

feature_cols = X_train.columns 

'''Imputation'''
imputer = get_imputation_pipeline()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

'''Scaling'''
scaler = get_scaling_pipeline()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=feature_cols,
    index=y_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=feature_cols,
    index=y_test.index
)

'''Testing best feature selection
for Logistic Regression'''
X_mi = select_features_mutual_info(X_train_scaled, y_train)
X_lasso = select_features_lasso(X_train_scaled, y_train)
X_rfe = select_features_rfe(X_train_scaled, y_train)

results = []
lr_model = LogisticRegression(class_weight="balanced", max_iter=2000)

results.append(benchmark_selection(lr_model, X_train_scaled, y_train, "All features"))
results.append(benchmark_selection(lr_model, X_mi, y_train, "Mutual Info"))
results.append(benchmark_selection(lr_model, X_lasso, y_train, "LASSO"))
results.append(benchmark_selection(lr_model, X_rfe, y_train, "RFE"))

results_df = pd.DataFrame(results)
print(results_df)

best_method = results_df.sort_values(
    "F1 Score (Mean)", ascending= False
).iloc[0]["Method"]

print("Best method: ", best_method)
'''Feature Scaled
with best method
only data used in Logistic Regression'''
if best_method == "Mutual Info":
    X_train_lr = X_mi
elif best_method == "LASSO":
    X_train_lr = X_lasso
elif best_method == "RFE":
    X_train_lr = X_rfe
else:
    X_train_lr = X_train_scaled

X_test_lr = X_test_scaled[X_train_lr.columns]


'''Logistic Regression
handled class imbalance with class weigths 
saved model with joblib'''
lr_model.fit(X_train_lr, y_train)
dump(lr_model,"lr_aps_model.joblib")

y_pred_lr = lr_model.predict(X_test_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

X_train_rf = pd.DataFrame(
    X_train,
    columns=feature_cols,
    index=y_train.index
)

X_test_rf = pd.DataFrame(
    X_test,
    columns=feature_cols,
    index=y_test.index
)

'''Random Forest Classifier
with handling class imbalance with class weigths
saved model with joblib'''
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced")
rf_model.fit(X_train_rf, y_train)
dump((cols_to_drop, imputer, rf_model),"rf_aps_model.joblib")

y_pred_rf = rf_model.predict(X_test_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("Confusion Matrix - Logistic Regression")
print(cm_lr)

print("Confusion Matrix - Random Forest")
print(cm_rf)