import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from joblib import dump

'''Loading processed datasets'''
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

'''Logistic Regression
handled class imbalance with class weigths 
saved model with joblib'''
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000)
lr_model.fit(X_train, y_train)
dump(lr_model,"src/models/lr_aps_model.joblib")

y_pred_lr = lr_model.predict(X_test)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

'''Random Forest Classifier
with handling class imbalance with class weigths
saved model with joblib'''
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced")
rf_model.fit(X_train, y_train)
dump((rf_model),"src/models/rf_aps_model.joblib")

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

'''Results of classification models'''
print("\nLogistic Regression")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_proba_lr))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf))

'''Advanced modeling: XGBOOST'''
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
'''ratio to help XGBOOST focus on rare failure cases'''
ratio = (y_train == 0).sum() / (y_train == 1).sum()

'''XGBoost with Hyperparameter Tuning'''
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio,
    eval_metric='aucpr',
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

random_search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_grid,
    n_iter=10, # Number of combinations to try
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_xgb = random_search.best_estimator_

dump(best_xgb, "src/models/xgb_aps_tuned_model.joblib")

y_pred_xgb = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

'''Comparison table'''
def calculate_scania_cost(y_true, y_pred):
    """Specific cost metric for Scania: FP=10, FN=500"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fp * 10) + (fn * 500)

comparison_data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost (Tuned)"],
    "ROC AUC": [
        roc_auc_score(y_test, y_proba_lr),
        roc_auc_score(y_test, y_proba_rf),
        roc_auc_score(y_test, y_proba_xgb)
    ],
    "Total Cost ($)": [
        calculate_scania_cost(y_test, y_pred_lr),
        calculate_scania_cost(y_test, y_pred_rf),
        calculate_scania_cost(y_test, y_pred_xgb)
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n--- Final Model Comparison ---")
print(df_comparison.sort_values(by="Total Cost ($)"))

# zapis do pliku
df_comparison.to_csv("reports/model_comparison.csv", index=False)