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
dump(lr_model,"lr_aps_model.joblib")

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
dump((rf_model),"rf_aps_model.joblib")

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
