import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def print_metrics(y_true, y_pred, y_proba):
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_true, y_proba))

def plot_confusion_matrix(y_true, y_pred, model_name, normalize=False):

    if normalize:
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        filename = f"{model_name}_normalized_confusion_matrix.png"
        title = f"Normalised Confusion Matrix\nfor {model_name} Model"
        fmt = ".2f"
    else:
        cm = confusion_matrix(y_true, y_pred)
        filename = f"{model_name}_confusion_matrix.png"
        title = f"Confusion Matrix for {model_name} Model"
        fmt = ",.0f"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(f"evaluation_plots/{filename}", bbox_inches='tight')
    plt.close()

    return cm


def plot_cost_matrix(cm, model_name, cost_fp=10, cost_fn=500):
    tn, fp, fn, tp = cm.ravel()
    cost_matrix = [
        [0, fp * cost_fp],
        [fn * cost_fn, 0]
    ]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cost_matrix,
        annot=True,
        fmt=",.0f",
        cmap="Reds",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Cost-Weighted Confusion Matrix")

    filename = f"{model_name}_cost_matrix.png"
    plt.savefig(f"evaluation_plots/{filename}", bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    filename = f"{model_name}_roc_curve.png"
    plt.savefig(f"evaluation_plots/{filename}", bbox_inches='tight')
    plt.close()


def find_optimal_threshold(
    y_true,
    y_proba,
    cost_fp=10,
    cost_fn=500,
    thresholds=np.linspace(0.01, 0.99, 100)
):
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = cost_fn * fn + cost_fp * fp

        results.append({
            "threshold": t,
            "fn": fn,
            "fp": fp,
            "tp": tp,
            "tn": tn,
            "cost": total_cost
        })

    df = pd.DataFrame(results)
    best_row = df.loc[df["cost"].idxmin()]

    return best_row, df

def plot_cost_vs_threshold(df, best_threshold, model_name):
    plt.figure(figsize=(7, 5))
    plt.plot(df["threshold"], df["cost"])
    plt.axvline(best_threshold, linestyle="--",
                label=f"Optimal threshold = {best_threshold:.2f}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Total Cost")
    plt.title("Cost vs Threshold")
    plt.legend()
    plt.grid(True)

    filename = f"{model_name}_cost_vs_threshold.png"
    plt.savefig(f"evaluation_plots/{filename}", bbox_inches='tight')
    plt.close()


def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n===== {model_name} =====")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print_metrics(y_test, y_pred, y_proba)

    cm = plot_confusion_matrix(y_test, y_pred, model_name=model_name)
    plot_confusion_matrix(y_test, y_pred, normalize=True, model_name=model_name)
    
    plot_cost_matrix(cm, model_name)

    plot_roc_curve(y_test, y_proba, model_name)

    best_row, df_costs = find_optimal_threshold(y_test, y_proba)
    print(f"Optimal threshold: {best_row['threshold']:.3f}")
    print(f"Minimal cost: {best_row['cost']:,}")

    plot_cost_vs_threshold(df_costs, best_row["threshold"], model_name)

    y_pred_new = (y_proba >= best_row["threshold"]).astype(int)
    plot_confusion_matrix(y_test, y_pred_new, normalize=True, model_name=model_name+" Adjusted")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

rf = joblib.load("rf_aps_model.joblib")
evaluate_model(rf, X_test, y_test, model_name="Random Forest")

lr = joblib.load("lr_aps_model.joblib")
evaluate_model(lr, X_test, y_test, model_name="Logistic Regression")

xg = joblib.load("xgb_aps_tuned_model.joblib")
evaluate_model(xg, X_test, y_test, model_name="Xgboost")