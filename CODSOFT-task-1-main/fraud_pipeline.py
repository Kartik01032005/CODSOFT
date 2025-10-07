import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.utils import resample
import joblib

data = pd.read_csv("creditcard.csv")
print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
print(data["Class"].value_counts())

plt.figure(figsize=(6,4))
data["Class"].value_counts().plot(kind="bar", color=["teal", "crimson"])
plt.title("Fraud Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

features = data.drop("Class", axis=1)
target = data["Class"]

scale = StandardScaler()
if set(["Time", "Amount"]).issubset(features.columns):
    features[["Time", "Amount"]] = scale.fit_transform(features[["Time", "Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, stratify=target, random_state=42
)
print(y_train.value_counts())

try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
except:
    train_data = pd.concat([X_train, y_train], axis=1)
    fraud = train_data[train_data["Class"] == 1]
    non_fraud = train_data[train_data["Class"] == 0]
    fraud_balanced = resample(fraud, replace=True, n_samples=len(non_fraud), random_state=42)
    upsampled_data = pd.concat([non_fraud, fraud_balanced])
    X_train_res = upsampled_data.drop("Class", axis=1)
    y_train_res = upsampled_data["Class"]

print(pd.Series(y_train_res).value_counts())

models = {
    "LR": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "RF": RandomForestClassifier(n_estimators=120, class_weight="balanced", n_jobs=-1, random_state=42)
}

model_stats = {}

for label, clf in models.items():
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "PR_AUC": average_precision_score(y_test, y_prob)
    }

    model_stats[label] = {"model": clf, **metrics, "preds": y_pred, "proba": y_prob}
    print(f"\n{label}")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

top_model = max(model_stats, key=lambda m: model_stats[m]["F1"])
chosen_model = model_stats[top_model]["model"]
print(f"\nSelected Model: {top_model}")

fpr, tpr, _ = roc_curve(y_test, model_stats[top_model]["proba"])
prec, rec, _ = precision_recall_curve(y_test, model_stats[top_model]["proba"])

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(fpr, tpr)
plt.title(f"ROC Curve - {top_model}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.subplot(1,2,2)
plt.plot(rec, prec)
plt.title(f"Precision-Recall - {top_model}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()

joblib.dump(chosen_model, f"fraud_detection_{top_model}.joblib")
print(f"Model saved: fraud_detection_{top_model}.joblib")

results_df = X_test.copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = model_stats[top_model]["preds"]
results_df["Fraud_Probability"] = model_stats[top_model]["proba"]

top_cases = results_df.sort_values("Fraud_Probability", ascending=False).head(15)
print(top_cases)
