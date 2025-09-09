from data_preprocessing import preprocess_data, add_features
from imblearn.over_sampling import SMOTE

import pandas as pd
import joblib, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from data_preprocessing import preprocess_data

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "transactions.csv"
MODEL_PATH = BASE / "models" / "fraud_model_rf.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

def train_model():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)
    df_encoded = preprocess_data(df)
    X = df_encoded.drop("IsFraud", axis=1)
    y = df_encoded["IsFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    joblib.dump(model, MODEL_PATH)

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": report
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    


if __name__ == "__main__":
    train_model()
