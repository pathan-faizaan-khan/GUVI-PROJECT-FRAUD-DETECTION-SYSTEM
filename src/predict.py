
import pandas as pd
import joblib

def load_model(path):
    return joblib.load(path)

def predict_single(model, amount, ttype, location, hour):
    df = pd.DataFrame([{
        "Amount": amount,
        "Type": ttype,
        "Location": location,
        "Hour": hour
    }])
    df_encoded = pd.get_dummies(df, columns=["Type", "Location"], drop_first=True)

    # align with training columns
    train_cols = model.feature_names_in_
    for col in train_cols:
        if col not in df_encoded:
            df_encoded[col] = 0
    df_encoded = df_encoded[train_cols]

    prob = model.predict_proba(df_encoded)[0,1]
    pred = int(prob >= 0.3)
    return {"prediction": pred, "probability": prob}
