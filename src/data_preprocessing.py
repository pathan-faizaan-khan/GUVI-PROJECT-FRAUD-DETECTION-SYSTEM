
import pandas as pd


def add_features(df):
    df["HighAmount"] = (df["Amount"] > 5000).astype(int)
    df["IsNight"] = ((df["Hour"] >= 22) | (df["Hour"] <= 6)).astype(int)
    return df

def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=["Type", "Location"], drop_first=True)
    return df_encoded
