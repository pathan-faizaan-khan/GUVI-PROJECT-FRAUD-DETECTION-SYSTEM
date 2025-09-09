
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fraud_distribution(df):
    ax = df["IsFraud"].value_counts().rename({0:"Genuine",1:"Fraud"}).plot(
        kind="bar", color=["green","red"], edgecolor="black"
    )
    plt.title("Fraud vs Genuine Transactions")
    plt.ylabel("Count")
    return ax

def fraud_rate_by_type(df):
    rate = df.groupby("Type")["IsFraud"].mean().sort_values(ascending=False)
    ax = rate.plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Fraud Rate by Transaction Type")
    plt.ylabel("Fraud Rate")
    return ax

def fraud_trend_by_hour(df):
    hourly = df.groupby("Hour")["IsFraud"].mean()
    ax = hourly.plot(kind="line", marker="o", color="purple")
    plt.title("Fraud Trend by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Fraud Rate")
    return ax

def correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    return ax
