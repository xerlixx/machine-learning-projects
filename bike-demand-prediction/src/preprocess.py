import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess(path="../data/train.csv"):
    df = pd.read_csv(path)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year.map({2011: 0, 2012: 1})

    y = df["count"]

    numeric_features = ["temp", "atemp", "humidity", "windspeed"]
    categorical_features = [
        "season", "holiday", "workingday", "weather",
        "hour", "dayofweek", "month", "year"
    ]

    X = df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features
