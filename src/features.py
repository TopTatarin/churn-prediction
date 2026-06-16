"""Cleaning, feature engineering, and train/test split utilities for the Telco churn dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and handle the well-known TotalCharges blank-string issue."""
    df = df.copy()

    # TotalCharges arrives as object dtype because ~11 rows have " " (blank) for
    # customers with tenure == 0 (brand-new customers who haven't been billed yet).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # These NaNs correspond to tenure == 0 customers; TotalCharges should be 0 for them.
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    # Any remaining NaNs (unexpected) get median-imputed as a safety net.
    if df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Operates after clean()."""
    df = df.copy()

    # charges_per_month: guard against tenure == 0 (new customers) -> use MonthlyCharges instead.
    df["charges_per_month"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # Tenure buckets: common churn-analysis grouping.
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"],
    )

    # Count of additional services subscribed (proxy for "stickiness").
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda col: col.map(lambda v: 1 if v not in ("No", "No internet service", "No phone service") else 0)
    ).sum(axis=1)

    return df


def encode_categoricals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode categorical columns.

    OHE chosen over target/ordinal encoding: all categoricals here are low-cardinality
    (2-4 levels) so OHE doesn't blow up dimensionality, and target encoding risks leakage
    on a dataset this small (~7k rows) without careful out-of-fold encoding.
    """
    df = df.copy()
    target = df[TARGET_COL].map({"Yes": 1, "No": 0}) if TARGET_COL in df.columns else None

    drop_cols = [c for c in (ID_COL, TARGET_COL) if c in df.columns]
    features = df.drop(columns=drop_cols)

    cat_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    features = pd.get_dummies(features, columns=cat_cols, drop_first=drop_first)

    if target is not None:
        features[TARGET_COL] = target
    return features


def build_processed_dataset(raw_csv_path: Path) -> pd.DataFrame:
    df = load_raw(raw_csv_path)
    df = clean(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified split on the churn target."""
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
