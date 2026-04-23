from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "beijing_selected_stations_cleaned.csv"
RESULTS_PATH = BASE_DIR / "outputs" / "model_results.csv"
MODEL_PATH = BASE_DIR / "models" / "best_pm25_model.joblib"
FEATURE_INFO_PATH = BASE_DIR / "outputs" / "model_feature_info.json"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_feature_info():
    if FEATURE_INFO_PATH.exists():
        with open(FEATURE_INFO_PATH, "r") as f:
            return json.load(f)
    return {}


def apply_filters(df, stations, start_date, end_date):
    filtered = df.copy()

    if stations:
        filtered = filtered[filtered["station"].isin(stations)]

    filtered = filtered[
        (filtered["datetime"].dt.date >= start_date) &
        (filtered["datetime"].dt.date <= end_date)
    ]

    return filtered


def prepare_model_dataset(df):
    model_df = df.copy()

    # Step 1: sort within station so lag features are created correctly
    model_df = model_df.sort_values(["station", "datetime"]).reset_index(drop=True)

    model_df["PM2.5_next_hour"] = model_df.groupby("station", observed=True)["PM2.5"].shift(-1)

    model_df["PM2.5_lag1"] = model_df.groupby("station", observed=True)["PM2.5"].shift(1)
    model_df["PM2.5_lag3"] = model_df.groupby("station", observed=True)["PM2.5"].shift(3)
    model_df["PM2.5_lag24"] = model_df.groupby("station", observed=True)["PM2.5"].shift(24)

    model_df["NO2_lag1"] = model_df.groupby("station", observed=True)["NO2"].shift(1)
    model_df["O3_lag1"] = model_df.groupby("station", observed=True)["O3"].shift(1)
    model_df["TEMP_lag1"] = model_df.groupby("station", observed=True)["TEMP"].shift(1)
    model_df["WSPM_lag1"] = model_df.groupby("station", observed=True)["WSPM"].shift(1)

    model_df = model_df.dropna().reset_index(drop=True)

    # Step 2: re-sort by time before splitting,
    # so test data includes all stations in the final period
    model_df = model_df.sort_values(["datetime", "station"]).reset_index(drop=True)

    return model_df


def get_feature_importance(model, numeric_features, categorical_features):
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    ohe = (
        model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_feature_names

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": estimator.feature_importances_
    }).sort_values("Importance", ascending=False)

    return importance_df