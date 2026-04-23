from pathlib import Path
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "beijing_selected_stations_cleaned.csv"
RESULTS_PATH = BASE_DIR / "outputs" / "model_results.csv"


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


def apply_filters(df, stations, start_date, end_date):
    filtered = df.copy()

    if stations:
        filtered = filtered[filtered["station"].isin(stations)]

    filtered = filtered[
        (filtered["datetime"].dt.date >= start_date) &
        (filtered["datetime"].dt.date <= end_date)
    ]

    return filtered