import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, apply_filters

st.set_page_config(page_title="Visualisations", page_icon="📊", layout="wide")

sns.set_theme(style="whitegrid")

st.title("Visualisation Section")

df = load_data()

min_date = df["datetime"].dt.date.min()
max_date = df["datetime"].dt.date.max()
station_options = sorted(df["station"].dropna().unique().tolist())

st.sidebar.header("Visualisation Filters")
selected_stations = st.sidebar.multiselect(
    "Select station(s)",
    options=station_options,
    default=station_options
)

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_df = apply_filters(df, selected_stations, start_date, end_date)

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

chart_option = st.selectbox(
    "Choose a visualisation",
    [
        "Mean pollutant concentrations by station",
        "Monthly PM2.5 trend by station",
        "Seasonal mean PM2.5 by station",
        "Weekday vs weekend PM2.5",
        "Correlation heatmap",
        "PM2.5 distribution",
        "Pollutant boxplots by station",
        "Hourly PM2.5 pattern by station",
        "PM2.5 vs temperature",
        "NO2 vs O3"
    ]
)

if chart_option == "Mean pollutant concentrations by station":
    station_pollution = (
        filtered_df.groupby("station", observed=True)[["PM2.5", "PM10", "NO2", "O3"]]
        .mean()
        .reset_index()
    )
    station_pollution_melted = station_pollution.melt(
        id_vars="station",
        var_name="Pollutant",
        value_name="Mean"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=station_pollution_melted, x="station", y="Mean", hue="Pollutant", ax=ax)
    ax.set_title("Mean Pollutant Concentrations by Station")
    ax.set_xlabel("Station")
    ax.set_ylabel("Mean Concentration")
    plt.xticks(rotation=20)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)

    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "Monthly PM2.5 trend by station":
    monthly_pm25 = (
        filtered_df.groupby(["station", "year", "month"], observed=True)["PM2.5"]
        .mean()
        .reset_index()
    )
    monthly_pm25["year_month"] = pd.to_datetime(monthly_pm25[["year", "month"]].assign(day=1))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly_pm25, x="year_month", y="PM2.5", hue="station", ax=ax)
    ax.set_title("Monthly Average PM2.5 by Station")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average PM2.5")
    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "Seasonal mean PM2.5 by station":
    seasonal_pm25 = (
        filtered_df.groupby(["season", "station"], observed=True)["PM2.5"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=seasonal_pm25, x="season", y="PM2.5", hue="station", ax=ax)
    ax.set_title("Seasonal Mean PM2.5 by Station")
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean PM2.5")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)

    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "Weekday vs weekend PM2.5":
    weekend_pm25 = (
        filtered_df.groupby(["station", "is_weekend"], observed=True)["PM2.5"]
        .mean()
        .reset_index()
    )
    weekend_pm25["day_type"] = weekend_pm25["is_weekend"].map({0: "Weekday", 1: "Weekend"})

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=weekend_pm25, x="station", y="PM2.5", hue="day_type", ax=ax)
    ax.set_title("Weekday vs Weekend PM2.5 by Station")
    ax.set_xlabel("Station")
    ax.set_ylabel("Average PM2.5")
    plt.xticks(rotation=20)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)

    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "Correlation heatmap":
    numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    corr = filtered_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "PM2.5 distribution":
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=filtered_df, x="PM2.5", hue="station", bins=40, kde=True, ax=ax, element="step")
    ax.set_title("PM2.5 Distribution by Station")
    ax.set_xlabel("PM2.5")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "Pollutant boxplots by station":
    pollutant_choice = st.selectbox(
        "Select pollutant for boxplot",
        ["PM2.5", "PM10", "NO2", "O3"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=filtered_df, x="station", y=pollutant_choice, ax=ax)
    ax.set_title(f"{pollutant_choice} by Station")
    ax.set_xlabel("Station")
    ax.set_ylabel(pollutant_choice)
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "Hourly PM2.5 pattern by station":
    hourly_pm25 = (
        filtered_df.groupby(["station", "hour"], observed=True)["PM2.5"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=hourly_pm25, x="hour", y="PM2.5", hue="station", marker="o", ax=ax)
    ax.set_title("Average Hourly PM2.5 Pattern by Station")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average PM2.5")
    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "PM2.5 vs temperature":
    sample_df = filtered_df.sample(min(5000, len(filtered_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=sample_df, x="TEMP", y="PM2.5", hue="station", alpha=0.5, ax=ax)
    ax.set_title("PM2.5 vs Temperature")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("PM2.5")
    plt.tight_layout()
    st.pyplot(fig)

elif chart_option == "NO2 vs O3":
    sample_df = filtered_df.sample(min(5000, len(filtered_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=sample_df, x="NO2", y="O3", hue="station", alpha=0.5, ax=ax)
    ax.set_title("NO2 vs O3")
    ax.set_xlabel("NO2")
    ax.set_ylabel("O3")
    plt.tight_layout()
    st.pyplot(fig)