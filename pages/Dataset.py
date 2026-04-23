import streamlit as st
import pandas as pd
from utils import load_data, apply_filters

st.set_page_config(page_title="Dataset", page_icon="📂", layout="wide")

st.title("Dataset Section")

df = load_data()

min_date = df["datetime"].dt.date.min()
max_date = df["datetime"].dt.date.max()
station_options = sorted(df["station"].dropna().unique().tolist())

st.sidebar.header("Dataset Filters")
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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{filtered_df.shape[0]:,}")
col2.metric("Columns", filtered_df.shape[1])
col3.metric("Stations", filtered_df['station'].nunique())
col4.metric("Date Range", f"{start_date} to {end_date}")

st.subheader("Filtered Dataset Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)

st.subheader("Column Data Types")
dtype_df = pd.DataFrame({
    "Column": filtered_df.columns,
    "Data Type": filtered_df.dtypes.astype(str).values
})
st.dataframe(dtype_df, use_container_width=True)

st.subheader("Missing Values")
missing_df = pd.DataFrame({
    "Column": filtered_df.columns,
    "Missing Count": filtered_df.isna().sum().values,
    "Missing Percent": (filtered_df.isna().mean() * 100).round(2).values
}).sort_values("Missing Count", ascending=False)
st.dataframe(missing_df, use_container_width=True)

st.subheader("Descriptive Statistics")
st.dataframe(filtered_df.describe(include="all").T, use_container_width=True)

st.subheader("Download Filtered Dataset")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered CSV",
    data=csv,
    file_name="filtered_beijing_air_quality.csv",
    mime="text/csv"
)