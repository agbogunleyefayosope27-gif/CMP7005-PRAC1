import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data,
    load_results,
    load_model,
    load_feature_info,
    prepare_model_dataset,
    get_feature_importance
)

st.set_page_config(page_title="Model Outputs", page_icon="🤖", layout="wide")
sns.set_theme(style="whitegrid")

st.title("Model Outputs Section")

results = load_results()
model = load_model()
feature_info = load_feature_info()
df = load_data()

if results.empty:
    st.error("model_results.csv was not found. Please save the model results from the notebook first.")
    st.stop()

if model is None:
    st.error("best_pm25_model.joblib was not found. Please save the trained model from the notebook first.")
    st.stop()

if not feature_info:
    st.error("model_feature_info.json was not found. Please save the feature info from the notebook first.")
    st.stop()

results_sorted = results.sort_values("RMSE").reset_index(drop=True)
best_row = results_sorted.iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Model", str(best_row["Model"]))
c2.metric("MAE", f"{best_row['MAE']:.2f}")
c3.metric("RMSE", f"{best_row['RMSE']:.2f}")
c4.metric("R²", f"{best_row['R2']:.3f}")

st.subheader("Model Comparison")
st.dataframe(results_sorted, use_container_width=True)

with st.expander("Features Used in the Final Modelling Pipeline", expanded=False):
    st.markdown(f"**Target Variable:** `{feature_info['target']}`")

    left, right = st.columns(2)

    with left:
        st.markdown("**Numeric Features**")
        st.dataframe(
            pd.DataFrame({"Numeric Features": feature_info["numeric_features"]}),
            use_container_width=True
        )

    with right:
        st.markdown("**Categorical Features**")
        st.dataframe(
            pd.DataFrame({"Categorical Features": feature_info["categorical_features"]}),
            use_container_width=True
        )

model_df = prepare_model_dataset(df)

target = feature_info["target"]
numeric_features = feature_info["numeric_features"]
categorical_features = feature_info["categorical_features"]

X = model_df[numeric_features + categorical_features]
y = model_df[target]

split_idx = int(len(model_df) * 0.8)

X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]
test_meta = model_df.iloc[split_idx:][["datetime", "station"]].reset_index(drop=True)

pred = model.predict(X_test)

comparison_df = pd.DataFrame({
    "datetime": test_meta["datetime"],
    "station": test_meta["station"],
    "actual": y_test.reset_index(drop=True),
    "predicted": pred
})

comparison_df["residual"] = comparison_df["actual"] - comparison_df["predicted"]

st.subheader("Prediction Diagnostics")

station_options = sorted(comparison_df["station"].dropna().unique().tolist())
selected_stations = st.multiselect(
    "Filter diagnostics by station",
    options=station_options,
    default=station_options
)

filtered_cmp = comparison_df[comparison_df["station"].isin(selected_stations)].copy()

if filtered_cmp.empty:
    st.warning("No prediction data are available for the selected station filter.")
    st.stop()

max_points = min(len(filtered_cmp), 2000)
default_points = min(500, max_points)

n_points = st.slider(
    "Number of time points to display in the line chart",
    min_value=100,
    max_value=max_points,
    value=default_points,
    step=50
)

plot_df = filtered_cmp.iloc[:n_points].copy()

tab1, tab2, tab3 = st.tabs([
    "Actual vs Predicted Over Time",
    "Predicted vs Actual Scatter",
    "Residual Analysis"
])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["datetime"], plot_df["actual"], label="Actual", linewidth=1.5)
    ax.plot(plot_df["datetime"], plot_df["predicted"], label="Predicted", linewidth=1.5)
    ax.set_title("Actual vs Predicted Next-Hour PM2.5")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("PM2.5")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=filtered_cmp.sample(min(5000, len(filtered_cmp)), random_state=42),
        x="actual",
        y="predicted",
        hue="station",
        alpha=0.5,
        ax=ax
    )

    lower = min(filtered_cmp["actual"].min(), filtered_cmp["predicted"].min())
    upper = max(filtered_cmp["actual"].max(), filtered_cmp["predicted"].max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2)

    ax.set_title("Predicted vs Actual PM2.5")
    ax.set_xlabel("Actual PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=filtered_cmp.sample(min(5000, len(filtered_cmp)), random_state=42),
        x="predicted",
        y="residual",
        hue="station",
        alpha=0.5,
        ax=ax
    )
    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted PM2.5")
    ax.set_ylabel("Residual")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Feature Importance")

importance_df = get_feature_importance(model, numeric_features, categorical_features)

if importance_df.empty:
    st.info("Feature importance is not available for this model type.")
else:
    top_n = st.slider(
        "Number of top features to display",
        min_value=5,
        max_value=min(20, len(importance_df)),
        value=min(10, len(importance_df)),
        step=1
    )

    top_importance = (
        importance_df.head(top_n)
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_importance["Feature"], top_importance["Importance"])
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        importance_df.head(top_n).reset_index(drop=True),
        use_container_width=True
    )

st.subheader("Download Prediction Results")
csv = filtered_cmp.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download prediction diagnostics CSV",
    data=csv,
    file_name="model_prediction_outputs.csv",
    mime="text/csv"
)