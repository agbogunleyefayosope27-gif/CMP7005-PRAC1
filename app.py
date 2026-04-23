import streamlit as st

st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Beijing Air Quality Analysis Dashboard")
st.markdown(
    """
This interactive application presents the outputs from the Beijing air-quality analysis project.

### Sections
- **Dataset**: inspect, filter, summarise, and download the cleaned dataset
- **Visualisations**: explore key trends, comparisons, and relationships in the data
- **Model Outputs**: review model performance, diagnostics, and feature importance

Use the navigation menu on the left to move between sections.
"""
)

st.info("This app is based on the cleaned dataset and modelling outputs produced in Tasks 1 to 3.")

st.subheader("Project Workflow")
st.markdown(
    """
1. Data selection and merging  
2. Data preprocessing and cleaning  
3. Exploratory analysis and visualisation  
4. Model building and evaluation  
5. Interactive application development  
"""
)

st.success("The application now includes Dataset, Visualisations, and Model Outputs pages.")