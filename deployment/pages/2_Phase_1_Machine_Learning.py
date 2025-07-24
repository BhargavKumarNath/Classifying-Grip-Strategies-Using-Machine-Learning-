import streamlit as st
import pandas as pd
from helpers import(
    load_ml_data, plot_eda_distribution, plot_kinematic_distributions, plot_correlation_heatmap, plot_pca_kmeans_clusters, plot_feature_importance
)

st.set_page_config(page_title="Machine Learning Phase", layout="wide")
st.title("Phase 1: Classical Machine Learning Analysis")

dataset_choice = st.selectbox(
    "Choose a dataset to analyse:",
    ("aiming", "prehension", "visual_illusions"),
    format_func=lambda x: x.replace("_", " ").title()
)
df = load_ml_data(dataset_choice)

tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Unsupervised Learning", "Supervised Learning"])

with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    st.write(f"Showing results for the **{dataset_choice.title()}** dataset")

    st.subheader("1. Data Distributions")
    st.markdown("""
    First, we examine the distributions of categorical variables to understand the experimental design and subject demographics. Then we look at the distributions of key kinematic variables.
""")
    if st.checkbox("Show Distribution Plots"):
        fig_dist = plot_eda_distribution(df)
        st.pyplot(fig_dist)

        fig_kinematic = plot_kinematic_distributions(df)
        st.pyplot(fig_kinematic)
    st.subheader("2. Correlation Analysis by Distance")
    st.markdown("""
    To understand how movement parameters relate to each other, we plot correlation matrices for each target distance. This helps reveal the underlying physics of the movements.
""")
    distance = st.radio("Select a distance for correlation analysis:", df["distance"].unique())
    fig_corr = plot_correlation_heatmap(df, distance)
    st.pyplot(fig_corr)
    st.markdown("""
    **Key Findings:** Across all distances, we see strong correlations like `pathLength` vs. `movTime` (positive) and `PeakVelocity` vs. `movTime` (negative), which validates the quality of data.
""")
    
with tab2:
    st.header("Unsupervised Learning: Discovering Grip Strategies")
    st.markdown("""
    Can we find natural groupings in the data without using the distance labels? we use Principal Component Analysis (PCA) for dimensionality reduction and K-Means clustering to find these groups.
""")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PCA & K-Means Clustering")
        st.markdown("""
        The plot on the right shows the movement trials projected onto the first two principal components. The points are coloured by the cluster assigned by the K-Means algorithm (with k=4, chosen via the elbow method)
        
        **Findings:** The data naturally seperates into three distinct clusters. This is a significatn result, as it strongly suggests that the primary factor driving kinematic differences is the target disstance, confirming our experimental setup.
""")
    with col2:
        fig_pca_kmeans = plot_pca_kmeans_clusters(df)
        st.pyplot(fig_pca_kmeans)

with tab3:
    
