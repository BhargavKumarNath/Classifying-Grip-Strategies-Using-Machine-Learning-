import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from utils import load_data, get_pca_transformed_data

st.set_page_config(page_title="Unsupervised Strategy Discovery", layout="wide")
st.title("💡 Unsupervised Strategy Discovery")
st.markdown("""
Here, we use unsupervised machine learning (clustering) to discover natural groupings or "strategies" within the movement data, without relying on predefined labels. The goal is to see if distinct movement patterns emerge on their own.
""")

# --- Data Selection ---
datasets_info = load_data()
dataset_choice = st.selectbox(
    "**Select a dataset to analyze:**",
    options=list(datasets_info.keys()),
    key="unsupervised_select"
)

# --- Robust Caching Function ---
# This function takes the selected dataset NAME as input.
# It performs all heavy computation and is cached. When the name changes,
# it re-runs, guaranteeing all returned data is consistent.
@st.cache_data
def perform_unsupervised_analysis(dataset_name):
    """
    Loads a dataset by name, performs PCA, and calculates clustering metrics.
    This is a cache-safe way to handle data processing for different selections.
    """
    df = datasets_info[dataset_name].copy()
    
    # 1. PCA Transformation
    X_pca, pca, _, _ = get_pca_transformed_data(df)
    
    # 2. Calculate Clustering Metrics
    sse = []
    silhouette_scores = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_pca)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))
    
    return df, X_pca, pca, k_range, sse, silhouette_scores

# --- Main Analysis ---
# Call the cached function. All variables below are now guaranteed to be in sync.
df, X_pca, pca, k_range, sse, silhouette_scores = perform_unsupervised_analysis(dataset_choice)

st.header("1. Data Preparation (Scaling & PCA)")
st.markdown("As in the EDA, we first scale the data and apply PCA. We retain components explaining 95% of the variance to ensure we capture the essential dynamics while reducing noise.")
st.success(f"Data prepared for **{dataset_choice}**. PCA reduced features to **{pca.n_components_}** components, explaining **{np.sum(pca.explained_variance_ratio_)*100:.2f}%** of the variance.")


st.header("2. Finding the Optimal Number of Clusters (k)")
st.markdown("We use two methods to determine the best 'k': the Elbow Method (finding the 'bend' in the sum of squared errors) and the Silhouette Score (measuring cluster separation).")

optimal_k_sil = k_range[np.argmax(silhouette_scores)]

col1, col2 = st.columns(2)
with col1:
    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
    ax_elbow.plot(k_range, sse, marker='o', linestyle='--')
    ax_elbow.set_xlabel('Number of Clusters (k)')
    ax_elbow.set_ylabel('Sum of Squared Errors (SSE)')
    ax_elbow.set_title('Elbow Method for Optimal k')
    ax_elbow.grid(True)
    st.pyplot(fig_elbow)

with col2:
    fig_sil, ax_sil = plt.subplots(figsize=(10, 6))
    ax_sil.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    ax_sil.set_xlabel('Number of Clusters (k)')
    ax_sil.set_ylabel('Silhouette Score')
    ax_sil.set_title('Silhouette Score for Optimal k')
    ax_sil.axvline(x=optimal_k_sil, color='r', linestyle='--', label=f'Optimal k = {optimal_k_sil}')
    ax_sil.legend()
    ax_sil.grid(True)
    st.pyplot(fig_sil)
    
st.info(f"Based on the Silhouette Score, the optimal number of clusters for the **{dataset_choice}** dataset is **k = {optimal_k_sil}**.")


st.header("3. Clustering Results and Interpretation")
k_selected = st.slider("Select number of clusters (k) to visualize:", 2, 10, int(optimal_k_sil))

# Perform clustering. This now works correctly because df and X_pca have matching lengths.
kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init='auto')
df['kmeans_strategy'] = kmeans.fit_predict(X_pca)
hierarchical = AgglomerativeClustering(n_clusters=k_selected, linkage='ward')
df['hierarchical_strategy'] = hierarchical.fit_predict(X_pca)

col_vis1, col_vis2 = st.columns(2)

with col_vis1:
    st.subheader(f"K-Means Clusters (k={k_selected}) on PCA Plot")
    pca_df = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
    pca_df['strategy'] = df['kmeans_strategy']
    
    fig_clusters, ax_clusters = plt.subplots(figsize=(10, 8))
    sns.scatterplot(ax=ax_clusters, x='PC1', y='PC2', hue='strategy', data=pca_df, palette='viridis', alpha=0.8, s=60)
    ax_clusters.set_title(f'K-Means Clusters on PCA Plot (k={k_selected})')
    ax_clusters.set_xlabel('Principal Component 1')
    ax_clusters.set_ylabel('Principal Component 2')
    st.pyplot(fig_clusters)
    
with col_vis2:
    st.subheader("Interpretation & Comparison")
    
    # Check relationship with experimental conditions
    condition_col = 'distance' if 'distance' in df.columns else 'illusion'
    st.write(f"**Relationship with '{condition_col}':**")
    crosstab_cond = pd.crosstab(df['kmeans_strategy'], df[condition_col])
    st.dataframe(crosstab_cond)

    st.write("**Comparison of K-Means vs. Hierarchical Clustering:**")
    crosstab_methods = pd.crosstab(df['kmeans_strategy'], df['hierarchical_strategy'])
    st.dataframe(crosstab_methods)
    
    ari_score = adjusted_rand_score(df['kmeans_strategy'], df['hierarchical_strategy'])
    st.metric(label="Adjusted Rand Index (ARI)", value=f"{ari_score:.4f}")
    st.markdown("*(ARI measures the similarity between two clusterings. 1.0 is a perfect match.)*")