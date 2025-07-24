import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import numpy as np

@st.cache_data
def load_ml_data(dataset_name):
    path = f"data/02_processed?{dataset_name}_master_dataset.csv"
    return pd.read_csv(path)

def plot_eda_distribution(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    st.write("Distribution of Categorical and Subject related Variables")

    # Example for one plot
    sns.countplot(ax=axes[0, 0], x="visCond", data=df)
    axes[0, 0].set_title("Visual Condition Distribution")

    sns.countplot(ax=axes[0, 1], x="surface", data=df)
    axes[0, 1].set_title("Surface Type")

    plt.tight_layout()
    return fig

def plot_kinematic_distributions(df):
    kinematic_vars = ['movTime', 'pathLength', 'PeakVelocity', 'MaxGripAperture', 'MAcc', 'MDec']
    fig, axes = plt.subplots(len(kinematic_vars), 2, figsize=(12, 20))
    st.write("Distribution of Key Kinematic Variables")

    for i, var in enumerate(kinematic_vars):
        sns.histplot(df[var], kde=True, ax=axes[i, 0])
        sns.boxplot(x=df[var], ax=axes[i, 1])
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, distance):
    df_dist = df[df["distance"] == distance]
    kinematic_features = ['movTime', 'pathLength', 'PeakVelocity', 'timeToPV', 'timeToPD', 'MaxGripAperture', 'timeMGA', 'MAcc', 'MDec']
    corr = df_dist[kinematic_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(f"Correlation Matrix for Distance: {distance}")
    return fig

def plot_pca_kmeans_clusters(df):
    kinematic_features = ['movTime', 'pathLength', 'PeakVelocity', 'timeToPV', 'timeToPD', 'MaxGripAperture', 'timeMGA', 'MAcc', 'MDec']
    X = df[kinematic_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
    ax.set_title("K Means Clusters on PCA Components")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Cluster 0", "Cluster 1", "Cluster 2"])
    return fig

def plot_feature_importance(df):
    kinematic_features = ['movTime', 'pathLength', 'PeakVelocity', 'MaxGripAperture', 'MAcc', 'MDec', 'Zmax_index', 'Zmax_thumb', 'Zmax_wrist']

    X = df[kinematic_features]
    y = df["distance"]
    groups = df["SubjectID"]

    # Simplified training for demonstration
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=kinematic_features).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=importances.values, y=importances.index, ax=ax)
    ax.set_title("Feature Importance for Predicting Target Distance")
    ax.set_xlabel("Importance Score")
    return fig

