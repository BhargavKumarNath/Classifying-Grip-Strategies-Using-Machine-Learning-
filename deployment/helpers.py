import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- DATA LOADING AND CACHING ---
@st.cache_data
def load_data(dataset_name):
    """Loads a specified dataset and caches it."""
    path = f'data/02_processed/{dataset_name}_master_dataset.csv'
    try:
        df = pd.read_csv(path)
        for col in ['Gender', 'Dominant.Eye']:
            if col in df.columns:
                df[col] = df[col].str.lower()
        if 'DOB' in df.columns:
            df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
            df['Age'] = datetime.now().year - df['DOB'].dt.year
        return df
    except FileNotFoundError:
        st.error(f"Error: The file for '{dataset_name}' was not found at {path}. Make sure the path is correct.")
        return None

# EDA PLOTTING FUNCTIONS 
@st.cache_data
def get_subject_df(_df):
    """Creates a subject-level dataframe to avoid double counting."""
    subject_vars = ['subjName', 'Gender', 'Age', 'Dominant.Eye']
    subject_vars = [v for v in subject_vars if v in _df.columns]
    return _df[subject_vars].drop_duplicates(subset=['subjName'])

@st.cache_resource
def plot_distributions(df, dataset_name):
    """Generates distribution plots for categorical and subject variables."""
    df_subjects = get_subject_df(df)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"Distribution of Variables ({dataset_name.title()})", fontsize=20)
    
    # Define plots based on dataset
    if dataset_name == 'aiming':
        sns.countplot(ax=axes[0, 0], x="visCond", data=df, hue="visCond", palette="viridis", legend=False)
        sns.countplot(ax=axes[0, 1], x="surface", data=df, hue="surface", palette="magma", legend=False)
        sns.countplot(ax=axes[1, 0], x="distance", data=df, hue="distance", palette="cividis", legend=False, order=["one", "two", "three"])
    elif dataset_name == 'prehension':
        sns.countplot(ax=axes[0, 0], x='visCond', data=df, palette='viridis')
        sns.countplot(ax=axes[0, 1], x='surface', data=df, palette='magma')
        sns.countplot(ax=axes[1, 0], x='distance', data=df, order=['near', 'middle', 'far'], palette='cividis')
    elif dataset_name == 'visual_illusions':
        sns.countplot(ax=axes[0, 0], x='visCond', data=df, palette='viridis')
        sns.countplot(ax=axes[0, 1], x='illusion', data=df, palette='magma')
        sns.countplot(ax=axes[1, 0], x='targetPos', data=df, palette='cividis')

    axes[0,0].set_title("Visual Condition")
    axes[0,1].set_title("Surface/Illusion Type")
    axes[1,0].set_title("Distance/Target Position")

    # Subject-level plots
    if not df_subjects.empty:
        sns.countplot(ax=axes[1, 1], x="Gender", data=df_subjects, hue="Gender", palette="plasma", legend=False)
        sns.histplot(ax=axes[2, 0], data=df_subjects, x="Age", bins=15, kde=True, color="teal")
        sns.countplot(ax=axes[2, 1], x="Dominant.Eye", data=df_subjects, hue="Dominant.Eye", palette="plasma", legend=False)
    
    axes[1,1].set_title("Gender Distribution")
    axes[2,0].set_title("Age Distribution")
    axes[2,1].set_title("Dominant Eye")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

@st.cache_resource
def plot_kinematic_distributions(_df, kinematic_vars):
    """Plots histograms and boxplots for key kinematic variables."""
    fig, axes = plt.subplots(len(kinematic_vars), 2, figsize=(16, 4 * len(kinematic_vars)))
    fig.suptitle("Distribution of Key Kinematic Variables", fontsize=20, y=1.0)
    
    for i, var in enumerate(kinematic_vars):
        if var in _df.columns:
            sns.histplot(ax=axes[i, 0], data=_df, x=var, kde=True, bins=40, color=sns.color_palette("mako", len(kinematic_vars))[i])
            axes[i, 0].set_title(f"Distribution of {var}")
            sns.boxplot(ax=axes[i, 1], data=_df, x=var, color=sns.color_palette("mako", len(kinematic_vars))[i])
            axes[i, 1].set_title(f"Boxplot of {var}")

    plt.tight_layout()
    return fig

@st.cache_resource
def plot_correlation_heatmap(_df, kinematic_vars):
    """Plots a correlation heatmap for the given kinematic variables."""
    if 'distance' in _df.columns:
        corr_vars = kinematic_vars + ['distance_ordinal']
        if 'distance_ordinal' not in _df.columns:
            dist_map = {v: i+1 for i, v in enumerate(_df['distance'].astype('category').cat.categories)}
            _df['distance_ordinal'] = _df['distance'].map(dist_map)
    else:
        corr_vars = kinematic_vars
        
    corr_matrix = _df[corr_vars].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title("Correlation Matrix of Key Kinematic Variables")
    return fig

@st.cache_resource
def plot_pca_scatter(_df, features_for_pca):
    """Performs PCA and plots the first two components colored by condition."""
    X = _df[features_for_pca]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    
    # Determine columns for coloring
    hue_cols = [col for col in ['distance', 'visCond', 'surface', 'illusion', 'targetPos'] if col in _df.columns]
    viz_df = pd.concat([pca_df, _df[hue_cols].reset_index(drop=True)], axis=1)

    fig, axes = plt.subplots(1, len(hue_cols), figsize=(8 * len(hue_cols), 7), sharex=True, sharey=True)
    if len(hue_cols) == 1: axes = [axes] 
    fig.suptitle("PCA of Kinematic Data: PC1 vs. PC2", fontsize=20)
    
    for i, hue_col in enumerate(hue_cols):
        sns.scatterplot(ax=axes[i], x="PC1", y="PC2", data=viz_df, hue=hue_col, palette=f"Set{i+1}", alpha=0.7, s=50)
        axes[i].set_title(f"Colored by {hue_col.title()}")
        axes[i].grid(True)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# MACHINE LEARNING FUNCTIONS 
@st.cache_data
def get_ml_inputs(_df, kinematic_features, target_col):
    """Prepares data for machine learning models."""
    X = _df[kinematic_features].copy()
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())
    
    y = _df[target_col]
    groups = _df['subjName']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, groups

@st.cache_resource
def run_clustering_analysis(_X_scaled):
    """Runs K-Means for a range of k and returns SSE and silhouette scores."""
    sse = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(_X_scaled)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(_X_scaled, kmeans.labels_))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Elbow Plot
    ax1.plot(k_range, sse, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Sum of Squared Errors (SSE)')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    # Silhouette Plot
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score Analysis')
    ax2.grid(True)
    
    plt.suptitle("Cluster Optimization Metrics")
    return fig

@st.cache_resource
def plot_clusters(_X_scaled, n_clusters):
    """Performs PCA, K-Means clustering and plots the results."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(_X_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(_X_scaled)
    
    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_plot['Cluster'] = labels
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.8, s=60, ax=ax)
    ax.set_title(f'K-Means Clusters (k={n_clusters}) on PCA Components')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(title='Discovered Strategy')
    ax.grid(True)
    
    return fig, labels

@st.cache_resource
def run_supervised_model(_X_scaled, _y, _groups, feature_names):
    """Runs RF with GroupKFold and returns accuracy and feature importances."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(_y)
    
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    group_kfold = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(pipeline, _X_scaled, y_encoded, cv=group_kfold, groups=_groups)
    
    # Fit on all data to get feature importances
    pipeline.fit(_X_scaled, y_encoded)
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='rocket', ax=ax)
    ax.set_title('Top 20 Most Important Features for Prediction')
    plt.tight_layout()

    return np.mean(cv_scores), fig