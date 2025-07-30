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
import os
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import GripTransformerClassifier, CNNTransformerClassifier


# DATA LOADING AND CACHING 
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
            df['DOB'] = pd.to_datetime(df['DOB'], format="%d/%m/%Y", errors='coerce')
            df['Age'] = datetime.now().year - df['DOB'].dt.year
        return df
    except FileNotFoundError:
        st.error(f"Error: The file for '{dataset_name}' was not found at {path}. Make sure the path is correct.")
        return None

# OUTLIER REMOVAL FUNCTION
@st.cache_data
def remove_outliers_iqr(_df, columns, multiplier=1.5):
    """Remove outliers using IQR method for specified columns."""
    df_clean = _df.copy()
    initial_rows = len(df_clean)
    
    outliers_info = {}
    
    for column in columns:
        if column in df_clean.columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Count outliers before removal
            outliers_count = len(df_clean[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)])
            outliers_info[column] = {
                'count': outliers_count,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows
    
    return df_clean, outliers_info, removed_rows

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
def plot_kinematic_distributions(_df, kinematic_vars, remove_outliers=False, outlier_multiplier=1.5):
    """Plots histograms and boxplots for key kinematic variables with optional outlier removal."""
    
    df_plot = _df.copy()
    outliers_info = {}
    removed_rows = 0
    
    if remove_outliers:
        df_plot, outliers_info, removed_rows = remove_outliers_iqr(df_plot, kinematic_vars, outlier_multiplier)
    
    fig, axes = plt.subplots(len(kinematic_vars), 2, figsize=(16, 4 * len(kinematic_vars)))
    
    # Add outlier removal info to title
    title = "Distribution of Key Kinematic Variables"
    if remove_outliers:
        title += f" (After Outlier Removal - {removed_rows} rows removed)"
    
    fig.suptitle(title, fontsize=20, y=1.0)
   
    for i, var in enumerate(kinematic_vars):
        if var in df_plot.columns:
            # Histogram
            sns.histplot(ax=axes[i, 0], data=df_plot, x=var, kde=True, bins=40, 
                        color=sns.color_palette("mako", len(kinematic_vars))[i])
            axes[i, 0].set_title(f"Distribution of {var}")
            
            # Add outlier info to histogram title if outliers were removed
            if remove_outliers and var in outliers_info:
                outlier_count = outliers_info[var]['count']
                axes[i, 0].set_title(f"Distribution of {var} ({outlier_count} outliers removed)")
            
            # Boxplot
            sns.boxplot(ax=axes[i, 1], data=df_plot, x=var, 
                       color=sns.color_palette("mako", len(kinematic_vars))[i])
            axes[i, 1].set_title(f"Boxplot of {var}")

    plt.tight_layout()
    return fig, df_plot, outliers_info

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
def get_ml_data(_df, kinematic_features):
    """
    The ONE function to prepare data for ALL ML tasks.
    It selects features, cleans rows with missing values, and scales the data.
    Returns the cleaned DataFrame and the scaled numpy array.
    """
    # Select only the features we need for modeling.
    df_model = _df[kinematic_features].copy()
    
    # --- The critical cleaning step ---
    initial_rows = len(df_model)
    df_model.dropna(inplace=True)
    final_rows = len(df_model)
    
    if initial_rows != final_rows:
        # Use st.session_state to show the warning only once per data load
        if 'last_warning_count' not in st.session_state or st.session_state.last_warning_count != (initial_rows - final_rows):
            st.warning(f"Note: {initial_rows - final_rows} rows were removed due to missing kinematic data.")
            st.session_state.last_warning_count = initial_rows - final_rows

    # Scale the cleaned features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)
    
    return df_model, X_scaled


@st.cache_resource
def run_clustering_analysis(_X_scaled):
    """Runs K-Means for a range of k and returns optimization plots."""
    sse = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(_X_scaled)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(_X_scaled, kmeans.labels_))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    ax1.plot(k_range, sse, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Sum of Squared Errors (SSE)')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score Analysis')
    ax2.grid(True)
    
    plt.suptitle("Cluster Optimization Metrics")
    return fig


# @st.cache_resource
def plot_clusters(_X_scaled, n_clusters):
    """
    Performs K-Means clustering and plots the results on PCA components.
    This function now ONLY clusters and plots, it does not re-process data.
    """
    # ... function body remains the same ...
    # Perform PCA just for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(_X_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    # Fit on the original scaled data, not the PCA-reduced data
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



# @st.cache_resource
def run_supervised_model(_X_scaled, _y, _groups, feature_names):
    """Runs RF with GroupKFold and returns accuracy and feature importances."""
    # ... function body remains the same ...
    le = LabelEncoder()
    y_encoded = le.fit_transform(_y)
    
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    group_kfold = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(pipeline, _X_scaled, y_encoded, cv=group_kfold, groups=_groups)
    
    pipeline.fit(_X_scaled, y_encoded)
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(20), x='importance', y='feature', palette='rocket', ax=ax)
    ax.set_title('Top 20 Most Important Features for Prediction')
    plt.tight_layout()

    return np.mean(cv_scores), fig

# --- DEEP LEARNING HELPERS ---

@st.cache_resource
def load_trained_model(run_folder_path, model_type):
    """Loads a pre trained model and its configuration from a run folder"""
    config_path = os.path.join(run_folder_path, "config.yaml")
    model_path = os.path.join(run_folder_path, "best_model.pth")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        st.error(f"Error: `config.yaml` or `best_model.pth` not found in {run_folder_path}")
        return None, None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_params = config["model"]

    # use dummy device for loading
    device = torch.device("cpu")

    if model_type == "Transformer":
        transformer_params = config["transformer"]
        model = GripTransformerClassifier(
            input_features = model_params["input_features"],
            num_classes = model_params["num_classes"],
            d_model = transformer_params["d_model"],
            nhead = transformer_params["nhead"],
            num_encoder_layers = transformer_params["num_encoder_layers"],
            dim_feedforward = transformer_params["dim_feedforward"],
            dropout = transformer_params["dropout"],
            seq_length = model_params["sequence_length"]
        )
    elif model_type == "CNNTransformer":
        cnn_transformer_params = config["cnn_transformer"]
        model = CNNTransformerClassifier(
            input_features = model_params["input_features"],
            num_classes = model_params["num_classes"],
            seq_length = model_params["sequence_length"],
            cnn_out_channels = cnn_transformer_params["cnn_out_channels"],
            d_model = cnn_transformer_params["d_model"],
            nhead = cnn_transformer_params["nhead"],
            num_encoder_layers = cnn_transformer_params["num_encoder_layers"],
            dim_feedforward = cnn_transformer_params["dim_feedforward"],
            dropout = cnn_transformer_params["dropout"]
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config

@st.cache_data
def load_dl_test_data(processed_data_path, raw_df_path):
    """Loads the test data and the label to name mapping"""
    X_test = np.load(os.path.join(processed_data_path, "X_test.npy"))
    y_test = np.load(os.path.join(processed_data_path, "y_test.npy"))

    try:
        raw_df = pd.read_csv(raw_df_path)
    except Exception as e:
        st.error(f"Could not read raw CSV to create label map: {e}")
        return None, None, None
    
    # Clean column names just in case
    raw_df.columns = [col.strip() for col in raw_df.columns]
    label_map = raw_df[['label_encoded', 'label']].drop_duplicates().sort_values('label_encoded').set_index('label_encoded')['label']
    class_names = label_map.tolist()
    
    return X_test, y_test, class_names

@st.cache_resource
def plot_training_history(_metrics_df):
    """Plots the loss and F1-score from the training metrics CSV."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Model Training History", fontsize=18)

    # Plotting Loss
    ax1.plot(_metrics_df['epoch'], _metrics_df['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(_metrics_df['epoch'], _metrics_df['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plotting F1-Score
    ax2.plot(_metrics_df['epoch'], _metrics_df['train_f1'], 'bo-', label='Training F1-Score (Macro)')
    ax2.plot(_metrics_df['epoch'], _metrics_df['val_f1'], 'ro-', label='Validation F1-Score (Macro)')
    ax2.set_title('F1-Score vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1-Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# @st.cache_data
def get_model_predictions(_model, _X_test, _y_test, batch_size=32):
    """Gets predictions for the entire test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model.to(device)
    
    test_dataset = TensorDataset(torch.from_numpy(_X_test).float(), torch.from_numpy(_y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = _model(features)
            preds = torch.argmax(outputs, dim=1)
            y_pred_list.extend(preds.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())
            
    return y_true_list, y_pred_list

def plot_attention_map(model, X_test_sample, true_label_name, class_names):
    """
    Generates the dual-axis attention plot and a dynamic insight string.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    single_sequence = torch.from_numpy(X_test_sample).float().unsqueeze(0).to(device)
    logits, attention_weights = model(single_sequence, return_attention=True)
    pred_label_idx = torch.argmax(logits, dim=1).item()
    pred_label_name = class_names[pred_label_idx]

    cls_attention = attention_weights[0, 0, 1:].detach().cpu().numpy()
    
    # --- DYNAMIC INSIGHT GENERATION ---
    peak_attention_timestep = np.argmax(cls_attention)
    peak_attention_value = np.max(cls_attention)
    
    # Define phases of the movement
    if peak_attention_timestep < 150:
        phase = "the initial planning phase"
    elif peak_attention_timestep > 400 and X_test_sample[peak_attention_timestep, 0] == 0:
        phase = "the zero-padded end of the sequence"
    else:
        phase = "the main execution phase"
        
    dynamic_insight = (
        f"For this specific sample, the model focused most of its attention at **timestep {peak_attention_timestep}** "
        f"(with a weight of {peak_attention_value:.4f}). This falls within **{phase}** of the movement."
    )
    # --- END DYNAMIC INSIGHT ---
    
    fig, ax1 = plt.subplots(figsize=(18, 6))
    timesteps = np.arange(X_test_sample.shape[0])
    
    ax1.plot(timesteps, X_test_sample[:, 0], color='black', label='Feature 1 (e.g., Index X)', alpha=0.6)
    ax1.plot(timesteps, X_test_sample[:, 1], color='gray', label='Feature 2 (e.g., Index Y)', alpha=0.5, linestyle='--')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Scaled Kinematic Value', color='black')
    ax1.legend(loc='upper left')
    ax1.grid(True, axis='x')

    ax2 = ax1.twinx()
    ax2.plot(timesteps, cls_attention, color='orange', linewidth=2.5, label='CLS Token Attention')
    ax2.fill_between(timesteps, 0, cls_attention, color='orange', alpha=0.2) # Add a fill for emphasis
    ax2.set_ylabel('Attention Weight', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0)
    ax2.legend(loc='upper right')

    plt.title(f'Attention Visualization\nTrue Label: {true_label_name} | Predicted: {pred_label_name}', fontsize=16)
    fig.tight_layout()
    
    return fig, dynamic_insight 