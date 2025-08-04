import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Use Streamlit's caching to load data efficiently. This is safe as it has no args.
@st.cache_data
def load_data():
    """Loads all three master datasets and returns them as a dictionary."""
    try:
        df_prehension = pd.read_csv("data/02_processed/prehension_master_dataset.csv")
        df_aiming = pd.read_csv("data/02_processed/aiming_master_dataset.csv")
        df_illusions = pd.read_csv("data/02_processed/visual_illusions_master_dataset.csv")
        return {
            "Prehension": df_prehension,
            "Aiming": df_aiming,
            "Visual Illusions": df_illusions
        }
    except FileNotFoundError as e:
        st.error(f"Fatal Error: Dataset not found. Please check the file path: {e}")
        st.stop()

# --- Utility functions are now NOT cached. ---
# They will be called by page-level cached functions, which is a safer pattern.

def get_kinematic_features(df, drop_cols=[]):
    """Selects all numeric columns, excluding identifiers and specified drop_cols."""
    identifiers = ['subjName', 'trialN']
    features = df.select_dtypes(include=np.number).columns.tolist()
    return [col for col in features if col not in identifiers and col not in drop_cols]

def get_curated_features(df, drop_cols=[]):
    """Removes leaky features from the feature list."""
    leaky_patterns = ['Xmax', 'Ymax', 'Zmax', 'FX', 'FY', 'FZ', 'Xloc', 'Yloc', 'Zloc']
    all_kinematic = get_kinematic_features(df, drop_cols)
    curated = [
        feat for feat in all_kinematic
        if not any(pattern in feat for pattern in leaky_patterns)
    ]
    return curated

def get_pca_transformed_data(df, n_components=0.95, features=None):
    """
    Scales and applies PCA to the dataframe.
    This is now a regular, non-cached function.
    """
    if features is None:
        features = get_kinematic_features(df)
    
    X = df[features].copy()
    
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca, X_scaled, scaler