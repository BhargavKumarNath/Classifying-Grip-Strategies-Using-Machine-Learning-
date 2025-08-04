import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from utils import load_data, get_pca_transformed_data

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")
st.title("🔎 Exploratory Data Analysis (EDA)")
st.markdown("""
This page details the initial exploration and cleaning of the three datasets. EDA is a critical first step to understand the data's structure, distributions, and inherent relationships before modeling.
""")

# --- Load Data ---
datasets = load_data()
dataset_choice = st.selectbox(
    "**Select a dataset to explore:**",
    options=list(datasets.keys())
)
df = datasets[dataset_choice].copy()

# --- Initial Inspection ---
st.header("1. Initial Data Inspection")
st.markdown(f"**Dataset Shape:** `{df.shape}`")
st.markdown(f"**Total Missing Values:** `{df.isnull().sum().sum()}`")
st.write("**Data Sample:**")
st.dataframe(df.head())

# --- Data Cleaning ---
with st.expander("Show Data Cleaning Steps", expanded=False):
    st.markdown("""
    - **Standardized Text:** Corrected inconsistent capitalization in `Gender` and `Dominant.Eye`.
    - **Created 'Age' Feature:** Calculated from `DOB`.
    - **Optimized Data Types:** Converted categorical columns to the `category` dtype for efficiency.
    - **Ordinal Encoding:** Converted `distance` categories to numerical values (e.g., 'one' -> 1).
    """)

# --- Categorical Variable Analysis ---
st.header("2. Distribution of Categorical Variables")
st.markdown("These plots show the balance of trials across different experimental conditions.")

categorical_cols = ['visCond', 'surface', 'distance']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(categorical_cols):
    order = sorted(df[col].unique()) if col != 'distance' else ['one', 'two', 'three'] if 'one' in df['distance'].unique() else ['near', 'middle', 'far']
    sns.countplot(ax=axes[i], x=col, data=df, hue=col, palette='viridis', order=order, legend=False)
    axes[i].set_title(f"Distribution of '{col}'")
    axes[i].tick_params(axis='x', rotation=45)
st.pyplot(fig)

# --- Kinematic Analysis ---
st.header("3. Key Kinematic Variable Analysis")
key_kinematic_vars = ['movTime', 'pathLength', 'MVel', 'MAcc', 'MGA', 'MDec']
# Adjust for aiming dataset which has different column names
if dataset_choice == 'Aiming':
    key_kinematic_vars = ['movTime', 'pathLength', 'MVel_y', 'MAcc', 'MGA_y', 'MDec']
    df.rename(columns={"MVel_y": "MVel", "MGA_y": "MGA"}, inplace=True)
    
key_kinematic_vars = [v for v in key_kinematic_vars if v in df.columns]

tab1, tab2 = st.tabs(["**Before Cleaning**", "**After Outlier Removal**"])

with tab1:
    st.subheader("Distributions Before Outlier Removal")
    st.markdown("Initial distributions often show skewness and outliers, which can negatively impact ML models.")
    fig_before, axes_before = plt.subplots(len(key_kinematic_vars), 2, figsize=(14, 20))
    for i, var in enumerate(key_kinematic_vars):
        sns.histplot(ax=axes_before[i, 0], data=df, x=var, kde=True, bins=40)
        axes_before[i, 0].set_title(f"Distribution of {var}")
        sns.boxplot(ax=axes_before[i, 1], data=df, x=var)
        axes_before[i, 1].set_title(f"Boxplot of {var}")
    plt.tight_layout()
    st.pyplot(fig_before)

# --- Outlier Removal Logic ---
@st.cache_data
def clean_data(_df, _vars):
    df_transformed = _df.copy()
    for var in _vars:
        if var in df_transformed.columns:
            pt = PowerTransformer(method='yeo-johnson')
            transformed_data = pt.fit_transform(df_transformed[[var]].dropna())
            df_transformed.loc[df_transformed[[var]].dropna().index, f'{var}_yj'] = transformed_data
    
    df_cleaned = df_transformed.copy()
    for var in _vars:
        transformed_var = f'{var}_yj'
        if transformed_var in df_cleaned.columns:
            Q1, Q3 = df_cleaned[transformed_var].quantile(0.25), df_cleaned[transformed_var].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[transformed_var] >= lower) & (df_cleaned[transformed_var] <= upper)]
    return df_cleaned

df_cleaned = clean_data(df, key_kinematic_vars)

with tab2:
    st.subheader("Distributions After Outlier Removal")
    st.markdown(f"After applying a Yeo-Johnson transformation and removing outliers via the IQR method, the distributions are more normalized. **{len(df) - len(df_cleaned)} rows removed ({((len(df) - len(df_cleaned))/len(df))*100:.2f}%).**")
    fig_after, axes_after = plt.subplots(len(key_kinematic_vars), 2, figsize=(14, 20))
    for i, var in enumerate(key_kinematic_vars):
        sns.histplot(ax=axes_after[i, 0], data=df_cleaned, x=var, kde=True, bins=30, color='green')
        axes_after[i, 0].set_title(f"Distribution of {var}")
        sns.boxplot(ax=axes_after[i, 1], data=df_cleaned, x=var, color='green')
        axes_after[i, 1].set_title(f"Boxplot of {var}")
    plt.tight_layout()
    st.pyplot(fig_after)


# --- Correlation and ANOVA ---
st.header("4. Relationship Analysis")
tab_corr, tab_anova = st.tabs(["**Correlation Matrix**", "**ANOVA**"])

with tab_corr:
    st.subheader("Correlation of Key Kinematic Variables")
    if 'distance_ordinal' not in df_cleaned.columns:
        dist_map = {'one': 1, 'two': 2, 'three': 3} if 'one' in df['distance'].unique() else {'near': 1, 'middle': 2, 'far': 3}
        df_cleaned['distance_ordinal'] = df_cleaned['distance'].map(dist_map)
    
    corr_vars = key_kinematic_vars + ["distance_ordinal"]
    corr_matrix = df_cleaned[corr_vars].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title("Correlation Matrix of Key Kinematic Variables")
    st.pyplot(fig_corr)

with tab_anova:
    st.subheader("Impact of Experimental Conditions on Kinematics (ANOVA)")
    st.markdown("Analysis of Variance (ANOVA) tests whether the mean of a kinematic variable is significantly different across experimental conditions.")
    plot_vars = key_kinematic_vars[:4]
    cond = st.radio("Analyze by condition:", ['distance', 'visCond', 'surface'], horizontal=True)

    fig_anova, axes_anova = plt.subplots(1, 4, figsize=(24, 6))
    for i, var in enumerate(plot_vars):
        order = sorted(df_cleaned[cond].unique()) if cond != 'distance' else ['one', 'two', 'three'] if 'one' in df_cleaned['distance'].unique() else ['near', 'middle', 'far']
        sns.violinplot(ax=axes_anova[i], x=cond, y=var, data=df_cleaned, hue=cond, palette="muted", order=order, legend=False)
        axes_anova[i].set_title(var)
        
        # ANOVA test
        groups = [df_cleaned[var][df_cleaned[cond] == c].dropna() for c in order]
        if len(groups) > 1:
            p_val = stats.f_oneway(*groups)[1]
            axes_anova[i].text(0.5, 0.95, f"p-value: {p_val:.3e}", ha="center", transform=axes_anova[i].transAxes, bbox=dict(facecolor='white', alpha=0.8))
    fig_anova.suptitle(f"Kinematic Variables by {cond}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig_anova)


# --- PCA Exploration ---
st.header("5. Dimensionality Reduction with PCA")
st.markdown("Principal Component Analysis (PCA) is used to reduce the high-dimensional kinematic data into a few 'principal components' that capture most of the variance. This helps in visualizing complex data.")

# We use the cleaned data for PCA
X_pca, pca, _, _ = get_pca_transformed_data(df_cleaned)

col_pca1, col_pca2 = st.columns(2)

with col_pca1:
    st.subheader("Explained Variance")
    fig_var, ax_var = plt.subplots(figsize=(10, 6))
    ax_var.plot(np.cumsum(pca.explained_variance_ratio_), marker='.')
    ax_var.set_xlabel("Number of Components")
    ax_var.set_ylabel("Cumulative Explained Variance")
    ax_var.set_title("PCA - Explained Variance by Components")
    ax_var.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax_var.grid(True)
    st.pyplot(fig_var)

with col_pca2:
    st.subheader("PCA: PC1 vs PC2")
    pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    pca_df = pd.concat([pca_df, df_cleaned.reset_index()], axis=1)
    
    color_by = st.radio("Color PCA plot by:", ['distance', 'visCond', 'surface'], horizontal=True, key="pca_color")
    
    fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
    sns.scatterplot(ax=ax_pca, x="PC1", y="PC2", data=pca_df, hue=color_by, palette="viridis", alpha=0.7)
    ax_pca.set_title(f"PCA of Kinematic Data, colored by {color_by}")
    st.pyplot(fig_pca)