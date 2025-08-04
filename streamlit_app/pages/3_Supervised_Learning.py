import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from utils import load_data, get_curated_features

st.set_page_config(page_title="Supervised Learning", layout="wide")
st.title("🎯 Supervised Learning: Predicting Conditions")
st.markdown("""
After discovering strategies with unsupervised learning, we now use supervised learning to see if we can **predict the experimental conditions** (e.g., target distance or illusion type) directly from the kinematic data.
""")

# --- Load Data and Select ---
datasets_info = load_data()
dataset_choice = st.selectbox(
    "**Select a dataset to model:**",
    options=list(datasets_info.keys()),
    key="supervised_select"
)

# --- Robust Caching Function ---
@st.cache_data
def perform_supervised_analysis(dataset_name):
    """
    Performs the entire supervised learning pipeline for a given dataset.
    This is cached to prevent re-computation on every widget interaction,
    and it guarantees data consistency.
    """
    df = datasets_info[dataset_name].copy()

    # 1. Define target and features
    target_col = 'distance' if dataset_name != 'Visual Illusions' else 'illusion'
    features_to_drop = ['targetSize'] if dataset_name == 'Visual Illusions' else []
    curated_features = get_curated_features(df, drop_cols=features_to_drop)

    X = df[curated_features]
    y = df[target_col]
    groups = df['subjName']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 2. Define the modeling pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # 3. Perform cross-validation
    group_kfold = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(pipeline, X, y_encoded, cv=group_kfold, groups=groups)
    
    # 4. Fit final model on all data to get feature importances
    pipeline.fit(X, y_encoded)
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    # 5. Create the importance DataFrame INSIDE the cached function
    feature_importance_df = pd.DataFrame({
        'feature': curated_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return cv_scores, feature_importance_df, target_col, len(curated_features)

# --- Main Analysis Script ---
# Call the cached function once
cv_scores, feature_importance_df, target_col, num_features = perform_supervised_analysis(dataset_choice)

# --- The Challenge of Leaky Features ---
st.header("1. The Challenge: Data Leakage")
st.warning("""
**Initial Problem:** A naive model might achieve suspiciously high accuracy. This is often due to **data leakage**, where some features inadvertently contain information about the outcome. For example, the final XYZ coordinates of a movement are perfectly correlated with the target's location!
""")

# --- Feature Curation ---
st.header("2. Refining the Model with Curated Features")
st.markdown("To build a robust and meaningful model, we must remove these 'leaky' features. We create a **curated feature set** that only includes variables describing the *dynamics* of the movement, not its final endpoint.")

with st.expander("Show Feature Curation Details"):
    st.write(f"**Number of curated (non-leaky) features used for the '{dataset_choice}' model:** {num_features}")
    st.write("**Example leaky patterns removed:** `Xmax`, `FX`, `Zloc`, etc.")

# --- Model Training and Evaluation ---
st.header("3. Model Performance with Curated Features")
st.markdown("""
We use a **Random Forest Classifier** with **GroupKFold cross-validation**. This ensures that data from the same subject does not appear in both the training and testing sets of a fold, leading to a more realistic estimate of performance on new, unseen subjects.
""")

st.metric(label=f"Mean Cross-Validation Accuracy (predicting '{target_col}')", value=f"{np.mean(cv_scores):.4f}")
st.write(f"**Accuracy Scores per Fold:** `{[round(s, 4) for s in cv_scores]}`")

# --- Feature Importance ---
st.header("4. Interpreting the Model: Feature Importances")
st.markdown("This plot shows the top 20 most influential features the model used to make its predictions. This tells us which aspects of movement are most critical for differentiating between the experimental conditions.")

fig, ax = plt.subplots(figsize=(12, 10))
# Fix for the seaborn warning
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), 
            hue='feature', palette='rocket', ax=ax, legend=False)
ax.set_title(f'Top 20 Important Features for Predicting {target_col.title()} (Curated)')
ax.set_xlabel('Importance Score')
ax.set_ylabel('Feature')
plt.tight_layout()
st.pyplot(fig)

st.write("**Top 5 Most Important Features:**")
st.dataframe(feature_importance_df.head(5))