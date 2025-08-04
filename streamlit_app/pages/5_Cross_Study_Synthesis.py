import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import IsolationForest
from utils import load_data, get_curated_features
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Cross-Study Synthesis", layout="wide")
st.title("🌐 Cross-Study Synthesis & Novel Analyses")
st.markdown("In this section, we move beyond analyzing each dataset in isolation. We combine them to uncover higher-level principles and apply novel analytical techniques.")

# --- Load Data ---
datasets = load_data()
df_prehension = datasets['Prehension'].copy()
df_aiming = datasets['Aiming'].copy()
df_illusions = datasets['Visual Illusions'].copy()


tab1, tab2, tab3 = st.tabs([
    "**1. Task Fingerprinting**", 
    "**2. Illusion Susceptibility**", 
    "**3. Anomaly Detection**"
])

# ==============================================================================
# TAB 1: TASK FINGERPRINTING
# ==============================================================================
with tab1:
    st.header("Task Fingerprinting: What Makes a Movement Unique?")
    st.markdown("""
    Can we identify the task (Prehension, Aiming, or Illusion) just by looking at the movement kinematics? We combine all three datasets and train a model to distinguish between them. The most important features for this model represent the "kinematic fingerprint" of each task.
    """)

    @st.cache_data
    def combine_and_model_tasks(_df_pre, _df_aim, _df_ill):
        _df_pre['task_type'] = 'prehension'
        _df_aim['task_type'] = 'aiming'
        _df_ill['task_type'] = 'illusion'
        
        common_cols = list(set(_df_pre.columns) & set(_df_aim.columns) & set(_df_ill.columns))
        # Ensure task_type and subjName are always included for modeling
        if 'task_type' not in common_cols: common_cols.append('task_type')
        if 'subjName' not in common_cols: common_cols.append('subjName')
            
        df_combined = pd.concat([
            _df_pre[common_cols], _df_aim[common_cols], _df_ill[common_cols]
        ], ignore_index=True)
        
        features = df_combined.select_dtypes(include=np.number).columns.tolist()
        features = [col for col in features if col not in ['subjName', 'trialN']]
        
        X = df_combined[features]
        # <<< FIX: The target 'y' should be the 'task_type' column, not all features.
        y = df_combined['task_type']
        groups = df_combined['subjName']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
        ])
        
        try:
            group_kfold = GroupKFold(n_splits=5)
            cv_scores = cross_val_score(pipeline, X, y_encoded, cv=group_kfold, groups=groups)
        except ValueError:
            # Fallback for when subjects are not shared across all tasks, which is the case here.
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X, y_encoded, cv=skf)

        pipeline.fit(X, y_encoded)
        importances = pipeline.named_steps['classifier'].feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': features, 'importance': importances
        }).sort_values('importance', ascending=False)
        
        return cv_scores, feature_importance_df

    cv_scores_task, feature_importance_df_task = combine_and_model_tasks(df_prehension, df_aiming, df_illusions)

    st.metric("Mean CV Accuracy for Task Prediction", f"{np.mean(cv_scores_task):.4f}")
    
    st.subheader("Top 20 Features for Identifying Task Type")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df_task.head(20), hue='feature', palette='viridis', ax=ax, legend=False)
    ax.set_title('Top 20 Features for Identifying Task Type (Task Fingerprint)')
    plt.tight_layout()
    st.pyplot(fig)


# ==============================================================================
# TAB 2: ILLUSION SUSCEPTIBILITY
# ==============================================================================
with tab2:
    st.header("Quantifying Individual Susceptibility to Illusions")
    st.markdown("""
    Not everyone is equally fooled by visual illusions. Can we quantify this "susceptibility" from their movements?
    
    **Methodology:**
    1. Train a model to predict the illusion type ('Ponzo' vs. 'Ebbinghaus').
    2. For each trial, calculate the model's prediction probability. An "ambiguous" trial has a probability near 50%.
    3. A subject's average ambiguity across all trials is calculated.
    4. **Susceptibility Score = 1 - (Average Ambiguity)**. A high score means the subject's movements were consistently and clearly classifiable, indicating they were more strongly affected by the illusions.
    """)
    
    @st.cache_data
    def calculate_susceptibility(_df_ill):
        features = get_curated_features(_df_ill, drop_cols=['targetSize'])
        X = _df_ill[features]
        y = _df_ill['illusion']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        # Find index of the 'Ponzo' class to track its probability
        try:
            ponzo_idx = list(le.classes_).index('Ponzo')
        except ValueError:
            # Fallback if 'Ponzo' isn't in this slice of data for some reason
            st.warning("Could not find 'Ponzo' class.")
            return pd.DataFrame() # Return empty df
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))
        ])
        pipeline.fit(X, y_encoded)
        
        probs = pipeline.predict_proba(X)
        _df_ill['P(Ponzo)'] = probs[:, ponzo_idx]
        _df_ill['ambiguity_score'] = 1 - np.abs(_df_ill['P(Ponzo)'] - 0.5) * 2
        
        subject_scores = _df_ill.groupby('subjName')['ambiguity_score'].mean().reset_index()
        subject_scores['susceptibility_score'] = 1 - subject_scores['ambiguity_score']
        return subject_scores.sort_values('susceptibility_score', ascending=False)

    subject_scores = calculate_susceptibility(df_illusions)

    st.subheader("Kinematic Susceptibility by Subject")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(x='subjName', y='susceptibility_score', data=subject_scores, order=subject_scores['subjName'], hue='subjName', palette='viridis', ax=ax, legend=False)
    ax.set_title('Kinematic Susceptibility to Visual Illusions by Subject', fontsize=16)
    ax.set_ylabel('Susceptibility Score (1 = Highly Susceptible)')
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    
    st.subheader("Kinematic Deep Dive: What Drives Susceptibility?")
    st.markdown("Comparing the movements of the most vs. least susceptible subjects reveals the source of the difference. Here, we plot the distribution of **Final Grip Orientation (FGOt)**.")
    
    n_compare = 3
    if len(subject_scores) > n_compare * 2:
        least_susceptible = subject_scores.tail(n_compare)['subjName'].tolist()
        most_susceptible = subject_scores.head(n_compare)['subjName'].tolist()
        
        df_least = df_illusions[df_illusions['subjName'].isin(least_susceptible)]
        df_most = df_illusions[df_illusions['subjName'].isin(most_susceptible)]

        fig_kde, ax_kde = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        sns.kdeplot(data=df_most, x='FGOt', hue='illusion', fill=True, common_norm=False, ax=ax_kde[0], palette='magma')
        ax_kde[0].set_title(f'MOST Susceptible Subjects (N={n_compare})')
        sns.kdeplot(data=df_least, x='FGOt', hue='illusion', fill=True, common_norm=False, ax=ax_kde[1], palette='magma')
        ax_kde[1].set_title(f'LEAST Susceptible Subjects (N={n_compare})')
        fig_kde.suptitle("Comparison of Kinematic Response by Susceptibility Group", fontsize=16)
        st.pyplot(fig_kde)
        st.success("**Observation:** The most susceptible subjects show a clear separation in their FGOt distributions for the two illusions, while the least susceptible subjects have highly overlapping distributions. Their movements are less differentiated.")
    else:
        st.warning("Not enough subjects to perform the susceptibility comparison.")

# ==============================================================================
# TAB 3: ANOMALY DETECTION
# ==============================================================================
with tab3:
    st.header("Identifying Anomalous Trials")
    st.markdown("""
    Using an **Isolation Forest** model, we can identify trials that are statistically unusual compared to the rest. This can help find erroneous recordings or genuinely unique, outlier movement strategies.
    
    We apply this to the **Visual Illusions** dataset.
    """)

    @st.cache_data
    def detect_anomalies(_df):
        features = get_curated_features(_df, drop_cols=['targetSize'])
        X = _df[features]
        X_scaled = StandardScaler().fit_transform(X)
        
        iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        _df['anomaly_label'] = iso_forest.fit_predict(X_scaled)
        return _df

    df_illusions_anomalies = detect_anomalies(df_illusions)
    anomalies = df_illusions_anomalies[df_illusions_anomalies['anomaly_label'] == -1]
    inliers = df_illusions_anomalies[df_illusions_anomalies['anomaly_label'] == 1]
    
    st.metric("Anomalous Trials Detected", f"{len(anomalies)}")

    st.subheader("Profile of Anomalous Trials vs. Inliers")
    comparison_features = ['movTime', 'MVel', 'MAcc', 'pathLength', 'MGA']
    # Ensure columns exist before trying to access them
    comparison_features = [f for f in comparison_features if f in anomalies.columns and f in inliers.columns]
    
    anomaly_profile = pd.DataFrame({
        'Anomalies': anomalies[comparison_features].mean(),
        'Inliers': inliers[comparison_features].mean()
    })
    st.dataframe(anomaly_profile.round(3))
    st.info("**Observation:** On average, anomalous trials tend to be slower, with longer path lengths and higher grip apertures.")