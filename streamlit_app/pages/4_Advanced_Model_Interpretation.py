import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
from utils import load_data, get_pca_transformed_data, get_curated_features
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Advanced Model Interpretation", layout="wide")
st.title("🧠 Advanced Model Interpretation")
st.markdown("Beyond accuracy scores, we need to understand *what* our models have learned. This section dives deeper into interpreting both the unsupervised and supervised models.")

# --- Load Data ---
datasets = load_data()

tab1, tab2 = st.tabs(["**Cluster 'Personas'**", "**Explaining Predictions with SHAP**"])

# ==============================================================================
# TAB 1: CLUSTER PERSONAS
# ==============================================================================
with tab1:
    st.header("Profiling Strategies with Kinematic 'Personas'")
    st.markdown("""
    Clustering gives us group labels, but what do they *mean*? We create 'personas' for each strategy by averaging key kinematic variables. This provides an intuitive, qualitative description of each movement pattern.
    
    We use **k=2** for this analysis to identify the two most dominant, opposing strategies.
    """)
    
    dataset_choice_persona = st.selectbox(
        "**Select a dataset for persona analysis:**",
        options=list(datasets.keys()),
        key="persona_select"
    )
    df_persona = datasets[dataset_choice_persona].copy()

    # Define persona features for each dataset
    if dataset_choice_persona == 'Prehension':
        persona_features = ['movTime', 'MVel', 'MAcc', 'MDec', 'pathLength', 'MGA']
    elif dataset_choice_persona == 'Aiming':
        persona_features = ['movTime', 'MVel_y', 'MAcc', 'MDec', 'pathLength', 'FGA']
        df_persona.rename(columns={"MVel_y": "MVel", "MGA_y": "MGA"}, inplace=True)
    else: # Visual Illusions
        persona_features = ['movTime', 'MVel', 'MAcc', 'MDec', 'pathLength', 'MGA']

    @st.cache_data
    def create_personas(_df, features, k=2):
        X_pca, _, _, _ = get_pca_transformed_data(_df, features=[f for f in features if f in _df.columns])
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        _df['strategy'] = kmeans.fit_predict(X_pca)
        personas = _df.groupby('strategy')[features].mean()
        
        scaler = MinMaxScaler()
        personas_scaled = pd.DataFrame(scaler.fit_transform(personas),
                                       index=personas.index,
                                       columns=personas.columns)
        return personas, personas_scaled

    # Filter persona_features to only include columns that exist in the dataframe
    persona_features_exist = [f for f in persona_features if f in df_persona.columns]
    personas_raw, personas_scaled = create_personas(df_persona, persona_features_exist)

    # Plotting Radar Chart
    st.subheader(f"Kinematic Personas for {dataset_choice_persona} (k=2)")
    
    labels = personas_scaled.columns
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Create a smaller, more compact radar chart
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    for i, row in personas_scaled.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Strategy {i+1}', linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)  # Smaller font for labels
    plt.title(f'Kinematic Personas ({dataset_choice_persona})', size=14, color='grey', y=1.05)  # Smaller title and closer to plot
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=8)  # Smaller legend
    plt.tight_layout()  # Optimize layout
    st.pyplot(fig, use_container_width=False)  # Don't force full container width
    
    with st.expander("Show Raw Mean Values"):
        st.dataframe(personas_raw.round(3))


# ==============================================================================
# TAB 2: SHAP ANALYSIS
# ==============================================================================
with tab2:
    st.header("Explaining Individual Predictions with SHAP")
    st.markdown("""
    For the **Visual Illusions** task, we go a step further. We use an **XGBoost** model (which performed slightly better than Random Forest) and apply **SHAP (SHapley Additive exPlanations)**. 
    
    SHAP is a powerful technique that breaks down any single prediction and shows how much each feature contributed to pushing the model's output from a baseline value to its final prediction. This allows us to understand *why* the model made a specific decision for a given trial.
    """)

    df_illusions = datasets['Visual Illusions'].copy()
    features_illusions = get_curated_features(df_illusions, drop_cols=['targetSize'])

    # Prepare data
    X = df_illusions[features_illusions]
    y = df_illusions['illusion']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    @st.cache_resource
    def train_and_explain_model(_X, _y):
        # For binary classification, XGBoost is simpler
        pipeline_xgb = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'))
        ])
        pipeline_xgb.fit(_X, _y)
        
        # Explain model
        scaler = pipeline_xgb.named_steps['scaler']
        model = pipeline_xgb.named_steps['classifier']
        X_scaled = scaler.transform(_X)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        return pipeline_xgb, explainer, shap_values, X_scaled, le

    pipeline_xgb, explainer, shap_values, X_scaled, le = train_and_explain_model(X, y_encoded)

    st.subheader("Global Feature Importance (SHAP Summary)")
    st.markdown("This plot summarizes the impact of every feature for every sample. Each point is a single trial. Red indicates a high feature value, blue a low one. The position on the x-axis shows its impact on the prediction.")
    
    fig_summary, ax_summary = plt.subplots(figsize=(8, 6))  # Smaller figure size
    # <<< FIX: For binary classification, `class_names` is not a valid argument for summary_plot.
    # The plot title will indicate the positive class.
    shap.summary_plot(shap_values, X, show=False, max_display=10)  # Limit features displayed
    plt.xlabel(f"SHAP value (impact on prediction towards '{le.classes_[1]}')", fontsize=10)
    plt.title("SHAP Summary Plot", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig_summary, use_container_width=False)
    
    st.subheader("Feature Dependence Plots")
    st.markdown("These plots show how a single feature's value affects its SHAP value, and we can color it by an interacting feature.")
    
    top_feature = 'FGOt'
    fig_dep, ax_dep = plt.subplots(figsize=(6, 4))  # Smaller figure size
    # <<< FIX: For binary classification, shap_values is 2D. We don't need to index the class.
    shap.dependence_plot(top_feature, shap_values, X, interaction_index=None, ax=ax_dep, show=False)
    ax_dep.set_ylabel(f"SHAP value for '{le.classes_[1]}'", fontsize=10)
    ax_dep.set_xlabel(f"{top_feature}", fontsize=10)
    ax_dep.set_title(f"SHAP Dependence: {top_feature}", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig_dep, use_container_width=False)
    
    st.subheader("Explaining a Single Prediction (Waterfall Plot)")
    st.markdown("This shows how features for a single trial work together to arrive at a final prediction.")
    
    trial_index = st.slider("Select a trial to explain:", 0, len(X)-1, 0)
    
    fig_waterfall, ax_waterfall = plt.subplots(figsize=(8, 5))  # Smaller figure size
    # <<< FIX: For binary classification, explainer.expected_value is a single value, and shap_values[trial_index] is a 1D array.
    explanation = shap.Explanation(
        values=shap_values[trial_index], 
        base_values=explainer.expected_value, 
        data=X.iloc[trial_index].values,
        feature_names=X.columns.tolist()
    )
    shap.waterfall_plot(explanation, max_display=10, show=False)  # Limit features displayed
    plt.tight_layout()
    st.pyplot(fig_waterfall, use_container_width=False)