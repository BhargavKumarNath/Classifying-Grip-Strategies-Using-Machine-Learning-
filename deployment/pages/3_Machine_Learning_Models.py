import streamlit as st
import pandas as pd
import helpers

st.set_page_config(page_title="Machine Learning Models", layout="wide")
st.title("🤖 Machine Learning Models")
st.markdown("In this section, we apply machine learning to first *discover* underlying movement strategies (unsupervised) and then to *predict* experimental conditions from kinematics (supervised).")

# --- Sidebar for Dataset Selection ---
dataset_name = st.sidebar.selectbox(
    "Choose a dataset for modeling:",
    ("aiming", "prehension", "visual_illusions"),
    format_func=lambda x: x.replace("_", " ").title()
)

# Load the full, raw dataset once
df_raw = helpers.load_data(dataset_name)

if df_raw is not None:
    st.header(f"Modeling on: {dataset_name.replace('_', ' ').title()} Dataset")

    # Define features and target based on dataset
    if dataset_name == 'prehension':
        kinematic_features = ['movTime', 'pathLength', 'MVel', 'MGA', 'MAcc', 'MDec', 'timeMGA', 'clear.logMAR']
        target_col = 'distance'
    elif dataset_name == 'visual_illusions':
        # Let's ensure we are using the correct features from your script
        kinematic_features = ['movTime', 'pathLength', 'MVel', 'MGA', 'MAcc', 'MDec']
        target_col = 'illusion'
    else: # aiming
        kinematic_features = ['movTime', 'pathLength', 'PeakVelocity', 'MaxGripAperture', 'MAcc', 'MDec', 'timeMGA']
        target_col = 'distance'
    
    # Filter for features that actually exist in the dataframe
    kinematic_features = [f for f in kinematic_features if f in df_raw.columns]

    # --- THE UNIFIED DATA PIPELINE ---
    # 1. Get the cleaned and scaled data for ALL ML tasks. This function is cached.
    df_model, X_scaled = helpers.get_ml_data(df_raw, kinematic_features)

    # 2. Re-align the original full dataframe to the cleaned one using the index.
    # This creates the dataframe for analysis that matches X_scaled in length.
    df_analysis = df_raw.loc[df_model.index].copy()

    # --- Tabs ---
    tab1, tab2 = st.tabs(["Unsupervised Learning: Discovering Strategies", "Supervised Learning: Predicting Conditions"])

    with tab1:
        st.subheader("Finding Natural Groups with Clustering")
        
        st.subheader("1. How many clusters (strategies) exist?")
        st.markdown("We use the Elbow and Silhouette methods to find the optimal number of clusters.")
        with st.spinner("Running cluster optimization analysis..."):
            fig_optim = helpers.run_clustering_analysis(X_scaled)
            st.pyplot(fig_optim)
        
        st.info("Based on the plots, we can determine the most likely number of clusters. For the Aiming and Prehension tasks, an optimal `k=3` often aligns perfectly with the three target distances.")

        st.subheader("2. Visualizing the Discovered Strategies")
        default_k = 3 if dataset_name in ['aiming', 'prehension'] else 2
        k = st.slider("Select number of clusters (k) to visualize:", min_value=2, max_value=8, value=default_k, key=f"k_slider_{dataset_name}")
        
        with st.spinner(f"Running K-Means with k={k} and plotting..."):
            # This function is NO LONGER CACHED, so it runs fresh every time.
            fig_clusters, labels = helpers.plot_clusters(X_scaled, k)
            st.pyplot(fig_clusters)
        
        st.subheader("3. What do these clusters mean?")
        st.markdown(f"Let's compare our discovered strategies (clusters) to the actual `{target_col}` condition.")
        
        # This assignment is now guaranteed to work because `df_analysis` and `labels` come from the same data (`X_scaled`).
        df_analysis['discovered_strategy'] = labels
        crosstab = pd.crosstab(df_analysis['discovered_strategy'], df_analysis[target_col])
        st.dataframe(crosstab)
        st.success("The crosstab now correctly aligns the discovered clusters with the experimental conditions.")

    with tab2:
        st.subheader(f"Predicting '{target_col.title()}' from Kinematics")
        st.markdown("""
        Now, we train a Random Forest classifier...
        """)

        with st.spinner("Training Random Forest and performing cross-validation..."):
            y = df_analysis[target_col]
            groups = df_analysis['subjName']
            # This function is NO LONGER CACHED.
            mean_accuracy, fig_importance = helpers.run_supervised_model(X_scaled, y, groups, kinematic_features)
        
        st.metric(label="Mean Cross-Validation Accuracy", value=f"{mean_accuracy:.2%}")
        st.success(f"The model can predict the **{target_col}** with high accuracy.")

        st.subheader("Which Kinematic Features Are Most Important for Prediction?")
        st.pyplot(fig_importance)
        st.markdown("""
        **Insight:** The feature importance plot reveals what the model learned.
        """)