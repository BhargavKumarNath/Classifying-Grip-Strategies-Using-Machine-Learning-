import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import helpers

st.set_page_config(page_title="Deep Learning Phase", layout="wide")
st.title("Phase 2: Deep Learning with Attention")
st.markdown("""
In this phase, we move beyond summarised statistics and analysed the **full time series data** of each movement. This allows us to capture the continuous dunamics and potentially discover more subtle  patterns. We leverage the **Transformer Architecture**, which is state of the art for sequence analysis, and its powerful **attention mechanism** to interpret the model's decisions.
""")

# Paths
VANILLA_TRANSFORMER_PATH = "outputs/GripTransformer_2025-07-15_13-00-54"
CNN_TRANSFORMER_PATH = "outputs/GripTransformer_2025-07-21_13-30-30"
PROCESSED_DATA_PATH = "data/03_processed_dl"
RAW_DATA_PATH = "data/02_processed/GripFormer_preprocessed_dataset.csv"

# Load Data 
X_test, y_test, class_names = helpers.load_dl_test_data(PROCESSED_DATA_PATH, RAW_DATA_PATH)

def display_model_results(model_path, model_type):
    """A helper function to display all results for a given model"""
    if not os.path.exists(model_path):
        st.error(f"Could not find the run folder: `{model_path}`. Please check the path.")
        return
    
    # Load model and metrics
    model, config = helpers.load_trained_model(model_path, model_type)
    metrics_df = pd.read_csv(os.path.join(model_path, "training_metrics.csv"))

    if model is None:
        return 
    
    st.subheader(f"Model Performance: {model_type}")

    # Display train history
    with st.expander("Show Training History (Loss & F1-Score)"):
        fig_history = helpers.plot_training_history(metrics_df)
        st.pyplot(fig_history)

    # Get predictions
    y_true, y_pred = helpers.get_model_predictions(model, X_test, y_test)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.text("Classification Report")
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        st.metric("Overall Accuracy", f"{report_df.loc["accuracy", "precision"]:.2f%}")
    
    with col2:
        st.text("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig_cm)

    
    st.subheader("Inperpreting the 'Black Box' with Attention")
    st.markdown("Here we visualise what parts of the movement sequence the model paid attention to when making a decision for a specific test sample")

    # Interactive sample selection
    sample_idx = st.slider("Select a test sample to inspect:", 0, len(X_test) - 1, 0, key=f"slider_{model_type}")

    with st.spinner("Generating attention plot..."):
        true_labels_name = class_names[y_test[sample_idx]]
        fig_attn = helpers.plot_attention_map(model, X_test[sample_idx], true_labels_name, class_names)
        st.pyplot(fig_attn)

