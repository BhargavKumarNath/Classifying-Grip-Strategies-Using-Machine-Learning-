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
    """A helper function to display all results for a given model with an improved layout."""
    if not os.path.exists(model_path):
        st.error(f"Could not find the run folder: `{model_path}`. Please check the path.")
        return

    # Load model and metrics
    model, config = helpers.load_trained_model(model_path, model_type)
    metrics_df = pd.read_csv(os.path.join(model_path, 'training_metrics.csv'))

    if model is None:
        return
        
    st.subheader(f"Model Performance: {model_type}")
    
    # Display training history in an expander
    with st.expander("Show Training History (Loss & F1-Score)"):
        fig_history = helpers.plot_training_history(metrics_df)
        st.pyplot(fig_history)

    # Get predictions for the whole test set
    y_true, y_pred = helpers.get_model_predictions(model, X_test, y_test)
    
    st.markdown("---")
    
    st.subheader("Test Set Evaluation")
    
    col1, col2 = st.columns(2) # Create two columns

    with col1:
        # Classification Report
        st.markdown("##### Classification Report")
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        
        # Display key overall metrics at the top
        st.metric("Overall Accuracy", f"{report_dict['accuracy']:.2%}")
        st.metric("Macro Avg F1-Score", f"{report_dict['macro avg']['f1-score']:.4f}")
        
        # Put the detailed report in an expander to save space
        with st.expander("Show Detailed Report"):
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df)
            
    with col2:
        # Confusion Matrix
        st.markdown("##### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig_cm)
    
    st.markdown("---") 
    
    st.subheader("Interpreting the 'Black Box' with Attention")
    st.markdown("Here we visualize what parts of the movement sequence the model paid attention to when making a decision for a specific test sample.")
    
    sample_idx = st.slider("Select a test sample to inspect:", 0, len(X_test) - 1, 0, key=f"slider_{model_type}")
    
    with st.spinner("Generating attention plot..."):
        true_label_name = class_names[y_test[sample_idx]]
        # --- CAPTURE THE DYNAMIC INSIGHT ---
        fig_attn, insight_text = helpers.plot_attention_map(model, X_test[sample_idx], true_label_name, class_names)
        st.pyplot(fig_attn)
        # --- DISPLAY THE DYNAMIC INSIGHT ---
        st.info(insight_text)


# Main Page Layout
tab1, tab2 = st.tabs(["Model 1:The Vanilla Transformer", "Model 2: The CNN Transformer"])

with tab1:
    st.header("First Attempt: A Standard Transformer Classifier")
    display_model_results(VANILLA_TRANSFORMER_PATH, "Transformer")
#     st.error("""
# **Critical Insights & Flaw:** The attention plot for the vanilla Transformer reveals a major problem. The model consistently focuses its attention on the **end of the sequence (timesteps > 400)**, which corresponds to the zero padding.
             
# **It's not learning the movement; it's learning the *length* of the movement!** This is a classic "shortcut" that produces artificially good results on some classes but fails to generalise. This discovery prompted the development of a more robust model.
# """)
    
with tab2:
    st.header("The Solution: A Hybrid CNN-Transformer")
    st.markdown("""
To force the model to learn from the kinematic patterns themselves, we prefixed the Transformer with a 1D Convolutional Neural Network (CNN). The CNN acts as a sophisticated feature extractor, identifying local patterns (like small hesitations or changes in acceleration) across the time-series. These rich, pre processed features are then fed to the Transformer.
""")
    display_model_results(CNN_TRANSFORMER_PATH, "CNNTransformer")
    st.success("""
**The Breakthrough Finding:** The Attention plot for the CNN-Transformer tells a completely different and far more compelling story. The model now focuses its attention almost entirely on the **initial planning of the movement (timesteps 0-150)**.
               
**This is a significant scientific insight.** It suggests that the most distinguishing characteristics of a grip are encoded in the preparatory phase, even before the main reach-to-grasp action begins. The model has learned a plausible and interpretable strategy that is directly relevant to motor control theories.
""")