import streamlit as st

st.set_page_config(
    page_title="Grip Strategy Classification",
    page_icon="👋",
    layout="wide"
)

# --- Title and Introduction ---
st.title("Classifying Grip Strategies using Machine Learning")
st.markdown("### A Dissertation Project by [Your Name]")
st.markdown("---")

# --- Background Section ---
st.header("Project Background and Aims")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    The analysis of human grasping behavior offers a profound window into how 3D objects are internally represented in the brain. 
    By analyzing kinematic data from motion capture systems, we can classify distinct grasping strategies and investigate how humans 
    plan and execute reaching-to-grasp movements based on an object’s properties like shape and orientation.
    
    This project presents a comprehensive machine learning pipeline designed to explore, classify, and interpret these complex motor behaviors across three distinct experimental datasets.
    """)
    
    st.subheader("Primary Aims:")
    st.markdown("""
    - **Develop a reproducible machine learning pipeline** to classify grip strategies from raw kinematic features.
    - **Explore dimensionality reduction techniques** (e.g., PCA) to identify key patterns and simplify complex, high-dimensional data.
    - **Investigate unsupervised clustering methods** to discover and group similar grip strategies without prior labels.
    - **Apply and refine supervised learning algorithms** to predict experimental conditions from movement data.
    - **Synthesize findings across datasets** to build a holistic understanding of human motor control and individual variability.
    """)

with col2:
    st.image("results/figures/grand_summary_visualization.png", 
             caption="Summary of the project's narrative: from group-level patterns to individual differences.",
             use_column_width=True)

st.info("👈 **Navigate through the project phases using the sidebar** to explore the full analysis pipeline.")