import streamlit as st
st.set_page_config(page_title="Conclusion", layout="wide")

st.markdown("<h1 style='text-align: center;'>Conclusion & Key Contributions</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #BDBDBD;'>Synthesizing Insights from a Multi-Phase Investigation</h3>", unsafe_allow_html=True)

st.markdown("---")

st.header("Summary of Key Research Findings")
st.markdown("""
This dissertation successfully executed a multi phase investigation into the classification of human grip strategies, progressing from foundational data exploration to the implementation of state-of-the-art deep learning models. The key findings across these phases represent a significant contribution to the field.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Phase 1: Foundational Machine Learning Insights")
    st.info("""
    **1. Validated Kinematic Principles:**
    - Exploratory Data Analysis confirmed that movement kinematics are fundamentally driven by task constraints, such as target distance. PCA visualizations clearly demonstrated that movements naturally cluster based on the target's location, validating the experimental data.
    
    **2. Data-Driven Discovery of Strategies:**
    - Unsupervised clustering algorithms successfully identified distinct movement strategies that directly corresponded to the predefined experimental conditions, proving these strategies are emergent properties of the data, not just theoretical constructs.

    **3. High-Accuracy Predictive Modeling:**
    - A Random Forest classifier achieved near-perfect accuracy (~99%) in predicting target distance when provided with endpoint information (`Zmax` features), establishing a strong upper-bound performance benchmark.
    """)

with col2:
    st.subheader("Phase 2: Advanced Deep Learning Discoveries")
    st.success("""
    **1. Diagnosed and Solved a Critical Model Flaw:**
    - The initial implementation of a standard Transformer model revealed a critical interpretability issue: the model "cheated" by learning sequence length from zero-padding, not kinematic patterns. This diagnosis was a key methodological finding.

    **2. Engineered a Superior Hybrid Architecture:**
    - A novel **CNN-Transformer model** was developed to overcome this flaw. This hybrid architecture significantly improved classification accuracy and, more importantly, learned a scientifically plausible strategy.
    
    **3. Uncovered a Novel Scientific Insight:**
    - The most significant contribution of this work is the interpretation of the CNN-Transformer's attention mechanism. The model learned to base its classification on the **initial movement planning phase (timesteps 0-150)**. This provides new, data-driven evidence that the preparatory neural signature of a grasp is highly predictive of its ultimate goal.
    """)

st.markdown("---")

st.header("Overall Contribution and Future Directions")
st.markdown("""
In conclusion, this project delivers more than just a classification system; it presents a complete, end-to-end blueprint for applying advanced machine learning to biomechanical data. It demonstrates how to move from data validation to insightful model interpretation, culminating in a finding that has direct relevance to motor control theory.

**Future work can build upon this foundation by:**
-   **Probing the CNN:** Further investigating the specific low-level features that the convolutional layers are extracting from the planning phase.
-   **Generalization:** Applying the robust CNN-Transformer architecture to other complex time-series classification problems in biomechanics, such as gait analysis or distinguishing expert vs. novice movements in sports.
-   **Real-Time Application:** Exploring the feasibility of deploying a lightweight version of this model for real-time feedback or prosthetic control, leveraging the insight that a decision can be made very early in the movement sequence.
""")
