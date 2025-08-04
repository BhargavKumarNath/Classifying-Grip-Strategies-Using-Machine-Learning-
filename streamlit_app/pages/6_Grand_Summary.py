import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Grand Summary", layout="wide")
st.title("📈 Grand Summary: The Full Narrative")
st.markdown("---")

st.markdown("""
This final visualization synthesizes the entire project's journey into a single, cohesive narrative, presented across four panels. It illustrates the progression from analyzing broad, group-level patterns to quantifying subtle, individual-level differences in motor control.
""")

# Display the pre-generated final figure
st.image("results/figures/grand_summary_visualization.png", use_column_width=True)

st.header("Interpreting the Panels")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Panel A: Clear Strategies Emerge")
    st.markdown("""
    - **Task:** Prehension (a highly constrained reaching-and-grasping task).
    - **Finding:** Unsupervised clustering on the PCA-reduced data reveals two clean, well-separated clusters.
    - **Conclusion:** When a task has strong constraints, participants converge on a limited set of distinct, easily classifiable motor strategies.
    """)
    
    st.subheader("Panel C: Quantifying Individual Differences")
    st.markdown("""
    - **Task:** Visual Illusions.
    - **Finding:** Using our novel "Kinematic Susceptibility" score, we can rank each participant based on how strongly their movements were affected by the illusions.
    - **Conclusion:** Group-level analysis can hide significant individual variability. We can develop metrics to quantify these differences in motor behavior.
    """)
    
with col2:
    st.subheader("Panel B: Ambiguous Group-Level Strategies")
    st.markdown("""
    - **Task:** Aiming (a less-constrained pointing task).
    - **Finding:** In contrast to Panel A, the clusters are messy and overlapping. There are no clearly defined group-level strategies.
    - **Conclusion:** As task constraints are relaxed, movement strategies become more varied and ambiguous at the group level.
    """)
    
    st.subheader("Panel D: Explaining Susceptibility")
    st.markdown("""
    - **Task:** Visual Illusions.
    - **Finding:** By comparing the kinematics of the most vs. least susceptible subjects, we see a clear cause. The most susceptible group shows divergent movement patterns for each illusion (e.g., in Final Grip Orientation), while the least susceptible group does not.
    - **Conclusion:** Individual differences in susceptibility are not random; they are directly reflected in measurable, specific aspects of the movement itself.
    """)