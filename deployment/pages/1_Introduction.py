import streamlit as st

st.set_page_config(page_title="Introduction", layout="wide")

st.markdown("""
<div style="text-align: center;">
    <h1>Classifying Human Grip Strategies using Machine Learning</h1>
    <h3 style="color: #A0A0A0;">A Dissertation Project on the Intersection of Motor Control and Data Science</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True) 

# SCIENTIFIC BACKGROUND SECTION 
st.subheader("Scientific Background: Decoding the Mind's Blueprint for Action")
st.markdown("""
How does the human brain translate a simple intention like picking up a cup into a fluid and precise movement? This fundamental question lies at the heart of motor neuroscience. The way we reach for and grasp objects provides a remarkable window into the brain's internal models of the world. By analysing the kinematics (the geometry of motion) of these actions, we can begin to decode the planning and execution strategies that govern our interactions with the environment.

This project leverages 3D motion capture data to build a computational framework capable of identifying and classifying these intricate grip strategies. The core hypothesis is that distinct, repeatable patterns exist within kinematic data, and that these patterns can be uncovered and predicted using modern machine learning techniques.
""")

st.markdown("---") 

# AIMS AND OBJECTIVES SECTION 
st.subheader("Project Aims and Objectives")
st.markdown("""
The primary goal of this dissertation is to develop and validate a reproducible data science pipeline for the classification of human grip strategies. This is achieved through the following key objectives:
""")
st.markdown("""
- **Develop a robust data processing pipeline** to transform raw 3D motion capture data into a clean, analysis-ready format.
- **Employ unsupervised learning (e.g., PCA, Clustering)** to discover and validate the existence of distinct, data-driven grip strategies without prior assumptions.
- **Train and evaluate supervised machine learning models** to accurately classify grip strategies based on kinematic features, testing the predictive power of movement dynamics.
- **Implement and interpret advanced deep learning models (Transformers)** to classify grip types from raw time-series data, pushing the boundaries of sequence analysis in this domain.
- **Utilize model interpretability techniques (e.g., Attention Visualization)** to gain novel scientific insights into the temporal dynamics of motor planning and execution.
""")

st.markdown("---") 

#  KEY CONTRIBUTION SECTION 
with st.container(border=True):
    st.markdown("##### 🔑 Key Contribution")
    st.write("""
    This work bridges the gap between classical motor control research and modern data science. It delivers not only a high-performance classification system but, more importantly, a framework for generating new, interpretable insights into the cognitive processes that underlie human grasping actions. The findings presented here have direct implications for both neuroscience and the development of advanced human-computer interfaces.
    """)
