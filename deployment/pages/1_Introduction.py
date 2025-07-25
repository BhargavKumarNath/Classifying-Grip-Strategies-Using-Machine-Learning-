import streamlit as st
st.set_page_config(page_title="Introduction", layout="wide")

st.title("Project Introduction")
st.header("Title: Classifying Grip Strategies using Machine Learning on 3D Motion Capture Data")

st.subheader("Background")
st.markdown("""
The analysis of human grasping behavior reveals how 3D objects are internally represented in the brain. By analysing kinematic data we can classify grasping strategies and investigate how humans plan and execute reaching-to-grasp movements based on an object's shape and orientation. This project will focus on developing a machine learning pipeline to classify grip strategies from 3D motion capture data.
""")

st.subheader("Aims and Objectives")
st.markdown("""
- **Develop a reproducible Machine Learning pipeline** to classify grip strategies based on kinematic features.
- **Explore dimensionality reduction techniques** (e.g., PCA) to identify key patterns in the data.
- **Investigate Clustering methods** to group similar grip strategies.
- **Visualise and interpret results** to understand the relationship between kinematic features and grasp planning.
- **Apply supervised learning algorithms** (e.g., Random Forest, SVM) to classify grip strategies based on predefined conditions.
- **Implement and interpret advanced Deep Learning models** (Transformers) for time series classification.
""")

