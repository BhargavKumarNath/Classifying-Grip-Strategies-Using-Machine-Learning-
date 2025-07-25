import streamlit as st

st.set_page_config(
    page_title="Grip Strategy Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Welcome to the Grip Strategy Classification Project")
st.sidebar.success("Select a page above to begin")

st.markdown("""
This interactive web application serves as a presentation medium for my dissertation project: **"Classifying Grip Strategies using Machine Learning on 3D Motion Capture Data"***.
            
Instead of static presentation, this tool allows you to explore the data, visualise the results, and understand the model's decisions in real time.
            
The project is devided into two main phases:
            1. **Classical Machine Learning:** Where we explore the data, discover inherent grip strategies using unsupervised learning, and predict experimental conditions using supervised models.
            2. **Deep Learning:** Where we leverage advanced Transformer and CNN Transformer models to classify grip types from raw time series data and interpret their decisiion making process using attention mechanisms.

**Please use the navigation panel on the left to explore the different sections of the project.**
""")

st.info("The content for each step of the project is organised into seperate pages. You may start with the 'Introduction'")
