import streamlit as st
import pandas as pd
import helpers

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")
st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Here, we investigate the characteristics of each dataset to understand the experimental design, data quality, and key kinematic patterns.")

# Sidebar for Dataset Selection
dataset_name = st.sidebar.selectbox(
    "Choose a dataset to explore:",
    ("aiming", "prehension", "visual_illusions"),
    format_func=lambda x: x.replace("_", " ").title()
)

df = helpers.load_data(dataset_name)

if df is not None:
    st.header(f"Analysis of: {dataset_name.replace('_', ' ').title()} Dataset")

    # Define kinematic variables based on dataset
    if dataset_name == 'prehension':
        kinematic_vars = ['movTime', 'pathLength', 'MVel', 'MGA', 'MAcc', 'MDec', 'timeMGA']
    elif dataset_name == 'visual_illusions':
        kinematic_vars = ['movTime', 'pathLength', 'MVel', 'MGA', 'MAcc', 'MDec']
    else: # aiming
        kinematic_vars = ['movTime', 'pathLength', 'PeakVelocity', 'MaxGripAperture', 'MAcc', 'MDec', 'timeMGA']
    
    # Make sure all variables exist in the dataframe
    kinematic_vars = [var for var in kinematic_vars if var in df.columns]

    # Use Tabs for Organization
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Distributions", "Correlations", "Dimensionality Reduction (PCA)"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("Dataset Info")
        st.text(str(df.info()))

    with tab2:
        st.subheader("Distributions of Experimental and Subject Variables")
        with st.spinner("Generating distribution plots..."):
            fig = helpers.plot_distributions(df, dataset_name)
            st.pyplot(fig)

        st.subheader("Distributions of Key Kinematic Variables")
        with st.spinner("Generating kinematic plots..."):
            fig_kin = helpers.plot_kinematic_distributions(df, kinematic_vars)
            st.pyplot(fig_kin)
            st.markdown("""
            **Observation:** Many kinematic variables, like `movTime` and `pathLength`, are right-skewed. This is typical in motor control data, where most movements are efficient, but a tail of slower/longer movements exists.
            """)

    with tab3:
        st.subheader("Correlation Between Kinematic Features")
        with st.spinner("Generating correlation heatmap..."):
            fig_corr = helpers.plot_correlation_heatmap(df, kinematic_vars)
            st.pyplot(fig_corr)
            st.markdown("""
            **Insight:** We consistently observe strong correlations that validate the data's integrity. For example, a strong positive correlation between `pathLength` and `movTime` (longer paths take more time) and a strong negative correlation between `PeakVelocity` and `movTime` (faster movements are quicker).
            """)
    
    with tab4:
        st.subheader("Dimensionality Reduction with Principal Component Analysis (PCA)")
        st.markdown("""
        PCA helps us visualize the high-dimensional kinematic data in a 2D space. The key question is: **Do the experimental conditions create separable clusters in the data?**
        """)
        with st.spinner("Running PCA and generating plots..."):
            features_for_pca = [col for col in kinematic_vars if col in df.columns]
            fig_pca = helpers.plot_pca_scatter(df, features_for_pca)
            st.pyplot(fig_pca)
            st.success("""
            **Key Finding:** The separation is most distinct when colored by **distance** or **target position**. This strongly suggests that the primary factor driving the variance in movement kinematics is the target's location, which is a foundational finding for our modeling phase.
            """)