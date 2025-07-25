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

# Sidebar for Outlier Removal Settings
st.sidebar.subheader("Outlier Removal Settings")
remove_outliers = st.sidebar.checkbox("Remove Outliers from Kinematic Variables", value=True)
outlier_multiplier = st.sidebar.slider("IQR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                                     help="Lower values = more aggressive outlier removal")

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

    # Show dataset info with outlier status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Dataset Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Kinematic Variables", len(kinematic_vars))

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
        
        # Add info about outlier removal settings
        if remove_outliers:
            st.info(f"📊 Outlier removal is **enabled** with IQR multiplier of {outlier_multiplier}")
        else:
            st.warning("⚠️ Outlier removal is **disabled** - plots may show compressed distributions due to extreme values")
        
        with st.spinner("Generating kinematic plots..."):
            # Unpack the tuple returned by the updated function
            fig_kin, df_processed, outliers_info = helpers.plot_kinematic_distributions(
                df, kinematic_vars, remove_outliers, outlier_multiplier
            )
            st.pyplot(fig_kin)
            
            # Show outlier removal summary if outliers were removed
            if remove_outliers and outliers_info:
                with st.expander("📋 Outlier Removal Summary", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Original Rows", f"{df.shape[0]:,}")
                        st.metric("Processed Rows", f"{df_processed.shape[0]:,}")
                    
                    with col2:
                        total_removed = df.shape[0] - df_processed.shape[0]
                        removal_pct = (total_removed / df.shape[0]) * 100
                        st.metric("Rows Removed", f"{total_removed:,}")
                        st.metric("Removal Percentage", f"{removal_pct:.1f}%")
                    
                    # Create detailed summary table
                    st.subheader("Outliers by Variable")
                    outlier_summary = []
                    for var, info in outliers_info.items():
                        outlier_summary.append({
                            'Variable': var,
                            'Outliers Removed': info['count'],
                            'Lower Bound': f"{info['lower_bound']:.3f}",
                            'Upper Bound': f"{info['upper_bound']:.3f}"
                        })
                    
                    outlier_df = pd.DataFrame(outlier_summary)
                    st.dataframe(outlier_df, use_container_width=True)
            
            st.markdown("""
            **Observation:** Many kinematic variables, like `movTime` and `pathLength`, are right-skewed. This is typical in motor control data, where most movements are efficient, but a tail of slower/longer movements exists.
            
            💡 **Tip:** Toggle outlier removal to see how extreme values affect the distribution visualization. Outlier removal helps reveal the underlying data patterns but should be used carefully in analysis.
            """)

    with tab3:
        st.subheader("Correlation Between Kinematic Features")
        
        # Use processed data for correlation if outliers were removed
        df_for_corr = df_processed if remove_outliers and 'df_processed' in locals() else df
        
        if remove_outliers and 'df_processed' in locals():
            st.info("Correlations calculated using outlier-cleaned data")
        
        with st.spinner("Generating correlation heatmap..."):
            fig_corr = helpers.plot_correlation_heatmap(df_for_corr, kinematic_vars)
            st.pyplot(fig_corr)
            st.markdown("""
            **Insight:** We consistently observe strong correlations that validate the data's integrity. For example, a strong positive correlation between `pathLength` and `movTime` (longer paths take more time) and a strong negative correlation between `PeakVelocity` and `movTime` (faster movements are quicker).
            
            **Analysis Note:** Outlier removal can strengthen correlation patterns by reducing noise from extreme measurements.
            """)
    
    with tab4:
        st.subheader("Dimensionality Reduction with Principal Component Analysis (PCA)")
        st.markdown("""
        PCA helps us visualize the high-dimensional kinematic data in a 2D space. The key question is: **Do the experimental conditions create separable clusters in the data?**
        """)
        
        # Use processed data for PCA if outliers were removed
        df_for_pca = df_processed if remove_outliers and 'df_processed' in locals() else df
        
        if remove_outliers and 'df_processed' in locals():
            st.info("PCA performed using outlier-cleaned data for clearer clustering")
        
        with st.spinner("Running PCA and generating plots..."):
            features_for_pca = [col for col in kinematic_vars if col in df_for_pca.columns]
            fig_pca = helpers.plot_pca_scatter(df_for_pca, features_for_pca)
            st.pyplot(fig_pca)
            st.success("""
            **Key Finding:** The separation is most distinct when colored by **distance** or **target position**. This strongly suggests that the primary factor driving the variance in movement kinematics is the target's location, which is a foundational finding for our modeling phase.
            
            ✨ **Enhancement:** Outlier removal often improves cluster separation in PCA by reducing the influence of extreme data points that can skew the principal components.
            """)

else:
    st.error("Failed to load the selected dataset. Please check the file path and try again.")