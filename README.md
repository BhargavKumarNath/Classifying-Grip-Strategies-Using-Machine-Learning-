# Classifying Grip Strategies using Machine Learning on 3D Motion Capture Data

## 1. Motivation
Human prehension (grasping and reaching) is a fundamental aspect of motor control, offering valuable insights into neural mechanisms and object representation. Traditional kinematic analysis often struggles to capture the complexity inherent in high-dimensional motion capture data. This project addresses this challenge by developing a machine learning framework for classifying, interpreting, and explaining motor strategies across various reaching and grasping tasks. The ability to accurately classify these strategies allows for deeper understanding of how humans interact with their environment and can potentially inform robotics and rehabilitation applications.

## 2. Design & Architecture
This project utilizes a multi-stage analytical pipeline built around machine learning techniques. The core architecture involves:

*   **Data Acquisition:** Three distinct datasets were used: prehension, aiming, and visual illusions (details on data sources are in the dissertation report).
*   **Preprocessing:**  Kinematic features are extracted from motion capture data. A `StandardScaler` is applied to normalize these features.
*   **Dimensionality Reduction & Clustering (Exploratory Phase):** Initial exploration involved unsupervised learning techniques like K-Means and Hierarchical Clustering, combined with Principal Component Analysis (PCA) for dimensionality reduction. This phase aimed to identify underlying motor strategies in constrained tasks.
*   **Supervised Learning:**  Random Forest and XGBoost classifiers were trained to predict experimental conditions from kinematic features.
*   **Explainable AI (XAI):** SHAP values are used to interpret model predictions and highlight influential features, particularly focusing on "Final Grip Orientation" (FGot).
*   **Task Fingerprint Model:** A novel integration of the three datasets to distinguish between motor tasks based on shared kinematic features.
*   **Kinematic Susceptibility Score:**  A data-driven metric quantifying the ambiguity of model predictions for individual movements during visual illusions.

**Technologies & Frameworks:**

*   Python 3.11
*   Scikit-learn (for machine learning algorithms, pipeline construction, cross-validation)
*   pandas, numpy, matplotlib, seaborn, scipy, streamlit, xgboost, shap, statsmodels

## 3. Phases of Development

1.  **Data Collection & Preprocessing:** Gathering and cleaning the three datasets, feature extraction from motion capture data.
2.  **Exploratory Data Analysis (EDA) & Unsupervised Learning (Month 2-3):** Applying K-Means, Hierarchical Clustering, and PCA to identify initial motor strategies.
3.  **Supervised Model Development & Training:** Implementing and training Random Forest and XGBoost classifiers; hyperparameter tuning.
4.  **Explainable AI Integration:** Applying SHAP values for feature importance analysis.
5.  **Task Fingerprint Model Creation:** Integrating datasets to create the "Task Fingerprint" model.
6.  **Kinematic Susceptibility Score Development:** Defining and implementing the data-driven metric.
7.  **Evaluation & Validation:** Rigorous testing using GroupKFold cross-validation to prevent data leakage, performance evaluation.
8. **Documentation & Code Refinement**

## 4. Results & Contribution

This project makes the following key contributions:

*   **Task Fingerprint Model:** A novel model that integrates three datasets and distinguishes between motor tasks based on shared kinematic features.
*   **Kinematic Susceptibility Score:** A new metric for quantifying prediction ambiguity, providing insights into data variability and model limitations.
*   Demonstrated the effectiveness of machine learning techniques (Random Forest, XGBoost) in classifying motor strategies from kinematic data with high accuracy (>99% initially, but refined after leakage mitigation).
*   Identified "Final Grip Orientation" (FGot) as a key driver of classification using XAI.

## 5. Future Work

*   **Expand Dataset Diversity:** Incorporate data from a wider range of tasks and subjects to improve model generalisability.
*   **Real-time Implementation:**  Develop a real-time system for classifying motor strategies during interactive tasks.
*   **Integration with Robotics:** Utilise the Task Fingerprint Model to inform robotic grasping and manipulation algorithms.
*   **Investigate Alternative XAI Techniques:** Explore other explainable AI methods to gain further insights into model behavior.
*   **Refine Kinematic Susceptibility Score:**  Further develop the metric to incorporate more nuanced aspects of prediction uncertainty.
* **Address Data Leakage:** Further investigate and mitigate potential sources of data leakage in the models.


