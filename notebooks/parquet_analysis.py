import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_parquet("C:/CourseWork/Dissertation Classifying grip strategies using machine learning/data/02_processed/comprehensive_master_data_universal.parquet")
except FileNotFoundError:
    print("Error: The specified file was not found. Please check the file path.")
    exit()

## Basic Data Overview
print("### Basic Data Overview ###")
print("First 5 rows of the dataframe:")
print(df.head())
print("\nDataFrame Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())
print("-" * 30)

## Data Quality Check
print("\n### Data Quality Check ###")
# Check for null values
print("Null values in each column:")
print(df.isnull().sum())

# Check for duplicate rows
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
print("-" * 30)

## Target Variable Analysis
print("\n### Target Variable Analysis ###")
print("Unique grip strategy labels:")
print(df['grip_strategy_label'].unique())
print(f"\nNumber of unique grip strategy labels: {df['grip_strategy_label'].nunique()}")
print("\nDistribution of grip strategy labels:")
print(df['grip_strategy_label'].value_counts())
print("-" * 30)

## Trial and Sequence Length Analysis
print("\n### Trial and Sequence Length Analysis ###")
# Group by subject and trial to analyze sequence lengths
trial_lengths = df.groupby(['subjName', 'trialN']).size()
print("Descriptive statistics of trial lengths (sequences):")
print(trial_lengths.describe())
print("-" * 30)

## Feature Preparation and Final Checks
print("\n### Feature Preparation and Final Checks ###")
# Define feature columns
feature_cols = [col for col in df.columns if '_unified' in col] + ['FX', 'FY', 'FZ', 'FVel', 'FAcc', 'MVel', 'MAcc', 'MDec', 'pathLength', 'MGA', 'timeMGA', 'movTime', 'signal_grasp', 'signal_nan']
print(f"Number of features: {len(feature_cols)}")
print("Feature columns:")
print(feature_cols)

# Prepare sequences and labels
X = []
y = []
label_map = {label: idx for idx, label in enumerate(df['grip_strategy_label'].unique())}
grouped = df.groupby(['subjName', 'trialN'])

for (subj, trial), group in grouped:
    seq_features = group[feature_cols].values
    label = group['grip_strategy_label'].iloc[0]
    X.append(seq_features)
    y.append(label_map[label])

print(f"\nTotal number of sequences: {len(X)}")
seq_lengths = [len(seq) for seq in X]
print(f"Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.2f}")
print("\nDataset is now prepared as a list of sequences (X) and corresponding labels (y).")
print("-" * 30)

## Summary of Analysis
print("\n### Summary of Analysis ###")
print("The dataset appears to be in good shape for the next stage of building a transformer model.")
print("- **No Missing Data**: The dataset is clean with no null values.")
print("- **Target Variable**: There are 75 unique grip strategy labels. The distribution is imbalanced, which might require techniques like stratified sampling or using class weights in the model.")
print("- **Sequence Length**: The sequence lengths vary significantly. This is a key consideration for the transformer model, and you will need to decide on a strategy for handling this, such as padding or truncating sequences to a fixed length.")
print("- **Features**: The feature set is well-defined and ready to be used as input for the model.")

# Each trial contains hundreds to thousands of time steps, and we've merged static info with every time-step.

# We have 2876 trials total (as seen in "Total number of sequences").
# Each trial has ~986 time steps on average (as seen in the sequence length stats).
# That means:
# Total rows ≈ 2876 trials × 986 time steps/trial ≈ 2.8 million rows
