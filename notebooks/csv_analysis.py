import pandas as pd

# What Each Row Represents
# Each row in the CSV represents one time step of a fixed-length trial (sequence), and includes:

# Metadata (sequence ID, timestep)

# Grip strategy label (text and encoded)

# 18 numeric features captured at that time step
# So:
# One trial (sequence) = 512 rows
# Total trials (sequences) = 947200 ÷ 512 = 1850

# Column Breakdown (22 Columns)
# Column	Description
# unique_sequence_id	Index of the trial (0 to 1849). All rows with same ID belong to one sequence.
# timestep	Time step index (0 to 511). Sequence always has length = 512.
# label	Original string label of the grip strategy (e.g., 'aiming_clear_black_one').
# label_encoded	Integer-encoded class (0–36).
# indexX_unified, indexY_unified, indexZ_unified	3D position of the index finger
# thumbX_unified, thumbY_unified, thumbZ_unified	3D position of the thumb
# wristX_unified, wristY_unified, wristZ_unified	3D position of the wrist
# FX, FY, FZ	Force in X, Y, Z directions
# FVel, FAcc	Force velocity and force acceleration
# MVel, MAcc, MDec	Motion velocity, acceleration, deceleration
# signal_grasp	A binary signal indicating grasp state (1 = grasp, 0 = no grasp)

# These 18 features are numerical and represent the sensor signals from each time step in a grasping trial.

# Dataset Shape
# Rows: 947,200

# Columns: 22

# Each sequence: 512 rows → 947200 ÷ 512 = 1850 trials

# Labels
# We have 37 unique grip strategy classes

# Each class appears in 50 sequences, so:

# 50 sequences × 512 rows = 25,600 rows per class

# Class distribution is perfectly balanced

# Summary Statistics
# Values are mostly centered and scaled to small positive/negative values

# signal_grasp is mostly 1.0 (around 94.5% of all steps)

# timestep ranges from 0 to 511 for each sequence

# label_encoded ranges from 0 to 36


# Set display options to ensure all columns are shown
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) 

def inspect_csv_data(filepath):
    """
    Loads a CSV file into a pandas DataFrame and prints key information
    about the dataset.

    Args:
        filepath (str): The path to the CSV file.
    """
    try:
        # Load the dataset from the provided path
        df = pd.read_csv(filepath)

        print("---" * 20)
        print(f"Data Loaded Successfully from: {filepath}")
        print("---" * 20 + "\n")

        # 1. Display the first 3 rows (now showing ALL columns)
        print("Each row in the CSV represents one time step of a fixed-length trial (sequence), and includes: \nMetadata (sequence ID, timestep) \nGrip strategy label (text and encoded) \n18 numeric features captured at that time step")
        print("## First 3 Rows")
        print("A quick peek at the first few records.\n")
        print(df.head(3))
        print("\n" + "---" * 20 + "\n")

        # 2. Display all column names
        print("## Column Names")
        print("All the features available in the dataset.\n")
        print(list(df.columns))
        print("\n" + "---" * 20 + "\n")

        # 3. Display data information (data types, non-null counts)
        print("## Data Information (Types & Non-Nulls)")
        print("A summary of column data types and memory usage.\n")
        df.info()
        print("\n" + "---" * 20 + "\n")

        # 4. Display the shape of the dataset
        rows, cols = df.shape
        print("## Dataset Shape")
        print(f"The dataset has {rows} rows and {cols} columns.\n")
        print(f"Shape: {df.shape}")
        print("\n" + "---" * 20 + "\n")

        # 5. Display descriptive statistics for numerical columns
        print("## Descriptive Statistics (for numerical columns)")
        print("A statistical summary of the numerical features.\n")
        print(df.describe())
        print("\n" + "---" * 20 + "\n")

        ## Target Variable Analysis
        print("\n### Target Variable Analysis ###")
        print("Unique grip strategy labels:")
        print(df['label'].unique())
        print(f"\nNumber of unique grip strategy labels: {df['label'].nunique()}")
        print("\nDistribution of grip strategy labels:")
        print(df['label'].value_counts())
        print("-" * 30)

    except FileNotFoundError:
        print(f"ERROR: The file '{filepath}' was not found. Please check the path and filename.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    file_to_inspect = "C:/CourseWork/Classifying_grip_strategies_ml/data/02_processed/GripFormer_preprocessed_dataset.csv"

    inspect_csv_data(file_to_inspect)