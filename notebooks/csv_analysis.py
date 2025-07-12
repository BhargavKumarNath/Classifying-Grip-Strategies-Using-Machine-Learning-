import pandas as pd

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
        print(f"The dataset has {rows} rows and {cols} columns. 📈\n")
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
    file_to_inspect = "C:/CourseWork/Dissertation Classifying grip strategies using machine learning/data/02_processed/GripFormer_preprocessed_dataset.csv"

    inspect_csv_data(file_to_inspect)