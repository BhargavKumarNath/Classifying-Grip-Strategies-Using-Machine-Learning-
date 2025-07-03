# Install the required package if not already installed
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}

# Load the package
library(arrow)

file_path <- "C:/CourseWork/Dissertation Classifying grip strategies using machine learning/data/02_processed/comprehensive_master_data_universal.parquet"

# Read the Parquet file
df <- read_parquet(file_path)

# Display the first few rows
print(head(df))
