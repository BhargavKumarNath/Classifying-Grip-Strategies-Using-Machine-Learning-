# Classifying Grip Strategies using Machine Learning on 3D Motion Capture Data

## Project Structure

The project’s data is organized into four main folders within the `data` directory:

- `01_raw/`  
  Contains the original data files and folders. Inside each task folder — *Aiming*, *Prehension*, and *Visual Illusions* — there is a `filtered_data` subfolder where all the R data files have been converted to CSV format.

- `02_processed/`  
  Contains cleaned and merged CSV files for each respective task folder (*Aiming*, *Prehension*, *Visual Illusions*). These processed CSV files in the `filtered_data` subfolders are ready for machine learning modeling.

Additional folders include:

- `src/`  
  Intended for source code scripts and modules related to data processing, feature engineering, and modeling. This folder will be populated in later stages of the project.

- `models/`  
  Will be used to store trained machine learning models, saved checkpoints, and related artifacts. This folder will also be populated as the project advances.


The `notebooks/` folder includes Jupyter notebooks for data filtering and preprocessing steps, along with pseudocode explanations. This folder will be updated progressively as the project advances.
