import torch
from torch.utils.data import Dataset
import numpy as np


# This script defines a PyTorch Dataset class to load and serve preprocessed sequence data (saved as .npy files) for deep learning models like Transformers.
# PyTorch provides a base class called torch.utils.data.Dataset.
# You can subclass it to define how your custom data (like .npy files) should be:

# loaded from disk 

# converted to tensors

# served one-by-one to the model during training

class GripSequenceDataset(Dataset):
    """ Custom PyTorch Dataset for loading grip strategy sequence"""
    def __init__(self, features_path: str, labels_path: str):
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sequence = self.features[idx]
        label = self.labels[idx]
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return sequence_tensor, label_tensor
    
if __name__ == "__main__":
    PROCESSED_DATA_PATH = "C:/CourseWork/Classifying_grip_strategies_ml/data/03_processed_dl/"
    X_TRAIN_PATH = f"{PROCESSED_DATA_PATH}/X_train.npy"
    Y_TRAIN_PATH = f"{PROCESSED_DATA_PATH}/y_train.npy"

    print("--- Testing GripSequenceDataset ---")

    # 1. Instantiate the dataset
    train_dataset = GripSequenceDataset(features_path=X_TRAIN_PATH, labels_path=Y_TRAIN_PATH)

    # 2. Check its length
    print(f"Number of training sequences: {len(train_dataset)}")

    # 3. Get a single sample
    first_sequence, first_label = train_dataset[0]

    # 4. Check the sample's shape and type
    print(f"\nShape of a single sequence: {first_sequence.shape}")
    print(f"Data type of sequence: {first_sequence.dtype}")      
    print(f"\nShape of a single label: {first_label.shape}")       
    print(f"Value of first label: {first_label.item()}")
    print(f"Data type of label: {first_label.dtype}")        

    print("\nDataset class seems to be working correctly!")
