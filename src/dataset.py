import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import config

class GripStrategyDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        seq_length = seq.shape[0]

        # Applt padding or truncation
        if seq_length < self.max_len:
            pad_width = self.max_len - seq_length

            # Pad with zeros
            padded_seq = seq[:self.max_len, :]
            mask = np.ones(self.max_len)
        
        return {
            "sequences": torch.tensor(padded_seq, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long)
        }

    def get_data_sequences(df):
        sequences = []
        labels = []
        for name, group in df.groupby(config.GROUPS_COLS):
            feature_cols = [c for c in df.columns if c not in config.GROUPS_COLS + [config.TARGET_COL]]
            sequences.append(group[feature_cols].values)
            labels.append(group[config.TARGET_COL].iloc[0])
        return sequences, labels
    