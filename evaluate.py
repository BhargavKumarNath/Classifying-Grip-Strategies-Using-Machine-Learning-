import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from src.models.transformer import GripFormer
from src.data.dataset import GripStrategyDataset
from src.utils.hooks import AttentionExtractor, get_attention_layers
from src.evaluate import visualise_attention, plot_confusion_matrix
import src.config as config
import os

# Create needed directories
os.makedirs("reports/figures", exist_ok=True)

# 1. Load data
df = pd.read_parquet(config.DATA_PATH)

# Drop features if needed
df = df.drop(columns=config.FEATURES_TO_DROP)

# 2. Extract sequences and labels
sequences, labels = GripStrategyDataset.get_data_sequences(df)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Create dataset
dataset = GripStrategyDataset(sequences, labels_encoded, max_len=config.MAX_SEQ_LEN)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# 3. Create class map
class_map = {i: c for i, c in enumerate(label_encoder.classes_)}

# 4. Load model
num_features = sequences[0].shape[1]
model = GripFormer(
    num_features=num_features,
    d_model=config.D_MODEL,
    nhead=config.N_HEADS,
    num_encoder_layers=config.N_LAYERS,
    num_classes=config.NUM_CLASSES,
    dropout=config.DROPOUT
)
model.load_state_dict(torch.load("models/best_model.pth", map_location=config.DEVICE))
model.to(config.DEVICE)

# 5. Run attention visualization on example 0
visualise_attention(model, dataset, class_map, example_idx=0)

# 6. Run full confusion matrix
plot_confusion_matrix(model, dataloader, class_map)
