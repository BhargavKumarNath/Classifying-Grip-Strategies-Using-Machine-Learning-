import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import mlflow
import mlflow.pytorch
import os
from src.data_loader import GripSequenceDataset
from src.model import LSTMClassifier, GripTransformerClassifier
from src.engine import train_epoch, val_epoch
from torchmetrics import F1Score

# Config
def load_config(config_path="C:/CourseWork/Dissertation Classifying grip strategies using machine learning/config/params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config):
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    processed_path = config["data"]["processed_path"]
    train_dataset = GripSequenceDataset(f"{processed_path}/X_train.npy", f"{processed_path}/y_train.npy")
    val_dataset = GripSequenceDataset(f"{processed_path}/X_val.npy", f"{processed_path}/y_val.npy")

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Model selection and initialisation
    model_params = config["model"]
    transformer_params = config["transformer"]
    model = GripTransformerClassifier(
        input_features=model_params["input_features"],
        num_classes=model_params["num_classes"],
        d_model=transformer_params["d_model"],
        nhead=transformer_params["nhead"],
        num_encoder_layers=transformer_params["num_encoder_layers"],
        dim_feedforward=transformer_params["dim_feedforward"],
        dropout=transformer_params["dropout"],
        seq_length=model_params["sequence_length"]
    ).to(device)

    