import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import mlflow
import mlflow.pytorch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import GripSequenceDataset
from src.model import LSTMClassifier, GripTransformerClassifier
from src.engine import train_epoch, val_epoch
from torchmetrics import F1Score

# Config
def load_config(config_path="C:/CourseWork/Classifying_grip_strategies_ml/config/params.yaml"):
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

    # Loss, optimiser, scheduler  metrics
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler  = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    # f1 scorer from torchmetrics
    f1_scorer = F1Score(task="multiclass", num_classes=model_params["num_classes"], average="macro").to(device)

    # ML FLow Experiment Tracking
    mlflow.set_experiment("Grip Strategy Classification")
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # Log hyperparameters
        mlflow.log_params(config["training"])
        mlflow.log_params(config["transformer"])
        mlflow.log_param("model_type", "Transformer")

        best_val_f1 = -0.1

        # Train loop 
        for epoch in range(config["training"]["epochs"]):
            print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")

            train_loss, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, device, f1_scorer)
            val_loss, val_f1 = val_epoch(model, val_loader, loss_fn, device, f1_scorer)

            # Step the scheduler based on validation F1 score
            scheduler .step(val_f1)

            # Log matrics to MLFlow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train F1={train_f1:.4f} | Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")

            # Save the best model based on Validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1               
                mlflow.pytorch.log_model(model, name="best_model")
                print(f"New best model saved at epoch {epoch+1} with Val F1: {best_val_f1:.4f}")
    
    print("\nTraining Finished")

if __name__ == "__main__":
    config = load_config()
    main(config)
