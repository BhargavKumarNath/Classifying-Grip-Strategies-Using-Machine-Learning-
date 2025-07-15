import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
from datetime import datetime
import copy
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import GripSequenceDataset
from src.model import LSTMClassifier, GripTransformerClassifier
from src.engine import train_epoch, val_epoch
from src.utils import save_experiment 
from torchmetrics import F1Score

def load_config(config_path='config/params.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create a unique directory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{config['training']['experiment_name']}_{timestamp}"
    run_path = os.path.join('outputs', run_name)
    os.makedirs(run_path, exist_ok=True)
    print(f"Saving results to: {run_path}")

    # Data Loading 
    processed_path = config['data']['processed_path']
    train_dataset = GripSequenceDataset(f'{processed_path}/X_train.npy', f'{processed_path}/y_train.npy')
    val_dataset = GripSequenceDataset(f'{processed_path}/X_val.npy', f'{processed_path}/y_val.npy')
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Model Selection and Initialization
    model_params = config['model']
    transformer_params = config['transformer']
    model = GripTransformerClassifier(
        input_features=model_params['input_features'],
        num_classes=model_params['num_classes'],
        d_model=transformer_params['d_model'],
        nhead=transformer_params['nhead'],
        num_encoder_layers=transformer_params['num_encoder_layers'],
        dim_feedforward=transformer_params['dim_feedforward'],
        dropout=transformer_params['dropout'],
        seq_length=model_params['sequence_length']
    ).to(device)

    # Loss, Optimizer, Scheduler, Metrics
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    f1_scorer = F1Score(task="multiclass", num_classes=model_params['num_classes'], average='macro').to(device)

    # Training Loop
    best_val_f1 = -1.0
    best_model_state = None
    metrics_log = []

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        
        train_loss, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, device, f1_scorer)
        val_loss, val_f1 = val_epoch(model, val_loader, loss_fn, device, f1_scorer)
        
        scheduler.step(val_f1)

        # Log metrics for this epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_f1': val_f1
        }
        metrics_log.append(epoch_metrics)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train F1={train_f1:.4f} | Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")

        # Save the model state if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Use deepcopy to ensure we save the state of the model at this exact point
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best F1 score: {best_val_f1:.4f}. Model state captured.")

    print("\nTraining finished.")
    
    # Save the best model and all results
    if best_model_state:
        # Create a temporary model to load the best state into
        final_best_model = model
        final_best_model.load_state_dict(best_model_state)
        save_experiment(run_path, final_best_model, metrics_log, config)
    else:
        print("No best model was saved as validation F1 never improved.")

if __name__ == '__main__':
    config = load_config()
    main(config)