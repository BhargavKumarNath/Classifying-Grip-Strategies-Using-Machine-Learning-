import torch
from tqdm import tqdm
import torchmetrics

def train_epoch(model, dataloader, loss_fn, optimizer, device, f1_scorer):
    """Performs one full training epoch"""
    model.train()
    total_loss = 0.0

    # Rest metrics at the start of each epoch
    f1_scorer.reset()

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for features, labels in progress_bar:
        features, labels = features.to(device), labels.to(device)

        # 1. Forward pass
        outputs = model(features)

        # 2. Calculate loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # 3. Backward pass and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        preds = torch.argmax(outputs, dim=1)
        f1_scorer.update(preds, labels)
    avg_loss = total_loss/len(dataloader)
    f1_scorer = f1_scorer.compute().item()
    return avg_loss, f1_scorer

def val_epoch(model, dataloader, loss_fn, device, f1_scorer):
    """Performs one full validation epoch"""
    model.eval()
    total_loss = 0.0

    f1_scorer.reset()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)

            # 1. Forward pass
            outputs = model(features)

            # 2. Cal loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Update metrics
            preds = torch.argmax(outputs, dim=1)
            f1_scorer.update(preds, labels)
    
    avg_loss = total_loss / len(dataloader)
    f1_scorer = f1_scorer.compute().item()
    return avg_loss, f1_scorer
