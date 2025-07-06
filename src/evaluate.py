import torch
import torch.nn as nn
import math
import config


def plot_confusion_matrix(model, dataloader, class_map, save_path = "reports/figures/confusion_matrix.png"):
    """Computes and plots a normalised confusion matrix for the model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(config.DEVICE)