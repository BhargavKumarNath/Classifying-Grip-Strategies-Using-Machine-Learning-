# The utils.py file defines a utility function for saving the outputs of a training experiment.



import torch
import os
import shutil
import pandas as pd

def save_experiment(run_path: str, model: torch.nn.Module, metrics: list, config: dict):
    """Saves the results of a training run"""

    model_path = os.path.join(run_path, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Best Model state_dict saved to {model_path}")

    # Save metrics to csv file
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(run_path, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Training metrics saved to {metrics_path}")

    # Save the config file used for this run
    config_path = os.path.join(run_path, "config.yaml")
    shutil.copyfile("config/params.yaml", config_path)
    print(f"Configuration saved to {config_path}")