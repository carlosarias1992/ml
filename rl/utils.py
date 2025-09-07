import torch
import numpy as np
import random
import glob
import os


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def find_latest_file_path(exp_name_base, filename):
    """Finds the path to the latest model checkpoint for a given experiment name base."""
    base_dir_pattern = f"*{exp_name_base}*"
    matching_dirs = [d for d in glob.glob(base_dir_pattern) if os.path.isdir(d)]
    
    if not matching_dirs:
        raise FileNotFoundError(f"No model directories found for experiment base '{exp_name_base}'.")

    latest_model_dir = max(matching_dirs, key=os.path.getctime)
    
    timestep_dirs = [d for d in os.listdir(latest_model_dir) if d.isdigit()]
    if not timestep_dirs:
        raise FileNotFoundError(f"No trained model checkpoints found in '{latest_model_dir}'.")

    latest_timestep_dir = max(timestep_dirs, key=int)
    model_path = os.path.join(latest_model_dir, latest_timestep_dir, filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
        
    print(f"Found latest model for '{exp_name_base}' at: {model_path}")
    return model_path
