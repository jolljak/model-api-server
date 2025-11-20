import os
import torch

def get_device():
    if os.getenv("FORCE_CPU") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
