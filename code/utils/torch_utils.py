import numpy as np
import torch

def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)   

def set_value(_x, x):
    if list(x.shape)!=list(_x.size()):
        _x.resize_(x.shape)
    _x.data.copy_(torch.from_numpy(x))
    
