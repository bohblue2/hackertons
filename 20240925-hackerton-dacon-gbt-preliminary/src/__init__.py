import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
root_path = "/root/dacon-2024-gbt-hackerton"
path_exists = os.path.exists(root_path)
if not path_exists:
    root_path = "/Users/baeyeongmin/Desktop/workspace/dacon-2024-gbt-hackerton"

import numpy as np
import torch

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True