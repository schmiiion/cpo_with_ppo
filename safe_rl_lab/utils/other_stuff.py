import random
import numpy as np
import torch
import os


def seed_all(seed: int):
    # 1. Python (used by many libraries internally)
    random.seed(seed)

    # 2. Numpy (CRITICAL: used by Gym/Safety Gym)
    np.random.seed(seed)

    # 3. Pytorch (CPU + GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4. Deterministic algorithms (optional, but needed for 100% repro)
    # This might slow down training slightly, but guarantees exactness.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. Environment variable (sometimes used by external libs)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Global seed set to: {seed}")