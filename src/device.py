import os
import torch


def get_device():
    """
    Return torch.device('cuda') if available and not forced to CPU,
    else torch.device('cpu').
    Use env var FORCE_CPU=1 or FORCE_CPU=true to force cpu.
    """
    force_cpu = os.environ.get("FORCE_CPU", "") in ("1", "true", "True")
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
