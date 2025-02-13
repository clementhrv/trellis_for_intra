from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch_geometric.data import Data

def get_num_workers(param: Dict[str, Any], default_num_workers: int) -> int:
    """
    Determines the number of workers to use for DataLoader based on dataset extension.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        default_num_workers (int): The default number of workers specified.

    Returns:
        int: The adjusted number of workers.
    """
    dataset_params = param.get("dataset", {})
    extension = dataset_params.get("extension", "")
    if extension == "h5":
        return 0
    elif extension == "xdmf":
        return default_num_workers
    else:
        raise ValueError(f"Dataset extension '{extension}' not supported.")
