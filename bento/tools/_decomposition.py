from typing import List, Literal

import numpy as np
import pandas as pd
import tensorly as tl
from spatialdata._core.spatialdata import SpatialData
from scipy.stats import zscore
from tensorly.decomposition import non_negative_parafac
from tqdm.auto import tqdm

#from .._utils import track


def decompose(
    tensor: np.ndarray,
    ranks: List[int],
    iterations: int = 3,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    random_state: int = 11,
):
    """
    Perform tensor decomposition on an input tensor using non-negative PARAFAC.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor for decomposition.
    ranks : List[int]
        List of ranks to perform decomposition for.
    iterations : int, default 3
        Number of times to run decomposition for each rank to compute confidence interval. The best iteration for each rank is saved.
    device : Literal["auto", "cpu", "cuda"], default "auto"
        Device to use for decomposition. If "auto", will use GPU if available.
    random_state : int, default 11
        Random state for reproducibility.

    Returns
    -------
    Tuple[Dict[int, List[np.ndarray]], pd.DataFrame]
        A tuple containing:
        - factors_per_rank: Dictionary mapping each rank to a list of factor matrices.
        - errors: DataFrame with columns 'rmse' and 'rank', containing the error for each rank.
    """
    # Replace nans with 0 for decomposition
    tensor_mask = ~np.isnan(tensor)
    tensor[~tensor_mask] = 0

    if isinstance(ranks, int):
        ranks = [ranks]

    # Use gpu if available
    tensor = tl.tensor(tensor)
    try:
        import torch

        tl.set_backend("pytorch")
        tensor = tl.tensor(tensor)

        if ((device == "auto") or (device == "cuda")) and torch.cuda.is_available():
            device = "cuda"
            tensor = tensor.to("cuda")
        else:
            device = "cpu"

    except ImportError:
        torch = None
        device = "cpu"

    factors_per_rank = dict()
    errors = []
    for rank in tqdm(ranks, desc=f"Device {device}"):
        best_factor = None
        best_error = np.inf
        for i in range(iterations):
            # non-negative parafac decomposition

            # TODO update to hals when random_state is supported
            weights, factors = non_negative_parafac(
                tensor, rank, init="random", random_state=random_state
            )

            if device == "cuda":
                weights = weights.cpu()
                factors = [f.cpu() for f in factors]
                tensor = tensor.cpu()

            # calculate error ignoring missing values
            tensor_mu = tl.cp_to_tensor((weights, factors))

            if device == "cuda":
                error = rmse(tensor, tensor_mu).numpy()
            else:
                error = rmse(tensor, tensor_mu)

            if error < best_error:
                best_error = error

                if torch:
                    best_factor = [f.numpy() for f in factors]
                else:
                    best_factor = factors

        factors_per_rank[rank] = best_factor
        errors.append([best_error, rank])

    errors = pd.DataFrame(errors, columns=["rmse", "rank"])
    errors["rmse"] = errors["rmse"].astype(float)

    return factors_per_rank, errors


def rmse(tensor: np.ndarray, tensor_mu: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two tensors, ignoring zero values.

    Parameters
    ----------
    tensor : np.ndarray
        Original tensor.
    tensor_mu : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        RMSE between the non-zero elements of the original and reconstructed tensors.
    """
    return np.sqrt((tensor[tensor != 0] - tensor_mu[tensor != 0]) ** 2).mean()
