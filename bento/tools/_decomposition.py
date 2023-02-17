from typing import List, Literal

import numpy as np
import pandas as pd
import tensorly as tl
from anndata import AnnData
from scipy.stats import zscore
from tensorly.decomposition import non_negative_parafac
from tqdm.auto import tqdm

from .._utils import track


def decompose(
    tensor: np.ndarray,
    ranks: List[int],
    iterations: int = 3,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    random_state: int = 11,
):
    """
    Perform tensor decomposition on an input tensor, optionally automatically selecting the best rank across a list of ranks.

    Parameters
    ----------
    tensor : np.ndarray
        numpy array
    ranks : int or list of int
        Rank(s) to perform decomposition.
    iterations : int, 3 by default
        Number of times to run decomposition to compute confidence interval at each rank. Only the best iteration for each rank is saved.
    device : str, optional
        Device to use for decomposition. If "auto", will use GPU if available. By default "auto".
    random_state : int, optional
        Random state for decomposition. By default 11.

    Returns
    -------
    factors_per_rank : dict
        Dictionary of factors for each rank.
    errors : pd.DataFrame
        Dataframe of errors for each rank.
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


def rmse(tensor, tensor_mu):
    return np.sqrt((tensor[tensor != 0] - tensor_mu[tensor != 0]) ** 2).mean()


@track
def to_tensor(
    data: AnnData, layers: List[str], scale: bool = False, copy: bool = False
):
    """
    Generate tensor from data where dimensions are (layers, cells, genes).

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    layers : list of str
        Keys in data.layers to build tensor.
    scale : bool
        Z scale across cells for each layer, by default False.
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        `uns['tensor']` : np.ndarray
            3D numpy array of shape (len(layers), adata.n_obs, adata.n_vars)
    """
    adata = data.copy() if copy else data

    # Build tensor from specified layers
    tensor = []
    for l in layers:
        tensor.append(adata.to_df(l).values)

    # Save tensor values
    tensor = np.array(tensor)

    # Z scale across cells for each layer
    if scale:
        for i, layer in enumerate(tensor):
            tensor[i] = zscore(layer, axis=1, nan_policy="omit")

    adata.uns["tensor"] = np.array(tensor)

    return adata
