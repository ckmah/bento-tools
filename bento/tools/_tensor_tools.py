import cell2cell as c2c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from tensorly.metrics.regression import RMSE
from tqdm.auto import tqdm

from .._utils import track, PATTERN_PROBS


@track
def to_tensor(data, layers, scale=False, copy=False):
    """
    Generate tensor from data where dimensions are (layers, cells, genes).

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    layers : list of str
        Keys in data.layers to build tensor.
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `uns['tensor']` : np.ndarray
            3D numpy array of shape (len(layers), adata.n_obs, adata.n_vars)
    """
    adata = data.copy() if copy else data

    cells = adata.obs_names.tolist()
    genes = adata.var_names.tolist()

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


@track
def signatures(
    data,
    layers,
    ranks,
    nruns=3,
    fillna=0,
    scale=True,
    device="auto",
    random_state=888,
    copy=False,
):
    """Perform tensor decomposition on the 3-dimensional tensor built from `[cell, gene, layers]`.

    Parameters
    ----------
    data : _type_
        _description_
    layers : _type_
        _description_
    rank : _type_
        _description_
    device : str, optional
        _description_, by default "auto"
    random_state : int, optional
        _description_, by default 888
    copy : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    adata = data.copy() if copy else data

    to_tensor(adata, layers=layers, scale=scale)
    tensor = adata.uns["tensor"].copy()

    # Replace nans with 0
    if fillna == 0:
        tensor_mask = ~np.isnan(tensor)
        tensor[~tensor_mask] = 0

    # Or impute nans with mean value across cell dimension (mean for each layer x gene)
    elif fillna == "mean":
        for i, layer in enumerate(tensor):
            imputed = np.where(
                np.isnan(layer), np.nanmean(layer, axis=0), layer
            )
            tensor[i] = imputed

    if isinstance(ranks, int):
        ranks = [ranks]

    # Use gpu if available
    tensor = tl.tensor(tensor)
    try:
        import torch

        tl.set_backend("pytorch")
        tensor = tl.tensor(tensor)

        if (
            (device == "auto") or (device == "cuda")
        ) and torch.cuda.is_available():
            device = "cuda"
            tensor = tensor.to("cuda")
        else:
            device = "cpu"

    except ImportError:
        device = "cpu"

    errors = []
    for rank in tqdm(ranks, desc=f"Device {device}"):
        for i in range(nruns):
            # non-negative parafac decomposition
            weights, factors = non_negative_parafac(
                tensor, rank, init="random", random_state=random_state
            )
            
            if device == "cuda":
                weights = weights.cpu()
                factors = [f.cpu() for f in factors]
                tensor = tensor.cpu()

            # calculate error ignoring missing values
            tensor_mu = tl.cp_to_tensor((weights, factors))
            error = np.sqrt(
                (tensor[tensor != 0] - tensor_mu[tensor != 0]) ** 2
            ).mean().numpy()
            errors.append([error, rank])

        # Save loadings from last decomposition run to adata
        tensor_loadings = {}
        sig_names = [f"Signature {i+1}" for i in range(factors[0].shape[1])]
        adata.uns[f"r{rank}_signatures"] = pd.DataFrame(
            factors[0].numpy(), index=layers, columns=sig_names
        )
        adata.obsm[f"r{rank}_signatures"] = pd.DataFrame(
            factors[1].numpy(), index=adata.obs_names, columns=sig_names
        )
        adata.varm[f"r{rank}_signatures"] = pd.DataFrame(
            factors[2].numpy(), index=adata.var_names, columns=sig_names
        )

    errors = pd.DataFrame(
        errors, columns=["rmse", "rank"]
    )
    errors["rmse"] = errors["rmse"].astype(float)
    adata.uns["signatures_error"] = errors

    return adata


def lp_signatures(data, ranks, nruns=3, device="auto", random_state=888, copy=False):
    """Calculate localization signatures by performing tensor decomposition on the dataset tensor.
        Wrapper for `bento.tl.decompose_tensor()`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData.
    ranks : int
        Rank(s) to perform decomposition.
    device : str, optional
        Type of device to use, valid options include `cpu`, `gpu`, and `auto`, by default `auto`.
        Option `auto` prefers `gpu` over `cpu` if available.
    random_state : int, optional
        Random seed used for reproducibility, by default 888.
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    _type_
        _description_
    """
    return decompose_tensor(
        data,
        PATTERN_PROBS,
        ranks,
        device=device,
        random_state=random_state,
        copy=copy,
    )
