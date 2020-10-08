import geopandas
import pandas as pd
import numpy as np
from collections import defaultdict

from tqdm import tqdm

from .._settings import settings
from .._settings import pandarallel

tqdm.pandas()

def subsample(data, fraction):
    """Subsample data compartmentalized by cell.
    TODO what is this doing
    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial transcriptomics data.
    fraction : float
        Float between (0, 1] to subsample data.

    Returns
    -------
    AnnData
        Returns subsampled view of original AnnData object.
    """
    keep = data.obs.groupby('cell').apply(
        lambda df: df.sample(frac=fraction)).index.droplevel(0)
    return data[keep, :]


def pca(data, n_components=2):
    """
    """
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data.uns['features'])
    columns = [str(c) for c in range(0, pca_components.shape[1])]
    pca_components = pd.DataFrame(pca_components,
                         index=data.uns['features'].index,
                         columns=columns)
    data.uns['pca_components'] = pca_components
    return data


def umap(data, n_components=2, n_neighbors=15, **kwargs):
    """
    """
    fit = u.UMAP(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
    umap_components = fit.fit_transform(data.uns['features'])
    columns = [str(c) for c in range(0, umap_components.shape[1])]
    umap_components = pd.DataFrame(umap_components,
                         index=data.uns['features'].index,
                         columns=columns)
    data.uns['umap_components'] = umap_components
    return data


def classify_genes(data, model, labels=None):
    # model fit predict return x y
    if labels:
        pred_labels = model.fit_predict(data['features'], labels)
    else:
        pred_labels = model.fit_predict(data['features'])
    return model, pred_labels