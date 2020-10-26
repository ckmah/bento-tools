from collections import defaultdict

import bento
import geopandas
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from umap import UMAP

from .._settings import pandarallel, settings

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


def pca(data, features, n_components=2, copy=False):
    """Perform principal component analysis on samples using specified features.

    Parameters
    ----------
    data : AnnData
        [description]
    features : [type]
        [description]
    n_components : int, optional
        [description], by default 2
    copy : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy() if copy else data

    if type(features) == str:
        features = [features]
    
    # Initialize PCA
    pca = PCA(n_components=n_components)

    # Compute pca
    features_x = np.array([bento.get_feature(adata, f) for f in features])
    data.uns['sample_pca'] = pca.fit_transform(features_x)

    # Save PCA outputs
    data.uns['pca'] = dict()
    data.uns['pca']['features_used'] = features
    data.uns['pca']['components_'] = PCA.components_
    data.uns['pca']['explained_variance_'] = PCA.explained_variance_
    data.uns['pca']['explained_variance_ratio_'] = PCA.explained_variance_ratio_
    return adata if copy else None


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


def _map_to_obs(data, name):

    if name not in data.uns['sample_data'].keys():
        print(f'{name} not found.')
        return
        
    data.obs[name] = data.uns['sample_data'][name][adata.obs['sample_id']]
    