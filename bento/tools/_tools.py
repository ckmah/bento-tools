import geopandas
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap as u

def pca(data, n_components=2):
    """
    """
    pca = PCA(n_components=n_components)
    x_new = pca.fit_transform(data['features'])
    columns = [str(c) for c in range(0, x_new.shape[1])]
    x_new = pd.DataFrame(x_new,
                         index=data['features'].index,
                         columns=columns)
    data['pca_components'] = x_new
    return data


def umap(data, n_components=2, n_neighbors=15):
    """
    """
    fit = u.UMAP(n_components=n_components, n_neighbors=n_neighbors)
    x_new = fit.fit_transform(data['features'])
    columns = [str(c) for c in range(0, x_new.shape[1])]
    x_new = pd.DataFrame(x_new,
                         index=data['features'].index,
                         columns=columns)
    data['umap_components'] = x_new
    return data
