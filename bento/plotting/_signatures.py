from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .._utils import PATTERN_PROBS, PATTERN_COLORS, pheno_to_color
from ._utils import savefig
from ._colors import red_light


@savefig
def signatures(adata, rank, fname=None):
    """Plot signatures for specified rank across each dimension.

    bento.tl.signatures() must be run first.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial formatted AnnData
    rank : int
        Rank of signatures to plot
    fname : str, optional
        Path to save figure, by default None
    """
    sig_key = f"r{rank}_signatures"
    layer_g = sns.clustermap(
        np.log2(adata.uns[sig_key] + 1).T,
        col_cluster=False,
        row_cluster=False,
        col_colors=pd.Series(PATTERN_COLORS, index=PATTERN_PROBS),
        standard_scale=0,
        cmap=red_light,
        linewidth=1,
        linecolor="black",
        figsize=(adata.uns[sig_key].shape[0], adata.uns[sig_key].shape[1] + 1),
    )
    sns.despine(ax=layer_g.ax_heatmap, top=False, right=False)
    plt.suptitle("Layers")

    gs_shape = adata.varm[sig_key].shape
    gene_g = sns.clustermap(
        np.log2(adata.varm[sig_key] + 1).T,
        row_cluster=False,
        cmap=red_light,
        standard_scale=0,
        figsize=(gs_shape[0], gs_shape[1] + 1),
    )
    sns.despine(ax=gene_g.ax_heatmap, top=False, right=False)
    plt.suptitle("Genes")

    os_shape = adata.obsm[sig_key].shape
    cell_g = sns.clustermap(
        np.log2(adata.obsm[sig_key] + 1).T,
        row_cluster=False,
        col_cluster=True,
        standard_scale=0,
        xticklabels=False,
        # col_colors=pheno_to_color(adata.obs["leiden"], palette="tab20")[1],
        cmap=red_light,
        figsize=(os_shape[0], os_shape[1] + 1),
    )
    sns.despine(ax=cell_g.ax_heatmap, top=False, right=False)
    plt.suptitle("Cells")


def signatures_error(adata, fname=None):
    """Plot error for each rank.

    bento.tl.signatures() must be run first.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial formatted AnnData
    fname : str, optional
        Path to save figure, by default None
    """
    errors = adata.uns["signatures_error"]
    sns.lineplot(data=errors, x="rank", y="rmse", ci=95, marker="o")
    sns.despine()

    return errors
