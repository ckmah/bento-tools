from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import zscore

from .._utils import PATTERN_PROBS, PATTERN_COLORS, pheno_to_color
from ._utils import savefig
from ._colors import red_light, red2blue


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


@savefig
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


@savefig
def factor(
    factors, labels, names, n_top=[None, None, 5], z_score=None, fname=None
):
    """ """
    n_factors = len(factors)
    fig, axes = plt.subplots(
        2,
        n_factors,
        figsize=(8, 10),
        gridspec_kw=dict(
            width_ratios=[1] + [4] * (n_factors - 1), height_ratios=[20, 1]
        ),
    )

    for i in range(n_factors):
        factor = factors[i]
        feature_labels = labels[i]
        name = names[i]

        if isinstance(n_top, list):
            n = n_top[i]
        else:
            n = n_top

        if isinstance(z_score, list):
            z = z_score[i]
        else:
            z = z_score

        _plot_loading(
            factor, feature_labels, name, n, z_score, axes[0][i], axes[1][i]
        )

    plt.tight_layout()


def _plot_loading(mtx, feature_labels, name, n_top, z_score, ax, cbar_ax):
    """
    Plot a heatmap representation of a loadings matrix, optionally z-scored and subsetted to the n_top rows of each factor.
    """
    mtx = pd.DataFrame(mtx, index=feature_labels)

    # Optionally z-score each column
    if z_score:
        mtx = mtx.apply(zscore, axis=0)
        center = 0
    else:
        center = None

    # Subset to factor
    if n_top:
        top_indices = []
        for col in mtx.columns:
            top_indices.extend(
                mtx.sort_values(col, ascending=False)
                .head(n_top)
                .index.tolist()
            )
        mtx = mtx.loc[top_indices]

    # Get hierarchical clustering row order
    row_order = sns.clustermap(
        mtx, col_cluster=False
    ).dendrogram_row.reordered_ind
    plt.close()

    # Plot heatmap
    sns.heatmap(
        mtx.iloc[row_order],
        center=center,
        cmap=red2blue,
        cbar_kws=dict(orientation="horizontal"),
        cbar_ax=cbar_ax,
        ax=ax,
    )
    ax.set_xlabel("Factors")
    ax.set_title(f"{name} [{mtx.shape[0]} x {mtx.shape[1]}]")
    sns.despine(ax=ax, right=False, top=False)
