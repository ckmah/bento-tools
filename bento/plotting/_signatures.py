import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore

from ._colors import red2blue, red_light
from ._utils import savefig


def colocation(
    sdata,
    rank,
    n_top=[None, None, 5],
    z_score=[False, True, True],
    cut=None,
    show_labels=[True, False, True],
    cluster=[False, True, False],
    self_pairs=False,
    figsize=(8, 6),
    fname=None,
):
    """Plot colocation of signatures for specified rank across each dimension.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        Spatial formatted SpatialData
    rank : int
        Rank of signatures to plot
    n_top : int, optional
        Number of top features to plot, by default 10
    z_score : bool, optional
        Whether to z-score each column of the matrix, by default False
    cut : float, optional
        Max cut-off for z-score color mapping, by default None
    show_labels : list, optional
        Whether to show labels for each dimension, by default [True, False, False]
    cluster : list, optional
        Whether to cluster rows, by default [False, True, True]
    self_pairs : [True, False, "only"], optional
        Whether to include self-pairs, value "only" shows only self-pairs, by default True
    fname : str, optional
        Path to save figure, by default None
    """
    factors = sdata.tables["table"].uns["factors"][rank].copy()
    labels = sdata.tables["table"].uns["tensor_labels"].copy()
    names = sdata.tables["table"].uns["tensor_names"].copy()

    # Perform z-scaling upfront
    for i in range(len(factors)):
        if isinstance(z_score, list):
            z = z_score[i]
        else:
            z = z_score

        if z:
            factors[i] = zscore(factors[i], axis=0)

    pairs = []
    for p in labels["pair"]:
        pair = p.split("_")
        pairs.append(pair)

    # Filter out self-pairs appropriately
    valid_pairs = [True] * len(pairs)
    if self_pairs == "only":
        valid_pairs = [True if p[0] == p[1] else False for p in pairs]
    elif not self_pairs:
        valid_pairs = [True if p[0] != p[1] else False for p in pairs]

    valid_pairs = np.array(valid_pairs)

    factors[2] = factors[2][valid_pairs]
    labels["pair"] = labels["pair"][valid_pairs]

    if self_pairs == "only":
        labels["pair"] = [p.split("_")[0] for p in labels["pair"]]

    return factor(
        factors,
        labels,
        names,
        n_top=n_top,
        cut=cut,
        show_labels=show_labels,
        cluster=cluster,
        figsize=figsize,
        fname=fname,
    )


@savefig
def factor(
    factors,
    labels,
    names,
    n_top=False,
    cut=None,
    show_labels=False,
    cluster=True,
    figsize=None,
    fname=None,
):
    """
    Plot a heatmap representation of a loadings matrix, optionally z-scored and subsetted to the n_top rows of each factor.

    Parameters
    ----------
    factors : list of np.ndarray
        List of factors to plot, in the order [layers, cells, *]
    labels : dict
        Dict of {name: labels} for each factor
    names : list of str
        List of names for each factor, in the order [layers, cells, *]
    n_top : int or list of int, optional
        Number of top features to plot, by default None. If None, all features are plotted.
    show_labels : bool or list of bool, optional
        Whether to show labels, by default None. If None, labels are shown.
    cluster : bool or list of bool, optional
        Whether to cluster rows, by default False. If False, rows are not clustered.
    """
    n_factors = len(factors)
    fig, axes = plt.subplots(
        1,
        n_factors,
        figsize=figsize,
        gridspec_kw=dict(
            width_ratios=[1] + [4] * (n_factors - 1),
            wspace=0.05,
        ),
        layout="constrained",
    )

    for i, name in enumerate(names):
        factor = factors[i]
        feature_labels = labels[name]
        factor = pd.DataFrame(factor, index=feature_labels)
        factor.columns.name = "Factors"

        name = names[i]

        if isinstance(n_top, list):
            n = n_top[i]
        else:
            n = n_top

        if isinstance(cut, list):
            cu = cut[i]
        else:
            cu = cut

        if isinstance(show_labels, list):
            show_l = show_labels[i]
        else:
            show_l = show_labels

        if isinstance(cluster, list):
            c = cluster[i]
        else:
            c = cluster

        if i == 0:
            factor = factor.T
            square = True
        else:
            square = False

        _plot_loading(
            factor,
            name=name,
            n_top=n,
            cut=cu,
            show_labels=show_l,
            cluster=c,
            ax=axes[i],
            square=square,
        )

    return fig


def _plot_loading(df, name, n_top, cut, show_labels, cluster, ax, **kwargs):
    """
    Plot a heatmap representation of a loadings matrix, optionally z-scored and subsetted to the n_top rows of each factor.

    Parameters
    ----------
    df : np.ndarray
        Matrix to plot
    name : str
        Name of factor
    n_top : int
        Number of top features to plot
    cut : float
        Cut-off for z-score color mapping
    show_labels : bool
        Whether to show row labels
    cluster : bool
        Whether to cluster rows
    ax : matplotlib.axes.Axes
        Axes to plot heatmap on
    cbar_ax : matplotlib.axes.Axes
        Axes to plot colorbar on
    kwargs : dict
        Additional keyword arguments to pass to sns.heatmap
    """

    # Optionally z-score each column
    cmap = red_light
    center = None
    vmin = None
    vmax = None
    if df.min().min() < 0:
        cmap = red2blue
        center = 0

        # Optionally set cut-off for z-score color mapping
        if cut:
            vmin = max(-abs(cut), df.min().min())
            vmax = min(abs(cut), df.max().max())

    # Subset to factor
    if n_top:
        top_indices = []
        for col in df.columns:
            top_indices.extend(
                df.sort_values(col, ascending=False).head(n_top).index.tolist()
            )
        df = df.loc[top_indices]

    # Get hierarchical clustering row order
    if cluster:
        row_order = sns.clustermap(df, col_cluster=False).dendrogram_row.reordered_ind
        plt.close()
        df = df.iloc[row_order]

    # Plot heatmap
    sns.heatmap(
        df,
        center=center,
        cmap=cmap,
        cbar_kws=dict(shrink=0.5, aspect=10),
        yticklabels=show_labels,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        rasterized=True,
        **kwargs,
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(f"{name}: [{df.shape[0]} x {df.shape[1]}]")
    sns.despine(ax=ax, right=False, top=False)
