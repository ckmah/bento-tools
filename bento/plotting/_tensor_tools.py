from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .._utils import TENSOR_DIM_NAMES, PATTERN_COLORS
from ._utils import savefig


def _get_loading(data, load, dim, scale):

    if dim == TENSOR_DIM_NAMES[0]:
        cluster_labels = np.zeros(len(load))
        unit_order = load.index

    elif dim == TENSOR_DIM_NAMES[1]:
        cluster_labels = data.obs["td_cluster"].sort_values().dropna()
        unit_order = cluster_labels.index.tolist()
    elif dim == TENSOR_DIM_NAMES[2]:
        cluster_labels = data.var["td_cluster"].sort_values().dropna()
        unit_order = cluster_labels.index.tolist()

    load_df = pd.DataFrame(
        [
            range(load.shape[0]),
            load.loc[unit_order],
            cluster_labels,
            load.index.tolist(),
        ],
        index=["index", "load", "group", "sample"],
    ).T
    load_df = load_df.astype({"index": int, "load": float, "group": str, "sample": str})

    if scale:
        load_df["load"] = MinMaxScaler().fit_transform(load_df[["load"]])

    load_df["group"] = (
        load_df["group"].str.replace("Factor", "Group").astype("category")
    )
    return load_df


@savefig
def lp_signatures(data, scale=True, fname=None):
    factors = list(data.uns["tensor_loadings"][TENSOR_DIM_NAMES[0]].columns)
    n_factors = len(factors)
    n_dims = len(TENSOR_DIM_NAMES)

    fig = plt.figure(figsize=(3 * n_dims, n_factors))

    # Create grid such that there is space between each row (factor), an extra row above and below everything, and an extra column for legends
    gs = fig.add_gridspec(
        n_factors,
        n_dims + 1,
        width_ratios=[0.5, 1, 1, 0.1],
        hspace=0.2,
        wspace=0.2
    )

    PATTERN_COL = 0
    LEGEND_COL = n_dims
    FACTOR_ROWS = [i for i in range(n_factors)]

    # Plot pattern loadings as barplot
    for factor, row in zip(factors, FACTOR_ROWS):
        if row == FACTOR_ROWS[0]:
            ax = fig.add_subplot(gs[row, PATTERN_COL])
        else:
            ax = fig.add_subplot(gs[row, PATTERN_COL], sharex=fig.axes[0])

        pattern_dim = TENSOR_DIM_NAMES[0]
        load = data.uns["tensor_loadings"][pattern_dim][factor]
        load_df = _get_loading(data, load, pattern_dim, scale=False)

        # Feature barplots
        ax.bar(
            x=load.index.fillna("na").tolist(),
            height=load_df["load"],
            color=PATTERN_COLORS,
            # alpha=0.5,
            # ax=ax,
        )

        if row == FACTOR_ROWS[0]:
            ax.set_title(str(pattern_dim).capitalize(), weight="600")

        # Set row labels
        ax.set_ylabel(factor, labelpad=3.5*len(factor), rotation=0, weight="600")

        # Turn off xlabels except bottom row
        if factor != factors[-1]:
            ax.set(xlabel=None)
        else:
            ax.set(xlabel="loading")

        # Turn off yticks
        ax.set(xticklabels=[])
        ax.tick_params(axis="x", which="both", length=3)

        # Format spines
        sns.despine(ax=ax, left=True)
        ax.spines["bottom"].set_color("#aaaaaa")

    # Plot cell and gene loadings as heatmaps
    for factor, row in zip(factors, FACTOR_ROWS):
        # Populate index 1 and 2 columns
        for col, dim in zip(range(1, n_dims), TENSOR_DIM_NAMES[1:]):

            ax = fig.add_subplot(gs[row, col])
            load = data.uns["tensor_loadings"][dim][factor]
            load_df = _get_loading(data, load, dim, scale)

            # Plot column title if first row
            if row == FACTOR_ROWS[0]:
                ax0 = fig.add_subplot(gs[0, col])
                ax0.set_title(f"{str(dim).capitalize()} ({len(load)})", weight="600")
                ax0.axis("off")

            # Plot colorbar for cell column
            if row == FACTOR_ROWS[0] and col == 1:
                cbar = True
                cbar_ax = fig.add_subplot(gs[FACTOR_ROWS[0], n_dims])
                cmap = 'Purples'
            # Plot colorbar for gene column
            elif row == FACTOR_ROWS[0] and col == 2 and not scale:
                cbar = True
                cbar_ax = fig.add_subplot(gs[FACTOR_ROWS[1], n_dims])
            else:
                cbar = False

            # Set colormap for cell loadings
            if col == 1:
                cmap = 'Purples'
            # Set colormap for gene loadings
            elif col == 2:
                cmap = 'Reds'

            # Colormap limits for zscoring
            if scale:
                vmin = 0
                vmax = 1

                # Plot (z-scored) loadings as heatmap
                sns.heatmap(
                    load_df[["load"]].T,
                    ax=ax,
                    xticklabels=False,
                    yticklabels=False,
                    cmap="RdBu_r",
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                    cbar=cbar,
                    cbar_ax=cbar_ax,
                )
            else:
                vmin = data.uns['tensor_loadings'][dim].min().min()
                vmax = data.uns['tensor_loadings'][dim].max().max()

                # Plot (z-scored) loadings as heatmap
                sns.heatmap(
                    load_df[["load"]].T,
                    ax=ax,
                    xticklabels=False,
                    yticklabels=False,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    cbar=cbar,
                    cbar_ax=cbar_ax,
                )

            # Format heatmap
            ax.tick_params(axis="y", which="both", length=0)
            ax.set(ylabel=None)
            sns.despine(ax=ax, top=False, bottom=False, left=False, right=False)

            
            