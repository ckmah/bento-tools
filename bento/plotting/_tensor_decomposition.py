from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .._utils import pheno_to_color, DIM_COLORS, PATTERN_COLORS
from ._utils import savefig


def _get_loading(data, load, dim, zscore):

    if dim == "Patterns":
        cluster_labels = np.zeros(len(load))
        unit_order = load.index

    elif dim == "Cells":
        cluster_labels = data.obs["td_cluster"].sort_values().dropna()
        unit_order = cluster_labels.index.tolist()
    elif dim == "Genes":
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

    if zscore:
        load_df["load"] = StandardScaler().fit_transform(load_df[["load"]])

    load_df["group"] = (
        load_df["group"].str.replace("Factor", "Group").astype("category")
    )
    return load_df


@savefig
def factors(data, zscore=False, fname=None):
    dims = ["Patterns", "Cells", "Genes"]
    factors = list(data.uns["tensor_loadings"][dims[0]].columns)
    n_factors = len(factors)
    n_dims = len(dims)

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

        load = data.uns["tensor_loadings"]["Patterns"][factor]
        load_df = _get_loading(data, load, "Patterns", zscore=False)

        # Feature barplots
        ax.bar(
            x=load.index.fillna("na").tolist(),
            height=load_df["load"],
            color=PATTERN_COLORS,
            # alpha=0.5,
            # ax=ax,
        )

        if row == FACTOR_ROWS[0]:
            ax.set_title("Patterns", weight="600")

        # Set row labels
        ax.set_ylabel(factor, labelpad=30, rotation=0, weight="600")

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
        for col, dim in zip(range(1, n_dims), ["Cells", "Genes"]):

            ax = fig.add_subplot(gs[row, col])
            load = data.uns["tensor_loadings"][dim][factor]
            load_df = _get_loading(data, load, dim, zscore)

            # Plot column title if first row
            if row == FACTOR_ROWS[0]:
                ax0 = fig.add_subplot(gs[0, col])
                ax0.set_title(f"{str(dim).capitalize()} ({len(load)})", weight="600")
                ax0.axis("off")

            # Plot colorbar for first heatmap
            if row == FACTOR_ROWS[0] and col == 1:
                cbar = True
                cbar_ax = fig.add_subplot(gs[FACTOR_ROWS[0], n_dims])
            else:
                cbar = False
                cbar_ax = None

            # Colormap limits for zscoring
            if zscore:
                vmin = -3
                vmax = 3
            else:
                vmin = None
                vmax = None

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

            # Format heatmap
            ax.tick_params(axis="y", which="both", length=0)
            ax.set(ylabel=None)
            sns.despine(ax=ax, top=False, bottom=False, left=False, right=False)
