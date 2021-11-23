from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .._utils import pheno_to_color, DIM_COLORS
from ._utils import savefig


def _get_loading(data, load, dim, zscore):

    if dim == "Features":
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
        index=["index", "load", "group", "cell"],
    ).T
    load_df = load_df.astype({"index": int, "load": float, "group": str, "cell": str})
    
    if zscore:
        load_df["load"] = StandardScaler().fit_transform(load_df[["load"]])
        
    load_df["group"] = (
        load_df["group"].str.replace("Factor", "Group").astype("category")
    )
    return load_df


@savefig
def factors(data, zscore=False, fname=None):
    dims = list(data.uns["tensor_loadings"].keys())
    factors = list(data.uns["tensor_loadings"][dims[0]].columns)
    n_factors = len(factors)
    n_dims = len(dims)

    fig = plt.figure(figsize=(3*n_dims,2*n_factors))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 10])
    gs0 = gs[0].subgridspec(n_factors, 1)
    gs1 = gs[1].subgridspec(n_factors * 3, n_dims - 1, height_ratios=[1,3,1] * n_factors)
    # Plot feature loadings
    for row, factor in enumerate(factors):

        if row == 0:
            ax = fig.add_subplot(gs0[row, 0])
        else:
            ax = fig.add_subplot(gs0[row, 0], sharex=fig.axes[0])

        load = data.uns["tensor_loadings"]["Features"][factor]
        load_df = _get_loading(data, load, "Features", zscore=False)

        # Feature barplots
        sns.barplot(
            y=load.index.fillna("na").tolist(),
            x=load_df["load"],
            color=DIM_COLORS[0],
            # alpha=0.5,
            ax=ax,
        )

        if row == 0:
            ax.set_title("Features", weight="600")
        
        # Set row labels
        ax.set_ylabel(factor, labelpad=30, rotation=0, weight="600")

        # Turn off xlabels except bottom row
        if factor != factors[-1]:
            ax.set(xlabel=None)
        else:
            ax.set(xlabel="loading")

        # Turn off yticks
        ax.tick_params(axis="y", which="both", length=0)
        
        # Format spines
        sns.despine(ax=ax, left=True)
        ax.spines['bottom'].set_color('#aaaaaa')

        # Plot cell and gene loadings
        for gs1_col, dim in zip(range(0, n_dims - 1), dims[1:]):
            row_adjusted = (row + 1) * 3 - 2
            ax = fig.add_subplot(gs1[row_adjusted, gs1_col])

            load = data.uns["tensor_loadings"][dim][factor]
            load_df = _get_loading(data, load, dim, zscore)

            if row == n_factors - 1:
                cbar=True
                cbar_ax=fig.add_subplot(gs1[row_adjusted+1, gs1_col])
            else:
                cbar=False
                cbar_ax=None
            
            if zscore:
                vmin=-3
                vmax=3
            else:
                vmin=None
                vmax=None
            
            sns.heatmap(
                load_df[["load"]].T,
                ax=ax,
                xticklabels=False,
                yticklabels=False,
                cbar_kws=dict(orientation="horizontal"),
                cmap="RdBu_r",
                center=0,
                vmin=vmin,
                vmax=vmax,
                cbar=cbar,
                cbar_ax=cbar_ax
            )
            
            ax.tick_params(axis="y", which="both", length=0)
            ax.set(ylabel=None)
            sns.despine(ax=ax, top=False, bottom=False, left=False, right=False)
            
            if row == 0:
                ax = fig.add_subplot(gs1[row, gs1_col])
                ax.set_title(str(dim).capitalize(), weight="600")
                ax.axis("off")
            
            # sns.violinplot(
            #     data=load_df,
            #     x="group",
            #     y="load",
            #     # hue="group",
            #     linewidth=1,
            #     dodge=False,
            #     color=violin_color,
            #     # palette=palette,
            #     ax=ax,
            # )
            # sns.stripplot(
            #     data=load_df,
            #     x="group",
            #     y="load",
            #     color="0.1",
            #     # jitter=0.2,
            #     alpha=0.5,
            #     s=2,
            #     linewidth=0,
            #     ax=ax,
            # )
            # ax.get_legend().remove()

    # axes = np.array(fig.get_axes()).reshape(n_factors, n_dims)
    # for col, dim in enumerate(dims):
    #     axes[0][col].set_title(str(dim).capitalize(), weight="600")
