# import proplot as plot

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .._utils import pheno_to_color


def factors(data, axis="cell", palette="Paired", figsize=(6, 6)):
    # Load factor loadings
    if axis == "cell":
        load = data.uns["tensor_loadings"]["Cells"]
    elif axis == "gene":
        load = data.uns["tensor_loadings"]["Genes"]
    elif axis == "feature":
        load = data.uns["tensor_loadings"]["Features"]
    else:
        raise ValueError("Invalid axis specified.")

    # Get cluster labels
    if axis == "cell":
        cluster_labels = data.obs.sort_values("td_cluster")["td_cluster"]
        unit_order = cluster_labels.index.tolist()
    elif axis == "gene":
        cluster_labels = data.var.sort_values("td_cluster")["td_cluster"].dropna()
        unit_order = cluster_labels.index.tolist()
    else:
        cluster_labels = np.zeros(len(load))
        unit_order = load.index

    factor2color, factor_colors = pheno_to_color(load.columns, palette=palette)
    cluster2color, cluster_colors = pheno_to_color(cluster_labels, palette=palette)

    #     clustermap_params = dict(
    #         row_cluster=False,
    #         col_cluster=False,
    #         row_colors=factor_colors,
    #         cmap="Reds",
    #         figsize=figsize,
    #     )

    #     if axis != "feature":
    #         cluster2color, sample_colors = pheno_to_color(cluster_labels, palette=palette)
    #         g = sns.clustermap(
    #             load.loc[unit_order].T,
    #             col_colors=sample_colors,
    #             xticklabels=False,
    #             **clustermap_params,
    #         )
    #     else:
    #         g = sns.clustermap(
    #             load.loc[unit_order].T,
    #             xticklabels=True,
    #             **clustermap_params,
    #         )

    #     plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    #     plt.tight_layout()

    # Plot cluster labels

#     --------------------------------
#     n_factors = load.shape[1]
#     fig, axes = plt.subplots(n_factors, 1)

    # Plot factor loadings
#     for f, ax in zip(load.columns, axes):
#         load_df = pd.DataFrame(
#             [range(load.shape[0]), load.loc[unit_order, f], cluster_labels],
#             index=["x", "y", "hue"],
#         ).T

#         if axis != "feature":
#             sns.scatterplot(
#                 data=load_df,
#                 x="x",
#                 y="y",
#                 hue="hue",
#                 s=8,
#                 linewidth=0,
#                 palette=palette,
#                 legend=False,
#                 ax=ax,
#             )
#             ax.set_xticks([])
#         else:
#             sns.barplot(
#                 x=load.index.fillna("na").tolist(),
#                 y=load_df["y"],
#                 hue=load_df["hue"],
#                 palette=palette,
#                 ax=ax,
#             )
#             ax.get_legend().remove()

#             if ax == axes[-1]:
#                 ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
#             else:
#                 ax.set_xticks([])

#         ax.set_ylabel(f, labelpad=30, rotation=0, weight="600")
#         ax.set_xlabel("")
#         sns.despine()

#     ax.set_xlabel(str(axis).capitalize(), weight="600")
#     plt.tight_layout()

    load["td_cluster"] = cluster_labels
    load_long = load.reset_index().melt(id_vars=['index', "td_cluster"])
    load_long.columns = ["sample", "td_cluster", "factor", "loading"]
    sns.factorplot(data=load_long, kind="bar", x="td_cluster", y="loading", hue="factor")