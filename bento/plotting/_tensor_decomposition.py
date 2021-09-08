import proplot as plot

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

def factors(data, axis="cell"):
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
        cluster_labels = LabelEncoder().fit_transform(
            data.obs.sort_values("td_cluster")["td_cluster"]
        )
        unit_order = data.obs.sort_values("td_cluster").index.tolist()
    elif axis == "gene":
        cluster_labels = LabelEncoder().fit_transform(
            data.var.sort_values("td_cluster")["td_cluster"]
        )
        unit_order = data.var.sort_values("td_cluster").index.tolist()
    else:
        unit_order = load.index

    # Plot cluster labels
    if axis != "feature":
        fig, ax = plot.subplots(axwidth=3, axheight=0.25, hspace=0)
        ax[0].heatmap(
            [cluster_labels] * 2,
            cmap="Set1",
        )
        ax.format(
            xticks="null",
            yticks="null",
            leftlabels=["Clusters    "],
            title=f"{axis.capitalize()} Loadings",
        )

    # Plot factor loadings
    fig, axes = plot.subplots(
        ncols=1, nrows=load.shape[1], axwidth=3, axheight=0.5, hspace="1em"
    )

    for f, ax in zip(load.columns, axes):
        ax.bar(range(load.shape[0]), load.loc[unit_order,f], linewidth=0)
        sns.despine()
    #     break
    axes.format(xticks="null",
#                 yticks="null",
                leftlabels=load.columns.tolist()
               )
