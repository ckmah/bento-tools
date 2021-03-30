import warnings

import altair as alt
import descartes
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, is_color_like
from shapely import geometry
from shapely.affinity import translate

from ..tools._tools import subsample_points
from ..io import get_points


# Masala color palette by Noor
# Note: tested colorblind friendliness, did not do so well
masala_palette = sns.color_palette(
    [
        (1.0, 0.78, 0.27),
        (1.0, 0.33, 0.22),
        (0.95, 0.6, 0.71),
        (0, 0.78, 0.86),
        (0.39, 0.81, 0.63),
        (0.25, 0.25, 0.25),
    ]
)


def spots_diff(data, groups):
    dl_results = data.uns["sample_data"][f"dl_{groups}"]
    # with sns.axes_style("whitegrid"):
    g = sns.relplot(
        data=dl_results.sort_values("pattern"),
        x="dy/dx",
        y="-log10padj",
        hue="pattern",
        size="-log10padj",
        sizes=(10, 200),
        col="phenotype",
        s=4,
        height=4,
        linewidth=0,
        palette="tab10",
    )
    for ax in g.axes.flatten():
        ax.grid(False, axis="y")
        ax.axhline(-np.log10(0.05), color="gray", lw=1, linestyle="--")
        ax.axvline(0, color="black", lw=1, linestyle="-")
        sns.despine(top=False, right=False, bottom=False, left=False)
        plt.xlim(-1, 1)
    plt.ylim(-0.05, 3)


def spots_freq(data, gene=None, groups=None, relative=True, stacked=False):
    """Plot SPOTS localization pattern frequencies for a given gene.

    Parameters
    ----------
    data : [type]
        [description]
    gene : str
        Gene name to show frequencies.
    groups : str, optional
        Sample category to stratify frequencies, by default None.
    relative : bool, optional
        Whether to calculate relative fractions or absolute number of cells, by default True.
    stacked : bool, optional
        Whether to use a single plot or multiple plots for each category, by default False.
    """
    if gene is None:
        gene_mask = ~data.uns["sample_index"]["gene"].isna()
    else:
        gene_mask = (data.uns["sample_index"]["gene"] == gene).values

    if sum(gene_mask) == 0:
        print(f"Gene {gene} not found.")
        return

    classes = data.uns["sample_data"]["patterns"].columns.tolist()

    if groups:
        pattern_fracs = pd.concat(
            [
                data.uns["sample_data"]["patterns"].reset_index(drop=True),
                data.uns["sample_data"][groups].reset_index(drop=True),
            ],
            axis=1,
        ).loc[gene_mask]

        if relative:
            pattern_fracs = pattern_fracs.groupby(groups).apply(
                lambda df: df[classes].sum() / df.shape[0]
            )
        else:
            pattern_fracs = pattern_fracs.groupby(groups).apply(
                lambda df: df[classes].sum()
            )

        pattern_fracs = pattern_fracs.reset_index()

    else:
        pattern_fracs = (
            data.uns["sample_data"]["patterns"].reset_index(drop=True).loc[gene_mask]
        )

        if relative:
            pattern_fracs = (
                (pattern_fracs[classes].sum() / pattern_fracs.shape[0]).to_frame().T
            )

        else:
            pattern_fracs = pattern_fracs[classes].sum().to_frame().T

    # Calculate polar angles
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    stats = pd.concat((pattern_fracs[classes], pattern_fracs[[classes[0]]]), axis=1)

    if groups:
        stats = pd.concat([stats, pattern_fracs[groups]], axis=1)

    def polar_gene(row, ax, color=None):
        with sns.axes_style("whitegrid"):

            # Plot gene trace
            ax.plot(angles, row, color=color)
            ax.fill(angles, row, facecolor=color, alpha=0.2)

            # Set theta axis labels
            ax.set_thetagrids(angles * 180 / np.pi, classes + [classes[0]])
            ax.tick_params(axis="x", labelsize="small")

            # Format r ticks
            ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="lower"))
            ax.tick_params(axis="y", labelsize="x-small")

            ax.grid(True, linestyle=":")

    if groups:
        ngroups = stats[groups].nunique()
        if stacked:
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)

            for name, row in stats.iterrows():
                row = row[row.index != groups]
                polar_gene(row, ax)

            labels = stats[groups].values.tolist()
            ax.legend(
                labels,
                loc=(0.9, 0.95),
                labelspacing=0.1,
                fontsize="small",
                frameon=False,
            )

        else:
            fig, axs = plt.subplots(
                1,
                ngroups,
                sharex=True,
                subplot_kw=dict(polar=True),
                figsize=(4 * ngroups, 3),
            )
            fig.subplots_adjust(wspace=0.3)

            for ax, (name, row) in zip(axs, stats.iterrows()):
                ax.set_title(row[groups])
                row = row[row.index != groups]
                polar_gene(row, ax)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        polar_gene(stats.iloc[0], ax)
        ax.set_title(gene)


def plot_cells(
    data,
    kind="points",
    cells=None,
    genes=None,
    fraction=0.1,
    hue=None,
    palette="Reds",
    size=3,
    alpha=1,
    masks="all",
    bw_adjust=1,
    dim=5,
    ax=None
):
    """
    Visualize distribution of variable in spatial coordinates.
    Parameters
    ----------
    data : dict
    type : str
        Options include 'points' and 'heatmap'
    cells: None, list
        Select specified list of cells. Default value of None selects all cells.
    genes: None, list
        Select specified list of genes. Default value of None selects all genes.
    fraction: float (0,1]
        Fraction to fraction when plotting. Useful when dealing with large datasets.
    scatter_hue: str
        Name of column in data.obs to interpet for scatter plot color. First tries to interpret if values are matplotlib color-like. For categorical data, tries to plot a different color for each label. For numerical, scales color by numerical values (need to be between [0,1]). Returns error if outside range.
    draw_masks: str, list
       masks to draw outlines for. Will always include outline for union of `masks`.
    """

    # Format cells input
    if cells is None:
        cells = data.obs.index.unique().tolist()
    else:
        # print('Subsetting cells...')
        if type(cells) != list:
            cells = [cells]

        cells = list(set(cells))

    if "-1" in cells:
        warnings.warn(
            "Detected points outside of cells. TODO write clean fx that drops these points"
        )

    # Format genes input
    if genes is None:
        genes = data.var_names
    else:
        # print('Subsetting genes...')
        if type(genes) != list:
            genes = [genes]

        genes = list(set(genes))

    # Add all masks if 'all'
    if masks == "all":
        masks = data.obs.columns[data.obs.columns.str.endswith("shape")]
    else:
        masks = [f'{m}_shape' for m in masks]

    # Convert draw_masks to list
    masks = [masks] if type(masks) is str else masks

    # Initialize figure
    if ax is None:
        fig = plt.figure(figsize=(dim, dim), dpi=100)
        ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    data = data[cells, genes]
    # Subset masks to specified cells
    cell_shapes = geopandas.GeoSeries(data.obs.loc[cells, "cell_shape"])

    # Plot clipping mask
    clip = cell_shapes.unary_union
    clip = clip.envelope.symmetric_difference(clip)
    clip_patch = descartes.PolygonPatch(clip, color="white")
    ax.add_patch(clip_patch)

    bounds = clip.bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # * Plot mask faces
    for mask in masks:
        if mask == "cell_shape":
            geopandas.GeoDataFrame(data.obs.loc[cells]).set_geometry("cell_shape").plot(
                ax=ax, facecolor="white", lw=0, alpha=1
            )
        else:
            geopandas.GeoDataFrame(data.obs.loc[cells]).set_geometry("nucleus_shape").plot(
                ax=ax, facecolor="tab:blue", lw=0, alpha=0.3
            )

    # Subset points to specified cells and genes
    points = get_points(data, cells=cells, genes=genes)

    # * fraction points per cell
    if fraction < 1:
        # print('Downsampling points...')
        points = subsample_points(points, fraction)

    # * Plot raw points
    if kind == "points":
        sns.scatterplot(
            data=points,
            x="x",
            y="y",
            hue=hue,
            s=size,
            alpha=alpha,
            linewidth=0,
            palette=palette,
            ax=ax,
        )

    # * Plot heatmap
    elif kind == "heatmap":
        sns.kdeplot(
            data=points,
            x="x",
            y="y",
            cmap=sns.color_palette(palette, as_cmap=True),
            bw_adjust=bw_adjust,
            thresh=None,
            levels=50,
            fill=True,
            alpha=alpha,
            ax=ax,
        )

    # Plot mask outlines
    for mask in masks:
        if mask == "cell_shape":
            geopandas.GeoDataFrame(data.obs).set_geometry("cell_shape").boundary.plot(
                ax=ax, lw=0.5, edgecolor="black"
            )
        else:
            geopandas.GeoDataFrame(data.obs).set_geometry("nucleus_shape").boundary.plot(
                ax=ax, lw=0.5, edgecolor="tab:blue"
            )

    return ax



# plot = Petal(bounds=[np.repeat(0,6), np.repeat(1.,6)], figsize=(30,5), cmap='Set1', labels=sf_pred_prob.columns.sort_values().tolist())
# plot.add(sf_pred_prob[sf_pred_prob.columns.sort_values()].iloc[:6].values)
# plot.show()


def pheno_to_color(pheno, palette):
    """
    Maps list of categorical labels to a color palette.
    Input values are first sorted alphanumerically least to greatest before mapping to colors. This ensures consistent colors regardless of input value order.

    Parameters
    ----------
    pheno : pd.Series
        Categorical labels to map
    palette: None, string, or sequence, optional
        Name of palette or None to return current palette. If a sequence, input colors are used but possibly cycled and desaturated. Taken from sns.color_palette() documentation.

    Returns
    -------
    dict
        Mapping of label to color in RGBA
    tuples
        List of converted colors for each sample, formatted as RGBA tuples.

    """
    if type(palette) is str:
        palette = sns.color_palette(palette)
    else:
        palette = palette

    values = list(set(pheno))
    values.sort()
    palette = sns.color_palette(palette, n_colors=len(values))
    study2color = dict(zip(values, palette))
    sample_colors = [study2color[v] for v in pheno]
    return study2color, sample_colors
