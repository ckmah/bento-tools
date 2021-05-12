import warnings
from functools import partial

import datashader as ds
import datashader.transfer_functions as tf
import geopandas
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datashader.mpl_ext import dsshow
from scipy.spatial.distance import cdist

from ..preprocessing import get_points

hv.extension("bokeh", "matplotlib")


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


def gene_umap(data, hue=None, **kwargs):

    umap = data.varm['loc_umap']
    if hue is not None:
        hue_vector = data.var.loc[umap.index, hue]
    else:
        hue_vector = hue
    ax = sns.scatterplot(data=umap, x=0, y=1, hue=hue_vector, **kwargs)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
    plt.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(hue)
    return ax


def spots_distr(data, genes=None, layer="pattern"):
    """Plot localization pattern frequencies for a subset of genes as a ridgeline plot. Default all genes.

    Parameters
    ----------
    data : [type]
        [description]
    """
    if genes is None:
        genes = data.var_names
    else:
        if type(genes) != list:
            genes = [genes]

        genes = list(set(genes))

    cell_freq = (
        data.to_df(layer=layer)
        .loc[:, genes]
        .T.reset_index(drop=True)
        .apply(lambda x: x.value_counts())
        .fillna(0)
    )

    cell_frac_long = (cell_freq / cell_freq.sum()).T.melt()
    cell_frac_long.columns = [layer, "fraction"]

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        pattern_names = cell_frac_long[layer].unique()

        g = sns.FacetGrid(
            cell_frac_long,
            row=layer,
            hue=layer,
            sharey=False,
            aspect=10,
            height=0.75,
            palette=sns.color_palette(),
            xlim=(-0.1, 1.1),
        )
        # subplot_kws=dict(facecolor=(0,0,0,0))
        # then we add the densities kdeplots for each pattern
        g.map(
            sns.kdeplot,
            "fraction",
            clip_on=False,
            fill=True,
            alpha=1,
            linewidth=1.5,
        )

        # here we add a white line that represents the contour of each kdeplot
        g.map(sns.kdeplot, "fraction", clip_on=False, color="w", lw=2)

        # here we add a horizontal line for each plot
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        # we loop over the FacetGrid figure axes (g.axes.flat) and add the pattern as text with the right color
        # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
        for i, ax in enumerate(g.axes.flat):
            ax.text(
                -0.5,
                0,
                pattern_names[i],
                # fontweight="bold",
                fontsize=14,
                color=ax.lines[-1].get_color(),
            )

        # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
        g.fig.subplots_adjust(hspace=-0.25)

        # eventually we remove axes titles, yticks and spines
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)

        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.xlabel("Gene fraction", fontsize=15)
        g.fig.suptitle(
            "Localization pattern distribution across cells",
            ha="center",
            fontsize=18,
            fontweight=20,
        )

        return g


def plot_cells(
    data,
    kind="scatter",
    cells=None,
    genes=None,
    markersize=3,
    alpha=0.2,
    hue=None,
    col=None,
    cmap="Blues",
    masks="all",
    binwidth=3,
    spread=True,
    width=10
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
        masks = [f"{m}_shape" for m in masks]

    # Convert draw_masks to list
    masks = [masks] if type(masks) is str else masks

    # Subset adata info by genes and cells
    data = data[cells, genes]
    points = get_points(data, cells=cells, genes=genes)
    points["gene"] = points["gene"].map(data.uns["point_gene_index"])
    points = geopandas.GeoDataFrame(
        points, geometry=geopandas.points_from_xy(points["x"], points["y"])
    )

    if hue:
        points[hue] = points[hue].astype("category")

    # Get masks and points
    shapes = geopandas.GeoDataFrame(data.obs[masks], geometry="cell_shape")
    masks = shapes.columns.tolist()

    bounds = shapes.bounds
    minx, miny, maxx, maxy = (
        np.floor(bounds["minx"].min()),
        np.floor(bounds["miny"].min()),
        np.ceil(bounds["maxx"].max()),
        np.ceil(bounds["maxy"].max()),
    )

    if not col:
        ncols = 1
        col_names = [""]
        fig_width = width
        fig_height = width
    else:
        ncols = points[col].nunique()
        col_names = points[col].unique()
        fig_width = width * ncols
        fig_height = width

    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_height))

    try:
        iterator = iter(axes)
    except TypeError:
        axes = [axes]
    else:
        axes = list(axes)

    for ax, c_name in zip(axes, col_names):

        if col:
            points_c = points.loc[points[col] == c_name]
        else:
            points_c = points

        # Plot mask outlines
        for mask in masks:
            shapes.set_geometry(mask).plot(
                color=(0, 0, 0, 0.05), edgecolor=(0, 0, 0, 0.3), ax=ax
            )

        # Plot points
        if kind == "scatter":
            col_value = None
            if hue:
                col_value = hue

            points_c.plot(
                column=col_value,
                markersize=markersize,
                alpha=0.5,
                cmap=cmap,
                legend=True,
                ax=ax,
            )

        elif kind == "dist_scatter":
            g1, g2 = points_c["gene"].unique()
            g1_points = points_c[points_c["gene"] == g1]
            g2_points = points_c[points_c["gene"] == g2]

            d_matrix = cdist(g1_points[["x", "y"]], g2_points[["x", "y"]])
            g1_points["min_dist"] = d_matrix.min(axis=1)
            g2_points["min_dist"] = d_matrix.min(axis=0)
            points_c = pd.concat([g1_points, g2_points])
            points_c = points_c.sort_values("min_dist", ascending=False)
            points_c.plot(
                column="min_dist",
                # edgecolor='black',
                # linewidth=0.5,
                markersize=[
                    markersize * max(v, 1)
                    for v in (-np.log(points_c["min_dist"]) / np.log(3) + 3)
                ],
                alpha=alpha,
                cmap=sns.cubehelix_palette(
                    start=0.5,
                    rot=0.2,
                    dark=0.3,
                    hue=1,
                    light=0.85,
                    reverse=True,
                    as_cmap=True,
                ),
                vmax=3,
                ax=ax,
            )

        elif kind == "hist":
            if hue:
                agg = ds.by(hue, ds.count())
            else:
                agg = ds.count()

            scaled_binw = ax.get_window_extent().width * binwidth / (maxx - minx)
            scaled_binh = ax.get_window_extent().height * binwidth / (maxy - miny)

            if spread:
                spread = partial(tf.dynspread, threshold=0.5)
            else:
                spread = None

            partist = dsshow(
                points_c,
                ds.Point("x", "y"),
                aggregator=agg,
                # agg_hook=lambda x: x.where((x.sel(gene=genes[0]) > 0) & (x.sel(gene=genes[1]) > 0)),
                norm="linear",
                cmap=cmap,
                width_scale=1 / scaled_binw,
                height_scale=1 / scaled_binh,
                shade_hook=spread,
                x_range=(minx, maxx),
                y_range=(miny, maxy),
                ax=ax,
            )

            if hue is None:
                plt.colorbar(
                    partist,
                    orientation="horizontal",
                    fraction=0.05,
                    aspect=10,
                    pad=0.02,
                    ax=ax,
                )

        ax.set_title(c_name)
        ax.axis("off")

    plt.close()
    return fig


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
