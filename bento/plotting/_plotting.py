import warnings
from functools import partial

import datashader as ds
import datashader.transfer_functions as tf
import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as plot
import seaborn as sns
from datashader.mpl_ext import dsshow
from matplotlib.colors import ListedColormap

from ..preprocessing import get_points
from ..tools import PATTERN_NAMES

matplotlib.rcParams["figure.facecolor"] = (0, 0, 0, 0)

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


def spots_diff(data, groups, adjusted=True, ymax=None):
    dl_results = data.uns[f"diff_{groups}"]

    if adjusted:
        rank = "-log10padj"
    else:
        rank = "-log10p"

    g = sns.relplot(
        data=dl_results.sort_values("pattern"),
        x="dy/dx",
        y=rank,
        hue="pattern",
        size=rank,
        sizes=(1, 20),
        col="phenotype",
        height=4,
        linewidth=0,
        palette="tab10",
        facet_kws=dict(xlim=(-1, 1)),
    )

    g.map(plt.axhline, y=-np.log10(0.05), color="gray", lw=1, linestyle="--", zorder=0)
    (
        g.map(plt.axvline, x=0, color="black", lw=1, linestyle="-", zorder=0)
        .set_axis_labels("Marginal Effect (dy/dx)", f"Significance ({rank})")
        .set_titles("{col_name}")
    )
    # g.map(sns.despine, top=False, right=False, bottom=False, left=False)


def gene_umap(data, hue=None, **kwargs):

    umap = data.varm["loc_umap"]
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


def spots_all_distr(data, groupby=None, layer="pattern", stacked=True, legend=True):

    if groupby:
        pattern_distr = (
            data.to_df(layer)
            .groupby(data.obs[groupby])
            .apply(lambda df: pd.Series(df.values.flatten()).value_counts())
        )
        pattern_distr = (pattern_distr / pattern_distr.sum(level=0)).reset_index()

    else:
        groupby = "Sample"
        pattern_distr = pd.Series(data.to_df(layer).values.flatten()).value_counts()
        pattern_distr = (
            (pattern_distr / pattern_distr.sum()).reset_index().reset_index()
        )
        pattern_distr.iloc[:, 0] = "Sample"

    pattern_distr.columns = [groupby, layer, "value"]
    # pattern_distr["value"] *= 100
    pattern_distr = pattern_distr.pivot(index=groupby, columns=layer, values="value")
    pattern_distr = pattern_distr.reindex(columns=PATTERN_NAMES, fill_value=0)
    # pattern_distr.drop("none", axis=1, inplace=True)

    with sns.axes_style(style="white"):
        ax = pattern_distr.plot(
            kind="bar",
            stacked=stacked,
            colormap=ListedColormap(sns.color_palette("muted6", as_cmap=True)),
            width=0.8,
            lw=0,
            figsize=(max(2, pattern_distr.shape[0] / 2), 4),
            # xlim=(0, 100),
            legend=False,
        )

        if pattern_distr.shape[0] == 1:
            ax.set_xlabel("")
        ax.set_ylabel("Total Fraction")
        if legend:
            ax.legend(bbox_to_anchor=(1, 1))
        sns.despine()
        plt.tight_layout()

    return ax


def spots_distr(data, level="cells", sharey=True, binwidth=10, layer="pattern"):
    """Plot localization pattern distributions across all cells (default) or genes.

    Parameters
    ----------
    data : [type]
        [description]
    """
    if level == "cells":
        axis = 1
    elif level == "genes":
        axis = 0

    # Count pattern occurences within axis
    pattern_distr = data.to_df("pattern").apply(
        lambda fiber: fiber.value_counts(), axis=axis
    )

    if axis == 0:
        pattern_distr = pattern_distr.T

    pattern_distr = ((pattern_distr.T * 100) / pattern_distr.sum(axis=1)).T

    distr_long = pattern_distr.drop("none", axis=1)
    distr_long["group"] = None
    distr_long = distr_long.melt(id_vars="group")
    distr_long.columns = ["group", "pattern", "value"]

    palette = sns.color_palette("muted6")
    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):

        pattern_names = distr_long["pattern"].unique()

        g = sns.FacetGrid(
            distr_long,
            col="pattern",
            hue="pattern",
            col_wrap=3,
            sharey=sharey,
            sharex=False,
            aspect=1.5,
            height=2,
            palette=palette,
            xlim=(0, 100),
        )
        # subplot_kws=dict(facecolor=(0,0,0,0))
        # then we add the densities kdeplots for each pattern
        g.map(
            sns.histplot,
            "value",
            stat="count",
            binwidth=binwidth,
            clip_on=False,
            alpha=1,
        )

        # here we add a horizontal line for each plot
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        if level == "cells":
            x_unit = "genes"
        else:
            x_unit = "cells"

        # we loop over the FacetGrid figure axes (g.axes.flat) and add the pattern as text with the
        # right color notice how ax.lines[-1].get_color() enables you to access the last line's
        # color in each matplotlib.Axes
        for i, ax in enumerate(g.axes.flat):
            ax.set_title(
                pattern_names[i],
                # fontweight="bold",
                fontsize=15,
                color=ax.lines[-1].get_color(),
            )
            ax.set_xlabel(f"Percent of {x_unit}", fontsize=15)

        # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
        g.fig.subplots_adjust(hspace=0.4)

        # eventually we remove axes titles, yticks and spines
        g.set_titles("")
        plt.setp(ax.get_xticklabels(), fontsize=15)

        g.fig.suptitle(
            f"Localization pattern distribution across {level}",
            ha="center",
            fontsize=18,
            fontweight=20,
        )

        plt.tight_layout()
    return g


def plot_cells(
    data,
    kind="scatter",
    hue=None,
    tile=False,
    cells=None,
    genes=None,
    pattern=None,
    markersize=3,
    alpha=1,
    cmap="blues",
    ncols=4,
    masks="all",
    binwidth=3,
    spread=False,
    legend=False,
    cbar=True,
    frameon=True,
    size=4,
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
    draw_masks: str, list
       masks to draw outlines for.
    """

    # Format cells input
    if cells is None:
        cells = data.obs.index.unique().tolist()
    else:
        cells = list(cells)

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

    if pattern:
        points = points.loc[points["pattern"] == pattern]

    points = geopandas.GeoDataFrame(
        points, geometry=geopandas.points_from_xy(points["x"], points["y"])
    )

    # Get masks and points
    shapes = geopandas.GeoDataFrame(data.obs[masks], geometry="cell_shape")
    masks = shapes.columns.tolist()

    # Plot each cell in separate subplots
    if tile:
        ncols = min(ncols, len(shapes))
        nrows = max(1, int(np.ceil(len(shapes) / ncols)))

        # Determine fixed radius of each subplot
        cell_bounds = shapes.bounds
        cell_maxw = (cell_bounds["maxx"] - cell_bounds["minx"]).max()
        cell_maxh = (cell_bounds["maxy"] - cell_bounds["miny"]).max()
        ax_radius = 1.1 * (max(cell_maxw, cell_maxh) / 2)

        # Create subplots
        fig, axs = plot.subplots(
            nrows=nrows, ncols=ncols, sharex=False, sharey=False, axwidth=size, space=0
        )
        axs.format(xticks=[], yticks=[])
        axs.axis(frameon)

        # Plot each cell separately
        for i, ax in enumerate(axs):
            try:
                s = shapes.iloc[[i]]
                p = points.loc[points["cell"] == s.index[0]]

                if i == 0:
                    legend = legend
                else:
                    legend = False
                _plot_cells(
                    masks,
                    s,
                    ax,
                    kind,
                    p,
                    markersize,
                    alpha,
                    binwidth,
                    spread,
                    hue,
                    cmap,
                    size,
                    legend,
                    cbar,
                )

                s_bound = s.bounds
                centerx = np.mean([s_bound["minx"], s_bound["maxx"]])
                centery = np.mean([s_bound["miny"], s_bound["maxy"]])
                ax.set_xlim(centerx - ax_radius, centerx + ax_radius)
                ax.set_ylim(centery - ax_radius, centery + ax_radius)

            except (IndexError, ValueError):
                ax.remove()

    else:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))
        plt.setp(ax, xticks=[], yticks=[])
        ax.axis(frameon)

        _plot_cells(
            masks,
            shapes,
            ax,
            kind,
            points,
            markersize,
            alpha,
            binwidth,
            spread,
            hue,
            cmap,
            size,
            legend,
            cbar,
        )

    if pattern:
        ax.set_title(pattern)

    # plt.close(fig)
    return fig


def _plot_cells(
    masks,
    shapes,
    ax,
    kind,
    points_c,
    markersize,
    alpha,
    binwidth,
    spread,
    hue,
    cmap,
    size,
    legend,
    cbar,
):
    # Plot mask outlines
    for mask in masks:
        shapes.set_geometry(mask).plot(
            color=(0, 0, 0, 0.05), edgecolor=(0, 0, 0, 0.3), ax=ax
        )

    if points_c.shape[0] == 0:
        return ax

    if hue == "pattern" and "none" in points_c[hue].cat.categories:
        points_c[hue].cat.remove_categories("none", inplace=True)
        cmap = dict(
            zip(
                [
                    "cell_edge",
                    "foci",
                    "nuclear_edge",
                    "perinuclear",
                    "protrusions",
                    "random",
                ],
                sns.color_palette("muted6", n_colors=6).as_hex(),
            )
        )

    # Plot points
    if kind == "scatter":
        if hue is None:
            points_c.plot(
                column=hue,
                markersize=markersize,
                color=cmap,
                alpha=alpha,
                ax=ax,
            )
        else:
            points_c.plot(
                column=hue,
                markersize=markersize,
                cmap=cmap,
                alpha=alpha,
                ax=ax,
            )

    # Plot density
    elif kind == "hist":

        if hue:
            aggregator = ds.count_cat(hue)
        else:
            aggregator = ds.count()

        if spread:
            spread = partial(tf.dynspread, threshold=0.5)
        else:
            spread = None

        # Percent of window size
        scaled_binwidth = size * binwidth * 0.5

        artist = dsshow(
            points_c,
            ds.Point("x", "y"),
            aggregator,
            norm="linear",
            cmap=cmap,
            color_key=cmap,
            width_scale=1 / scaled_binwidth,
            height_scale=1 / scaled_binwidth,
            vmin=0,
            shade_hook=spread,
            ax=ax,
        )

        if legend:
            plt.legend(handles=artist.get_legend_elements())

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            ax.figure.colorbar(artist, cax=cax, orientation="vertical")
        # plt.tight_layout()

    bounds = shapes.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    return ax


def pheno_to_color(pheno, palette):
    """
    Maps list of categorical labels to a color palette.
    Input values are first sorted alphanumerically least to greatest before mapping to colors.
    This ensures consistent colors regardless of input value order.

    Parameters
    ----------
    pheno : pd.Series
        Categorical labels to map
    palette: None, string, or sequence, optional
        Name of palette or None to return current palette.
        If a sequence, input colors are used but possibly cycled and desaturated.
        Taken from sns.color_palette() documentation.

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
