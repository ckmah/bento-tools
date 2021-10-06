import warnings

warnings.filterwarnings("ignore")

from functools import partial


ds = None
tf = None
geopandas = None
plot = None
dsshow = None

import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from matplotlib.colors import ListedColormap

from ..preprocessing import get_points
from ..tools import PATTERN_NAMES

matplotlib.rcParams["figure.facecolor"] = (0, 0, 0, 0)


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


def cell_patterns(data, cells=None, fraction=True):
    """Plot pattern frequency across cells.

    The mean and 95% confidence interval for all cells is denoted with black points and grey boxes.
    The blue points denote the mean frequencies for the specified subset of cell(s). 
    Parameters
    ----------
    data : spatial formatted AnnData
    cells : list of str, optional
        A list of cell names, by default None. The mean frequencies are plotted as blue points.
    fraction : bool, optional
        Whether to normalize pattern frequencies as a fraction, by default True.
    Returns
    -------
    Axes
        Returns a matplotlib Axes containing plotted figure elements.
    """
    pattern_freq = data.obs[PATTERN_NAMES]

    if fraction:
        pattern_freq /= data.n_vars
        
    # Calculate mean and 95% confidence interval
    stats = pattern_freq.apply(
        lambda pattern: [pattern.mean(), *np.percentile(pattern, [2.5, 97.5])]
    )
    stats.index = ["mean", "ci_lower", "ci_upper"]
    stats = stats.T

    stats = stats.sort_values("mean", ascending=True)

    # Plot mean as points
    ax = plt.gca()
    base_size=7
    plt.scatter(x=stats["mean"], y=stats.index, c="grey", s=base_size ** 2, zorder=100)

    # Plot CI as grey box
    plt.hlines(
        y=range(stats.shape[0]),
        xmin=stats["ci_lower"],
        xmax=stats["ci_upper"],
        color="grey",
        alpha=0.1,
        linewidth=2 * base_size,
        zorder=1,
    )

    # Values to compare
    if cells is not None:
        stats["value"] = data.obs.loc[cells, PATTERN_NAMES].mean()

        if fraction:
            stats["value"] /= data.n_vars
        plt.scatter(x=stats["value"], y=stats.index, c="skyblue", s=base_size ** 2, zorder=100)

        # Stems
        plt.hlines(
            y=range(stats.shape[0]),
            xmin=stats["mean"],
            xmax=stats["value"],
            color="skyblue",
            alpha=1,
            linewidth=1,
            zorder=10,
        )

    # styling
    if fraction:
        plt.xlim(0, 1)
        plt.xlabel(f'Fraction ({data.n_vars} genes)')
    else:
        plt.xlim(0, data.n_vars)
        plt.xlabel(f'Frequency ({data.n_vars} genes)')
    ax.set_title(f"n = {data.n_obs} cells")

    ax.tick_params(axis="y", length=0)
    sns.despine(left=True)

    return ax


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

    global ds, tf, plot, dsshow

    if ds is None:
        import datashader as ds

    if tf is None:
        import datashader.transfer_functions as tf

    if plot is None:
        import proplot as plot

    if dsshow is None:
        from datashader.mpl_ext import dsshow

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

def simplex_plot(simplex_df,
                 num_classes,
                 colors,
                 s=30,
                 alpha=0.5,
                 linewidth=2):
    # simplex_df is a pandas dataframe with 6 columns:
    #   - population1_class
    #   - population2_class
    #   - population1_simplex_x
    #   - population1_simplex_y
    #   - population2_simplex_x
    #   - population2_simplex_y
    # population 1 is the reference population
    # population 2 spots (genes) will be colored as they are classified in population 1
    # simplex coordinates are pre-computed to be a sum of unit vectors determined by the number of classes
    unit_vecs = _unit_vectors(num_classes)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,15))
    # create frame of simplex plot
    for n in range(len(unit_vecs)-1):
        ax1.plot([unit_vecs[n][0],unit_vecs[n+1][0]],[unit_vecs[n][1],unit_vecs[n+1][1]],color='k',linestyle='--',linewidth=linewidth)
        ax2.plot([unit_vecs[n][0],unit_vecs[n+1][0]],[unit_vecs[n][1],unit_vecs[n+1][1]],color='k',linestyle='--',linewidth=linewidth)
    ax1.plot([unit_vecs[0][0],unit_vecs[-1][0]],[unit_vecs[0][1],unit_vecs[-1][1]],color='k',linestyle='--',linewidth=linewidth)
    ax2.plot([unit_vecs[0][0],unit_vecs[-1][0]],[unit_vecs[0][1],unit_vecs[-1][1]],color='k',linestyle='--',linewidth=linewidth)
    # scatter plot simplex plot data
    for cls in list(np.unique(simplex_df['population1_class'])):
        df_cls = simplex_df[simplex_df['population1_class'] == cls]
        ax1.scatter(x=df_cls['population1_simplex_x'],y=df_cls['population1_simplex_y'],s=s,alpha=alpha,c=colors[cls])
        ax2.scatter(x=df_cls['population2_simplex_x'],y=df_cls['population2_simplex_y'],s=s,alpha=alpha,c=colors[cls])
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    handles, labels = ax1.get_legend_handles_labels()
    plt.tight_layout
    
def _unit_vectors(num_zones):
    vectors = []
    for i in range(num_zones):
        rad = (math.pi*(4*i + num_zones))/(2*num_zones)
        x = round(math.cos(rad),2)
        y = round(math.sin(rad),2)
        vectors.append([x,y])
    return vectors