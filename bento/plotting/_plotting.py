import warnings
from functools import partial

import datashader as ds
import datashader.transfer_functions as tf
import geopandas
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datashader.mpl_ext import dsshow

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
        facet_kws=dict(xlim=(-1, 1))
    )

    g.map(plt.axhline, y=-np.log10(0.05),
          color="gray", lw=1, linestyle="--", zorder=0)
    (g.map(plt.axvline, x=0, color="black", lw=1, linestyle="-", zorder=0)
      .set_axis_labels("Marginal Effect (dy/dx)", f"Significance ({rank})")
      .set_titles("{col_name}"))
    # g.map(sns.despine, top=False, right=False, bottom=False, left=False)


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


def spots_distr(data, direction='cells', sharey=True, binwidth=10, layer="pattern"):
    """Plot localization pattern distributions across all cells (default) or genes.

    Parameters
    ----------
    data : [type]
        [description]
    """
    if direction == 'cells':
        axis = 1
    elif direction == 'genes':
        axis = 0

    # Count pattern occurences within axis
    pattern_distr = data.to_df('pattern').apply(
        lambda fiber: fiber.value_counts(), axis=axis)

    if axis == 0:
        pattern_distr = pattern_distr.T

    pattern_distr = ((pattern_distr.T * 100) / pattern_distr.sum(axis=1)).T

    # Pattern fraction at dataset level
    pattern_total_distr = pattern_distr.sum(axis=0)
    pattern_total_distr = ((pattern_total_distr*100) /
                           pattern_total_distr.sum()).to_frame().T.drop('none', axis=1).melt()
    pattern_total_distr.columns = ['pattern', 'value']

    distr_long = pattern_distr.drop('none', axis=1)
    distr_long['group'] = None
    distr_long = distr_long.melt(id_vars='group')
    distr_long.columns = ['group', 'pattern', 'value']

    palette = sns.color_palette('muted6')

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        sns.barplot(data=pattern_total_distr, x='value', y='pattern',
                    hue='pattern', palette=palette, dodge=False, ax=ax)
        ax.set_title('Percent of total samples', fontsize=15)
        ax.set_xlabel('Percent', fontsize=15)
        ax.set_ylabel('')
        ax.legend().remove()
        sns.despine()

        pattern_names = distr_long['pattern'].unique()

        g = sns.FacetGrid(
            distr_long,
            col='pattern',
            hue='pattern',
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
            stat='count',
            binwidth=binwidth,
            clip_on=False,
            alpha=1,
        )

        # here we add a horizontal line for each plot
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        if direction == 'cells':
            x_unit = 'genes'
        else:
            x_unit = 'cells'

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
            f"Localization pattern distribution across {direction}",
            ha="center",
            fontsize=18,
            fontweight=20,
        )
    return g


def plot_cells(
    data,
    kind="scatter",
    tile=False,
    cells=None,
    genes=None,
    markersize=3,
    alpha=0.2,
    col=None,
    col_order=None,
    cmap="tab:blue",
    ncols=4,
    cell_ncols=4,
    masks="all",
    binwidth=3,
    spread=False,
    width=8
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
    points = geopandas.GeoDataFrame(
        points, geometry=geopandas.points_from_xy(points["x"], points["y"])
    )

    # Get masks and points
    shapes = geopandas.GeoDataFrame(data.obs[masks], geometry="cell_shape")
    masks = shapes.columns.tolist()

    # Set number of columns
    if col:
        ncols = min(ncols, points[col].nunique())
        col_names = points[col].unique()
    else:
        ncols = 1
        col_names = [""]

    if col_order:
        col_names = col_order

    nrows = max(1, int(np.ceil(len(col_names) / ncols)))

    # Create subplots for each "col" value
    fig = plt.figure(figsize=(width, width))
    col_grid = fig.add_gridspec(nrows, ncols, wspace=0, hspace=0)

    nd_col_names = np.array(col_names)
    nd_col_names.resize(nrows*ncols)
    nd_col_names = nd_col_names.reshape(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            # Set subplot title
            c_name = nd_col_names[i, j]

            if col:
                points_c = points.loc[points[col] == c_name]
            else:
                points_c = points

            if tile is False:
                col_ax = fig.add_subplot(col_grid[i, j])
                col_ax.set_title(c_name)

                _plot_cells(masks, shapes, col_ax, kind, points_c,
                            markersize, binwidth, spread, cmap)
            else:
                cell_ncols = min(len(shapes), cell_ncols)
                cell_nrows = max(1, int(np.ceil(len(shapes) / cell_ncols)))
                cell_grid = col_grid[i, j].subgridspec(
                    cell_nrows, cell_ncols, wspace=0, hspace=0)
                cell_axes = cell_grid.subplots()

                # Determine unit width/height of each subplot
                cell_bounds = shapes.bounds
                cell_maxw = (cell_bounds['maxx'] - cell_bounds['minx']).max()
                cell_maxh = (cell_bounds['maxy'] - cell_bounds['miny']).max()
                ax_radius = max(cell_maxw, cell_maxh) / 2

                for celli, cell_ax in enumerate(cell_axes.flat):
                    try:
                        cell_shape = shapes.iloc[[celli], :]
                        centroidx, centroidy = cell_shape.centroid[0].xy
                        centroidx = centroidx[0]
                        centroidy = centroidy[0]

                        cell_ax.set_xlim(centroidx-ax_radius,
                                         centroidx + ax_radius)
                        cell_ax.set_ylim(centroidy-ax_radius,
                                         centroidy + ax_radius)

                        _plot_cells(masks,
                                    cell_shape,
                                    cell_ax,
                                    kind,
                                    points_c.loc[points_c['cell']
                                                 == shapes.index[celli]],
                                    markersize,
                                    binwidth,
                                    spread,
                                    cmap
                                    )

                        cell_ax.set_xlim(centroidx-ax_radius,
                                         centroidx + ax_radius)
                        cell_ax.set_ylim(centroidy-ax_radius,
                                         centroidy + ax_radius)

                    except (IndexError, ValueError):
                        cell_ax.remove()

    plt.close()
    return fig


def _plot_cells(masks, cell_shape, ax, kind, points_c, markersize, binwidth, spread, cmap):
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

    if not isinstance(type(cmap), type(sns.color_palette(as_cmap=True))):
        cmap = sns.light_palette(cmap, as_cmap=True)

    # Plot mask outlines
    for mask in masks:
        cell_shape.set_geometry(mask).plot(
            color=(0, 0, 0, 0.05), edgecolor=(0, 0, 0, 0.3), ax=ax
        )

    # Plot points
    if kind == "scatter":
        col_value = None

        points_c.plot(
            column=col_value,
            markersize=markersize,
            alpha=0.5,
            legend=True,
            ax=ax,
        )

    # Plot density
    elif kind == "hist":

        if spread:
            spread = partial(tf.dynspread, threshold=0.5)
        else:
            spread = None

        # Percent of window size
        scaled_binwidth = max(ax.get_window_extent().width * binwidth * .01,
                              ax.get_window_extent().height * binwidth * .01)

        dsshow(
            points_c,
            ds.Point("x", "y"),
            norm="linear",
            cmap=cmap,
            width_scale=1 / scaled_binwidth,
            height_scale=1 / scaled_binwidth,
            shade_hook=spread,
            ax=ax,
        )

    return ax


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
