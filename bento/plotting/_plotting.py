import warnings

from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

from functools import partial

ds = None
tf = None
geopandas = None
plot = None
dsshow = None
mplcyberpunk = None

import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as plot
import seaborn as sns
from matplotlib.colors import ListedColormap

from .._utils import PATTERN_NAMES
from ..preprocessing import get_points

matplotlib.rcParams["figure.facecolor"] = (0, 0, 0, 0)
plot.rc["autoformat"] = False
plot.rc["grid"] = False


def qc_metrics(adata):
    """
    Plot quality control metric distributions.
    """
    fig, axs = plt.subplots(1, 4, figsize=(16, 3))
    plt.subplots_adjust(hspace=1)
    sns.kdeplot(adata.obs["total_counts"], fill=True, ax=axs[0])
    sns.kdeplot(adata.obs["n_genes_by_counts"], fill=True, ax=axs[1])
    sns.kdeplot(adata.obs["cell_area"], fill=True, ax=axs[2])
    sns.kdeplot(np.log2(adata.X.flatten() + 1), fill=True, ax=axs[3])
    sns.despine()

    titles = [
        "Transcripts per Cell",
        "Genes per Cell",
        "Cell Area",
        "Transcripts per Gene",
    ]
    xlabels = [
        "mRNA count",
        "Gene count",
        "Area $\mathregular{unit^2}$",
        "mRNA count",
    ]
    for i, ax in enumerate(axs):
        #     ax.set_yticks([])
        #     ax.set_ylabel("")
        ax.set_xlabel(xlabels[i], fontsize=14)
        ax.set_title(titles[i], fontsize=16)

    axs[3].set_xlabel("log2(tx count + 1)")
    plt.tight_layout()


def pattern_distribution(data, relative=False):
    """Visualize gene pattern distributions.

    Each point is a gene, denoting the number of cells the gene is detected in (x-axis) vs pattern frequency (y-axis).

    Parameters
    ----------
    data : spatial formatted AnnData
    relative : bool
        If True, plot fraction of detected cells, by default False. Otherwise plot absolute frequency.
    """
    pattern_stats = []
    for p in PATTERN_NAMES:
        p_stats = data.var[[f"{p}_fraction", f"{p}_count", "fraction_detected"]]
        p_stats = p_stats.rename(
            {
                f"{p}_fraction": "pattern_fraction",
                f"{p}_count": "pattern_count",
                "fraction_detected": "cell_fraction",
            },
            axis=1,
        )
        p_stats["pattern"] = p
        pattern_stats.append(p_stats)
    pattern_stats = pd.concat(pattern_stats).reset_index()

    g = sns.FacetGrid(
        data=pattern_stats, col="pattern", hue="pattern", col_wrap=3, size=2
    )

    if relative:
        y = "pattern_fraction"
    else:
        y = "pattern_count"

    with sns.axes_style("white"):
        g.map_dataframe(
            sns.scatterplot,
            "cell_fraction",
            y,
            s=16,
            linewidth=0,
            alpha=0.5,
        )

        for ax in g.axes:
            ax.set_xlim(0, 1)
            ax.axvline(0.5, lw=1, c="pink", ls="dotted")
            ax.set_xlabel(f"cell_fraction ({data.n_obs} cells)")
            sns.despine()

    return g


def pattern_diff(data, phenotype, group_name, relative=False):
    """Visualize gene pattern frequencies between groups of cells by plotting log2 fold change and pattern count."""
    diff_stats = data.uns[f"{phenotype}_dp"]

    if relative:
        y = "n_cells_detected"
    else:
        y = "pattern_count"

    g = sns.relplot(
        data=diff_stats,
        x=f"{group_name}_log2fc",
        y=y,
        # size="cell_fraction",
        # sizes=(2**2, 2**6),
        hue=f"{group_name}_rank",
        col="pattern",
        col_wrap=3,
        height=2.5,
        palette="crest",
        s=20,
        # alpha=0.4,
        linewidth=0,
    )

    for ax in g.axes:
        # ax.set_xlim(-4, 4)
        ax.axvline(0, lw=0.5, c="grey")
        ax.axvline(-2, lw=1, c="pink", ls="dotted")
        ax.axvline(2, lw=1, c="pink", ls="dotted")

        sns.despine()

    return g


def gene_patterns(data, gene=None, groups=None, relative=True, stacked=False):
    """Plot pattern frequency of genes.
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

    global mplcyberpunk
    if mplcyberpunk is None:
        import mplcyberpunk

    # Calculate polar angles
    angles = np.linspace(0, 2 * np.pi, len(PATTERN_NAMES), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # Groups x patterns dataframe
    all_pcounts = []
    for p in PATTERN_NAMES:
        if groups:
            pcounts = data.to_df(p)[gene].groupby(data.obs[groups]).agg("sum")
        else:
            pcounts = pd.Series([data.to_df(p)[gene].sum()], name=p).to_frame()
        all_pcounts.append(pcounts)

    all_pcounts = pd.concat(all_pcounts, axis=1)

    # Normalize by number of cells
    if relative:
        all_pcounts /= data.n_obs

    if all_pcounts.shape[0] == 1:
        all_pcounts.index = [gene]

    if stacked or all_pcounts.shape[0] == 1:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, polar=True)
        axs = [ax] * all_pcounts.shape[0]
    else:
        fig, axs = plt.subplots(
            1,
            all_pcounts.shape[0],
            sharex=True,
            subplot_kw=dict(polar=True),
            figsize=(4 * all_pcounts.shape[0], 4),
        )
        fig.subplots_adjust(wspace=0.3)

    colors = sns.color_palette("muted", n_colors=all_pcounts.shape[0])
    with sns.axes_style("whitegrid"):
        for i, rowname in enumerate(all_pcounts.index):
            pcounts = all_pcounts.iloc[i].values.tolist()
            pcounts = pcounts + [pcounts[0]]
            current_ax = axs[i]

            # Plot gene trace
            current_ax.plot(angles, pcounts, color=colors[i], linewidth=1)
            current_ax.scatter(angles, pcounts, color=colors[i], s=10)

            # Set theta axis labels
            current_ax.set_thetagrids(
                angles * 180 / np.pi, PATTERN_NAMES + [PATTERN_NAMES[0]]
            )
            current_ax.tick_params(axis="x", labelsize="medium", pad=15)

            # Format r ticks
            current_ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="lower"))
            current_ax.tick_params(axis="y", labelsize="medium", labelcolor="gray")

            # Set grid style
            current_ax.grid(
                True, axis="x", linestyle="-", linewidth=0.5, color="gray", alpha=0.5
            )
            current_ax.grid(
                True, axis="y", linestyle=":", linewidth=0.5, color="gray", alpha=0.5
            )
            current_ax.set_facecolor((0.98, 0.98, 0.98))
            current_ax.spines["polar"].set_color("gray")
            current_ax.spines["polar"].set_alpha(0.5)

        handles = [
            Line2D([], [], c=color, lw=1, marker="o", markersize=4, label=g)
            for g, color in zip(all_pcounts.index, colors[: all_pcounts.shape[0]])
        ]
        plt.legend(
            handles=handles,
            loc=(0.9, 0.95),
            labelspacing=0.1,
            fontsize="medium",
            frameon=False,
        )

        plt.tight_layout()

        for ax in axs:
            mplcyberpunk.make_lines_glow(ax, n_glow_lines=3)
            mplcyberpunk.add_underglow(ax, alpha_underglow=0.2)
            if ax == axs[-1]:
                break

    plt.close()
    return fig


def cell_patterns(data, cells=None, relative=True):
    """Plot pattern frequency across cells.

    The mean and 95% confidence interval for all cells is denoted with black points and grey boxes.
    The blue points denote the mean frequencies for the specified subset of cell(s).
    Parameters
    ----------
    data : spatial formatted AnnData
    cells : list of str, optional
        A list of cell names, by default None. The mean frequencies are plotted as blue points.
    relative : bool, optional
        Whether to normalize pattern frequencies to total cells, by default True.
    Returns
    -------
    Axes
        Returns a matplotlib Axes containing plotted figure elements.
    """
    pattern_freq = data.obs[[f"{p}_count" for p in PATTERN_NAMES]]

    if relative:
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
    base_size = 7
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

        if relative:
            stats["value"] /= data.n_vars
        plt.scatter(
            x=stats["value"], y=stats.index, c="skyblue", s=base_size ** 2, zorder=100
        )

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
    if relative:
        plt.xlim(0, 1)
        plt.xlabel(f"Fraction ({data.n_vars} genes)")
    else:
        plt.xlim(0, data.n_vars)
        plt.xlabel(f"Frequency ({data.n_vars} genes)")
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

    global ds, tf, dsshow

    if ds is None:
        import datashader as ds

    if tf is None:
        import datashader.transfer_functions as tf

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

    plt.close(fig)
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
