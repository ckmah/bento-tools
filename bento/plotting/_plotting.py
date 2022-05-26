import warnings

warnings.filterwarnings("ignore")

import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import radviz
import seaborn as sns
from upsetplot import UpSet, from_indicators

from ._utils import savefig
from .._utils import PATTERN_NAMES, PATTERN_COLORS
from ..preprocessing import get_points
from ..tools._lp import lp_stats


@savefig
def qc_metrics(adata, fname=None):
    """
    Plot quality control metric distributions.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    color = "lightseagreen"

    fig, axs = plt.subplots(2, 3, figsize=(8, 5))

    kde_params = dict(color=color, shade=True, legend=False)

    with sns.axes_style("ticks"):
        sns.kdeplot(adata.obs["total_counts"], ax=axs[0][0], **kde_params)

        sns.distplot(
            adata.X.flatten() + 1,
            color=color,
            kde=False,
            hist_kws=dict(log=True),
            ax=axs[0][1],
        )

        sns.kdeplot(adata.obs["n_genes_by_counts"], ax=axs[0][2], **kde_params)

        sns.kdeplot(adata.obs["cell_area"], ax=axs[1][0], **kde_params)

        sns.kdeplot(adata.obs["cell_perimeter"], ax=axs[1][1], **kde_params)

        dual_colors = sns.light_palette(color, n_colors=2, reverse=True)
        no_nucleus_count = (adata.obs["nucleus_shape"] == None).sum()
        pie_values = [adata.n_obs - no_nucleus_count, no_nucleus_count]
        pie_percents = np.array(pie_values) / adata.n_obs * 100
        pie_labels = [
            f"Yes\n{pie_values[0]} ({pie_percents[0]:.1f}%)",
            f"No\n{pie_values[1]} ({pie_percents[1]:.1f}%)",
        ]
        axs[1][2].pie(pie_values, labels=pie_labels, colors=dual_colors)
        pie_inner = plt.Circle((0, 0), 0.6, color="white")
        axs[1][2].add_artist(pie_inner)

        sns.despine()

    titles = [
        "Transcripts per Cell",
        "Transcripts per Gene",
        "Genes per Cell",
        "Cell Area",
        "Cell Perimeter",
        "Cell has Nucleus",
    ]
    xlabels = ["mRNA count", "Gene count", "Gene count", "Pixels", "Pixels", ""]

    for i, ax in enumerate(np.array(axs).reshape(-1)):
        #         ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%1.e"))
        ax.set_xlabel(xlabels[i], fontsize=12)
        ax.set_title(titles[i], fontsize=14)
        ax.grid(False)

    plt.tight_layout()


@savefig
def lp_dist(data, percentage=False, scale=1, fname=None):
    """Plot pattern combination frequencies as an UpSet plot.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    percentage : bool, optional
        If True, label each bar as a percentage else label as a count, by default False
    scale : int, optional
        scale > 1 scales the plot larger, scale < 1 scales. the plot smaller, by default 1
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    sample_labels = []
    for p in PATTERN_NAMES:
        p_df = data.to_df(p).reset_index().melt(id_vars="cell")
        p_df = p_df[~p_df["value"].isna()]
        p_df = p_df.set_index(["cell", "gene"])
        sample_labels.append(p_df)

    sample_labels = pd.concat(sample_labels, axis=1) == 1
    sample_labels = sample_labels == 1
    sample_labels.columns = PATTERN_NAMES

    # Drop unlabeled samples
    # sample_labels = sample_labels[sample_labels.sum(axis=1) > 0]

    # Sort by degree, then pattern name
    sample_labels["degree"] = -sample_labels[PATTERN_NAMES].sum(axis=1)
    sample_labels = (
        sample_labels.reset_index()
        .sort_values(["degree"] + PATTERN_NAMES, ascending=False)
        .drop("degree", axis=1)
    )

    upset = UpSet(
        from_indicators(PATTERN_NAMES, data=sample_labels),
        element_size=scale * 40,
        min_subset_size=sample_labels.shape[0] * 0.001,
        facecolor="lightgray",
        sort_by=None,
        show_counts=(not percentage),
        show_percentages=percentage,
    )

    for p, color in zip(PATTERN_NAMES, PATTERN_COLORS):
        if sample_labels[p].sum() > 0:
            upset.style_subsets(present=p, max_degree=1, facecolor=color)

    upset.plot()
    plt.suptitle(f"Localization Patterns\n{data.n_obs} cells, {data.n_vars} genes")


@savefig
def lp_gene_dist(data, fname=None):
    """Plot the cell fraction distribution of each pattern as a density plot.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    lp_stats(data)

    col_names = [f"{p}_fraction" for p in PATTERN_NAMES]
    gene_frac = data.var[col_names]
    gene_frac.columns = PATTERN_NAMES
    # Plot frequency distributions
    sns.displot(
        data=gene_frac,
        kind="kde",
        multiple="layer",
        height=3,
        palette=PATTERN_COLORS,
    )
    plt.xlim(0, 1)
    sns.despine()


@savefig
def lp_genes(
    data,
    kind="scatter",
    hue="Pattern",
    sizes=(2, 100),
    gridsize=20,
    random_state=4,
    fname=None,
    **kwargs,
):
    """
    Plot the pattern distribution of each gene in a RadViz plot. RadViz projects
    an N-dimensional data set into a 2D space where the influence of each dimension
    can be interpreted as a balance between the influence of all dimensions.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    kind : str
        'Scatter' for scatter plot, 'hex' for hex plot, default "scatter"
    hue : str
        Name of columns in data.obs to color points, default "Pattern"
    sizes : tuple
        Minimum and maximum point size to scale points, default (2, 100)
    gridsize : int
        Number of hex bins along each axis, default 20
    fname : str, optional
        Save the figure to specified filename, by default None
    **kwargs
        Options to pass to matplotlib plotting method.
    """
    lp_stats(data)

    palette = dict(zip(PATTERN_NAMES, PATTERN_COLORS))

    # RADVIZ plot
    if kind == "hex":
        figsize = (6, 6)
    else:
        figsize = (6, 6)
    fig = plt.figure(figsize=figsize)

    # Use Plot the "circular" axis and labels, hide points
    # TODO move "pattern" computation to lp_stats
    col_names = [f"{p}_fraction" for p in PATTERN_NAMES]
    gene_frac = data.var[col_names]
    gene_frac.columns = PATTERN_NAMES
    gene_frac["Pattern"] = gene_frac.idxmax(axis=1)
    gene_frac_copy = gene_frac.copy()
    gene_frac_copy["Pattern"] = ""
    ax = radviz(gene_frac_copy, "Pattern", s=0)
    del gene_frac_copy
    ax.get_legend().remove()
    circle = plt.Circle((0, 0), radius=1, color="black", fill=False)
    ax.add_patch(circle)

    # Hide 2D axes
    ax.axis(False)

    # Get points
    pts = []
    for c in ax.collections:
        pts.extend(c.get_offsets().data)

    pts = np.array(pts).reshape(-1, 2)
    xy = pd.DataFrame(pts, index=gene_frac.index)
    xy["Pattern"] = gene_frac["Pattern"]

    # Plot points as scatter or hex
    if kind == "scatter":

        del ax.collections[0]

        # Scale point size by max
        xy["Fraction of cells"] = gene_frac.iloc[:, :5].max(axis=1)

        # Plot points
        sns.scatterplot(
            data=xy.sample(frac=1, random_state=random_state),
            x=0,
            y=1,
            size="Fraction of cells",
            hue=hue,
            sizes=sizes,
            linewidth=0,
            palette=palette,
            ax=ax,
            **kwargs,
        )
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=False)

    elif kind == "hex":
        # Hexbin
        xy.plot.hexbin(
            x=0,
            y=1,
            gridsize=gridsize,
            extent=(-1, 1, -1, 1),
            cmap=sns.light_palette("lightseagreen", as_cmap=True),
            mincnt=1,
            colorbar=False,
            ax=ax,
            **kwargs,
        )
        # [left, bottom, width, height]
        plt.colorbar(
            ax.collections[-1], cax=fig.add_axes([1, 0.4, 0.05, 0.3]), label="genes"
        )


@savefig
def lp_diff(data, phenotype, fname=None):
    """Visualize gene pattern frequencies between groups of cells by plotting
    log2 fold change and -log10p, similar to volcano plot. Run after `bento.tl.lp_diff()`

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    phenotype : str
        Variable used to group cells when calling `bento.tl.lp_diff()`.
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    diff_stats = data.uns[f"diff_{phenotype}"]

    g = sns.relplot(
        data=diff_stats,
        x=f"log2fc",
        y="-log10padj",
        size=4,
        hue="pattern",
        col="phenotype",
        col_wrap=3,
        height=2.5,
        palette="tab10",
        s=20,
        linewidth=0,
    )

    for ax in g.axes:
        ax.axvline(0, lw=0.5, c="grey")  # -log2fc = 0
        ax.axvline(-2, lw=1, c="pink", ls="dotted")  # log2fc = -2
        ax.axvline(2, lw=1, c="pink", ls="dotted")  # log2fc = 2
        ax.axhline(
            -np.log10(0.05), c="pink", ls="dotted", zorder=0
        )  # line where FDR = 0.05
        sns.despine()

    return g


@savefig
def plot_cells(
    data,
    kind="scatter",
    fov=None,
    fov_key="batch",
    hue=None,
    palette=None,
    cmap="Blues_r",
    tile=False,
    lw=1,
    col_wrap=4,
    binwidth=20,
    style=None,
    shape_names=["cell_shape", "nucleus_shape"],
    legend=True,
    frameon=True,
    axsize=4,
    ax=None,
    fname=None,
    **kwargs,
):
    """Visualize 2-dimensional distribution of points in spatial coordinates.

    Parameters
    ----------
    data : AnnData
        Spatial AnnData format
    kind : 'scatter', 'hist', or 'kde', optional
        Kind of plot to draw, wrapper for Seaborn `sns.scatterplot()`, `sns.histplot()`,
        and `sns.kdeplot()` respectively. By default "scatter".
    fov : str, 'all', list of str, optional
        If 'all' or list, plots all or specified fields of view (`batch` in `data.obs.keys()`) in
        separate subplots  and `tile` is ignored.
    hue : str, optional
        Grouping variable present in `data.uns['points'].columns` that will produce different colors.
        Can be either categorical or numeric, although color mapping will behave differently in latter case.
        By default None.
    palette : string, list, dict, or `matplotlib.colors.Colormap`, optional
        Method for choosing the colors to use when mapping the hue semantic. String values are
        passed to `sns.color_palette()`. List or dict values imply categorical mapping, while a colormap
        object implies numeric mapping. By default None.
    cmap : str, optional
        The mapping from data values to color space, only applies to `kind=hist` and `kind=kde` options.
        By default "Blues_r".
    tile : bool, optional
        If True, plots each cell in separate subplots, else plots all cells in a single subplot assuming
        a common coordinate space. By default False.
    lw : int, optional
        Linewidth of drawn `shape_names`, by default 1
    col_wrap : int, optional
        If `tile=True`, or `fov` is specified, wraps tiling at this width, by default 4
    binwidth : int, optional
        For `kind=hist`, sets width of each bin, by default 20
    style : str, optional
        If `kind=scatter`, grouping variable present in `data.uns['points'].columns` that will produce
        elements with different styles.  Can have a numeric dtype but will always be treated as categorical.
        By default None.
    shape_names : list, optional
        List of shapes to plot, by default ["cell_shape", "nucleus_shape"]
    legend : bool, optional
        Whether to plot legend, by default True
    frameon : bool, optional
        Whether to draw figure frame, by default True
    axsize : int, optional
        Size of subplot in inches, by default 4
    ax : _type_, optional
        Axes in which to draw the plot. If None, use the currently-active Axes.
    fname : str, optional
        Save the figure to specified filename, by default None
    """
    # Add all shape_names if 'all'
    if shape_names == "all":
        shape_names = data.obs.columns[data.obs.columns.str.endswith("shape")]

    # Convert draw_shape_names to list
    shape_names = [shape_names] if isinstance(shape_names) is str else shape_names

    # Subset adata info by genes and cells
    points = get_points(data, cells=data.obs_names, genes=data.var_names)
    points[fov_key] = points[fov_key].astype(str)

    points = geopandas.GeoDataFrame(
        points, geometry=geopandas.points_from_xy(points["x"], points["y"])
    )

    # Get shape_names and points
    shapes_gdf = data.obs[shape_names]
    shapes_gdf[fov_key] = shapes_gdf[fov_key].astype(str)
    shapes_gdf = geopandas.GeoDataFrame(data.obs[shape_names], geometry="cell_shape")
    shape_names = shapes_gdf.columns.tolist()

    # Plot in single figure if not tiling
    if fov:
        # Assume fov is a single field of view
        if fov == "all":
            fov = list(set(data.obs[fov_key].sort_values()))

        if isinstance(fov) is list:
            nfov = len(fov)
            ncols = min(col_wrap, nfov)
            nrows = max(1, int(np.ceil(nfov / ncols)))
            fig, axs = plt.subplots(
                nrows,
                ncols,
                sharex=False,
                sharey=False,
                subplot_kw=dict(facecolor="black"),
                figsize=(axsize * ncols, axsize * nrows),
            )
            plt.subplots_adjust(wspace=0, hspace=0)

            # Plot list of fovs in separate subplots
            for i, ax in enumerate(np.array(axs).flatten()):
                cur_fov = fov[i]
                # Subset to single fov
                fov_points = points[points[fov_key] == cur_fov]
                fov_shapes_gdf = shapes_gdf[shapes_gdf[fov_key] == cur_fov]
                _plot_cells_subplot(
                    fov_points,
                    fov_shapes_gdf,
                    kind,
                    hue,
                    palette,
                    cmap,
                    style,
                    lw,
                    binwidth,
                    shape_names,
                    legend,
                    frameon,
                    ax,
                    **kwargs,
                )

        # Plot single fov
        elif isinstance(fov) is str:
            points = points[points[fov_key] == fov]
            shapes_gdf = shapes_gdf[shapes_gdf[fov_key] == fov]
            _plot_cells_subplot(
                points,
                shapes_gdf,
                kind,
                hue,
                palette,
                cmap,
                style,
                lw,
                binwidth,
                shape_names,
                legend,
                frameon,
                ax,
                **kwargs,
            )
        else:
            pass
    else:
        if not tile:
            _plot_cells_subplot(
                points,
                shapes_gdf,
                kind,
                hue,
                palette,
                cmap,
                style,
                lw,
                binwidth,
                shape_names,
                legend,
                frameon,
                ax,
                **kwargs,
            )

        # Plot each cell in separate subplots
        else:
            # Determine fixed radius of each subplot
            cell_bounds = shapes_gdf.bounds
            cell_maxw = (cell_bounds["maxx"] - cell_bounds["minx"]).max()
            cell_maxh = (cell_bounds["maxy"] - cell_bounds["miny"]).max()
            ax_radius = 1.1 * (max(cell_maxw, cell_maxh) / 2)

            # Initialize subplots
            ncols = min(col_wrap, len(shapes_gdf))
            nrows = max(1, int(np.ceil(len(shapes_gdf) / ncols)))
            fig, axs = plt.subplots(
                nrows,
                ncols,
                sharex=False,
                sharey=False,
                subplot_kw=dict(facecolor="black"),
                figsize=(axsize * ncols, axsize * nrows),
            )
            plt.subplots_adjust(wspace=0, hspace=0)

            # Plot cells separately
            for i, ax in enumerate(np.array(axs).flatten()):
                try:
                    # Select subset data
                    s = shapes_gdf.iloc[[i]]
                    p = points.loc[points["cell"] == s.index[0]]

                    # Plot points
                    if kind == "scatter":
                        _spatial_scatter(p, hue, palette, style, ax, **kwargs)
                    elif kind == "hist":
                        _spatial_hist(p, hue, cmap, binwidth, ax, **kwargs)
                    elif kind == "kde":
                        _spatial_kde(p, hue, cmap, ax, **kwargs)

                    # Plot shapes
                    _spatial_line(s, shape_names, lw, ax)

                    # Set axes boundaries to be square; make sure size of cells are relative to one another
                    s_bound = s.bounds
                    centerx = np.mean([s_bound["minx"], s_bound["maxx"]])
                    centery = np.mean([s_bound["miny"], s_bound["maxy"]])
                    ax.set_xlim(centerx - ax_radius, centerx + ax_radius)
                    ax.set_ylim(centery - ax_radius, centery + ax_radius)

                    ax.set(xticks=[], yticks=[], xlabel=None, ylabel=None)
                    ax.axis(frameon)

                    # Set frame to white
                    for spine in ax.spines.values():
                        spine.set_edgecolor("white")

                    # Only make legend for last plot
                    if legend and i == len(shapes_gdf) - 1:
                        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
                    else:
                        ax.legend().remove()

                except (IndexError, ValueError):
                    ax.remove()


def _plot_cells_subplot(
    points,
    shapes_gdf,
    kind,
    hue,
    palette,
    cmap,
    style,
    lw,
    binwidth,
    shape_names,
    legend,
    frameon,
    ax,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    # fig, ax = plt.subplots(1, 1, figsize=(axsize, axsize))
    ax.set(xticks=[], yticks=[], facecolor="black", adjustable="datalim")
    ax.axis(frameon)

    # Set frame to white
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Plot points
    if kind == "scatter":
        _spatial_scatter(points, hue, palette, style, ax, **kwargs)
    elif kind == "hist":
        _spatial_hist(points, hue, cmap, binwidth, ax, **kwargs)
    elif kind == "kde":
        _spatial_kde(points, hue, cmap, ax, **kwargs)

    # Plot shapes
    _spatial_line(shapes_gdf, shape_names, lw, ax)

    if not legend:
        ax.legend().remove()


def _spatial_line(geo_df, shape_names, lw, ax):
    for sname in shape_names:
        geo_df.set_geometry(sname).plot(
            color=(0, 0, 0, 0), edgecolor=(1, 1, 1, 0.8), lw=lw, ax=ax
        )


def _spatial_scatter(points_gdf, hue, palette, style, ax, **kwargs):
    # Override scatterplot parameter defaults
    scatter_kws = dict(linewidth=0, s=10)
    scatter_kws.update(**kwargs)

    # Remove categories with no data; otherwise legend is very long
    for cat in [hue, style]:
        if cat in points_gdf.columns and points_gdf[cat].dtype == "category":
            points_gdf[cat].cat.remove_unused_categories(inplace=True)

    sns.scatterplot(
        data=points_gdf,
        x="x",
        y="y",
        hue=hue,
        palette=palette,
        style=style,
        ax=ax,
        **scatter_kws,
    )


def _spatial_hist(points_gdf, hue, cmap, binwidth, ax, **kwargs):
    # Override scatterplot parameter defaults
    hist_kws = dict()
    hist_kws.update(**kwargs)

    # Remove categories with no data; otherwise legend is very long

    if hue in points_gdf.columns and points_gdf[hue].dtype == "category":
        points_gdf[hue].cat.remove_unused_categories(inplace=True)

    sns.histplot(
        data=points_gdf,
        x="x",
        y="y",
        hue=hue,
        cmap=cmap,
        binwidth=binwidth,
        ax=ax,
        **hist_kws,
    )


def _spatial_kde(points_gdf, hue, cmap, ax, **kwargs):
    kde_kws = dict()
    kde_kws.update(**kwargs)

    sampled_df = points_gdf.sample(frac=0.2)

    # Remove categories with no data; otherwise legend is very long
    if hue in sampled_df.columns and sampled_df[hue].dtype == "category":
        sampled_df[hue].cat.remove_unused_categories(inplace=True)

    sns.kdeplot(data=sampled_df, x="x", y="y", hue=hue, cmap=cmap, ax=ax, **kde_kws)
