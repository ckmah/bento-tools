import warnings

from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

from functools import partial

ds = None
tf = None
geopandas = None
plot = None
dsshow = None

import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from upsetplot import UpSet, from_indicators
from matplotlib.colors import ListedColormap
import scanpy as sc

from ._utils import savefig
from .._utils import PATTERN_NAMES, PATTERN_COLORS
from ..preprocessing import get_points


@savefig
def qc_metrics(adata, fname=None):
    """
    Plot quality control metric distributions.
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
def pattern_plot(data, percentage=False, scale=1, fname=None):
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
        p_df = p_df[p_df["value"] == 1]
        p_df = p_df.set_index(["cell", "gene"])
        sample_labels.append(p_df)

    sample_labels = pd.concat(sample_labels, axis=1) == 1
    sample_labels = sample_labels == 1
    sample_labels.columns = PATTERN_NAMES

    # Sort by degree, then pattern name 
    sample_labels['degree'] = -sample_labels[PATTERN_NAMES].sum(axis=1)
    sample_labels = sample_labels.reset_index().sort_values(['degree'] + PATTERN_NAMES, ascending=False).drop('degree', axis=1)
    
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
def pattern_diff(data, phenotype, fname=None):
    """Visualize gene pattern frequencies between groups of cells by plotting log2 fold change and -log10p."""
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
        #         alpha=0.4,
        linewidth=0,
    )

    for ax in g.axes:
        ax.axvline(0, lw=0.5, c="grey")
        ax.axvline(-2, lw=1, c="pink", ls="dotted")
        ax.axvline(2, lw=1, c="pink", ls="dotted")
        ax.axhline(-np.log10(0.05), c="pink", ls="dotted", zorder=0)
        sns.despine()

    return g


def umap(data, **kwargs):
    f"""{sc.pl.umap.__doc__}"""
    adata = data.copy()

    adata.obs = adata.obs.loc[:, adata.obs.dtypes != "geometry"]
    sc.pl.umap(adata, **kwargs)


@savefig
def plot_cells(
    data,
    kind="scatter",
    hue=None,
    palette=None,
    cmap='Blues_r',
    tile=False,
    lw=1,
    col_wrap=4,
    binwidth=20,
    style=None,
    shape_names=['cell_shape', 'nucleus_shape'],
    legend=True,
    frameon=True,
    axsize=4,
    ax=None,
    fname=None,
    **kwargs
):
    """
    Visualize distribution of variable in spatial coordinates.
    Parameters
    ----------
    data : dict
    kind : {'scatter', 'hist', 'kde'}
        Selects the underlying seaborn plot function.
    """

    # Add all shape_names if 'all'
    if shape_names == "all":
        shape_names = data.obs.columns[data.obs.columns.str.endswith("shape")]

    # Convert draw_shape_names to list
    shape_names = [shape_names] if type(shape_names) is str else shape_names

    # Subset adata info by genes and cells
    points = get_points(data, cells=data.obs_names, genes=data.var_names)

    points = geopandas.GeoDataFrame(
        points, geometry=geopandas.points_from_xy(points["x"], points["y"])
    )

    # Get shape_names and points
    shapes_gdf = geopandas.GeoDataFrame(data.obs[shape_names], geometry="cell_shape")
    shape_names = shapes_gdf.columns.tolist()

    # Plot in single figure
    if not tile:
        if ax is None:
            ax = plt.gca()
        # fig, ax = plt.subplots(1, 1, figsize=(axsize, axsize))
        ax.set(xticks=[], yticks=[], facecolor='black', adjustable="datalim")
        ax.axis(frameon)
        
        # Set frame to white
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

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
            subplot_kw=dict(facecolor='black'),
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
                    spine.set_edgecolor('white')

                # Only make legend for last plot
                n_genes = len(np.unique(p["gene"]))
                if legend and i == len(shapes_gdf)-1:
                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
                else:
                    ax.legend().remove()

            except (IndexError, ValueError):
                ax.remove()


def _spatial_line(geo_df, shape_names, lw, ax):
    for sname in shape_names:
        geo_df.set_geometry(sname).plot(
            color=(0, 0, 0, 0), edgecolor=(1,1,1, 0.8), lw=lw, ax=ax
        )


def _spatial_scatter(points_gdf, hue, palette, style, ax, **kwargs):
    # Override scatterplot parameter defaults
    scatter_kws = dict(linewidth=0, s=10)
    scatter_kws.update(**kwargs)

    # Remove categories with no data; otherwise legend is very long
    for cat in [hue, style]:
        if cat in points_gdf.columns and points_gdf[cat].dtype == 'category':
            points_gdf[cat].cat.remove_unused_categories(inplace=True)

    sns.scatterplot(
        data=points_gdf, x="x", y="y", hue=hue, palette=palette, style=style, ax=ax, **scatter_kws
    )


def _spatial_hist(points_gdf, hue, cmap, binwidth, ax, **kwargs):
    # Override scatterplot parameter defaults
    hist_kws = dict()
    hist_kws.update(**kwargs)

    # Remove categories with no data; otherwise legend is very long

    if hue in points_gdf.columns and points_gdf[hue].dtype == 'category':
        points_gdf[hue].cat.remove_unused_categories(inplace=True)

    sns.histplot(
        data=points_gdf, x="x", y="y", hue=hue, cmap=cmap, binwidth=binwidth, ax=ax, **hist_kws
    )


def _spatial_kde(points_gdf, hue, cmap, ax, **kwargs): 
    kde_kws = dict()
    kde_kws.update(**kwargs)
    
    sampled_df=points_gdf.sample(frac=0.2)
    
    # Remove categories with no data; otherwise legend is very long
    if hue in sampled_df.columns and sampled_df[hue].dtype == 'category':
        sampled_df[hue].cat.remove_unused_categories(inplace=True)
        
    sns.kdeplot(data=sampled_df, x="x", y="y", hue=hue, cmap=cmap, ax=ax, **kde_kws)