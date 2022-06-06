import warnings

warnings.filterwarnings("ignore")

import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import pandas as pd
from pandas.plotting import radviz
import seaborn as sns
from upsetplot import UpSet, from_indicators

from tqdm.auto import tqdm

from ._utils import savefig
from .._utils import PATTERN_NAMES, PATTERN_COLORS, TENSOR_DIM_NAMES
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

        sns.kdeplot(adata.obs["cell_density"], ax=axs[1][1], **kde_params)

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
        "Transcript Density",
        "Cell has Nucleus",
    ]
    xlabels = [
        "mRNA count",
        "Gene count",
        "Gene count",
        "Pixels",
        "Transcripts/pixel",
        "",
    ]

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
    ax=None,
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
    if not ax:
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

    if hue and hue != "Pattern":
        gene_frac = gene_frac.join(data.var[hue])

    if not ax:
        ax = radviz(gene_frac_copy, "Pattern", s=0)
    else:
        radviz(gene_frac_copy, "Pattern", s=0, ax=ax)
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
def cellplot(
    adata,
    fovs="0",
    fov_key="batch",
    hue=None,
    col=None,
    kind="hist",
    legend=True,
    palette=None,
    hue_order=None,
    hue_norm=None,
    col_wrap=None,
    col_order=None,
    shape_names=["cell_shape", "nucleus_shape"],
    height=8,
    facet_kws=None,
    fname=None,
    **kwargs,
):
    # Get points
    points = get_points(adata, asgeo=False)

    # Process fovs
    if fovs == "all":
        fovs = adata.obs[fov_key].unique().tolist()

    fovs = [fovs] if isinstance(fovs, str) else fovs

    # Add all shape_names if 'all'
    if shape_names == "all":
        shape_names = adata.obs.columns[adata.obs.columns.str.endswith("shape")]

    # Convert draw_shape_names to list
    shape_names = [shape_names] if isinstance(shape_names, str) else shape_names

    # Get shapes
    obs_columns = list(shape_names)
    if col:
        # Likely mistake
        if col == 'fov' or col == 'fovs':
            col = fov_key
            
        obs_columns.append(col)
    if fov_key:
        obs_columns.append(fov_key)
    obs_columns = list(set(obs_columns))

    # Get shapes
    shapes = adata.obs.reset_index()[obs_columns]

    # Make sure col is same type across points and shapes
    if points[fov_key].dtype != shapes[fov_key].dtype:
        points[fov_key] = points[fov_key].astype(str)
        shapes[fov_key] = shapes[fov_key].astype(str)

    if fovs[0] != "all":
        # Subset to specified col values only; less filtering = faster plotting
        points = points[points[fov_key].isin(fovs)]
        shapes = shapes[shapes[fov_key].isin(fovs)]

    if col:
        # Make sure col is same type across points and shapes
        if points[col].dtype != shapes[col].dtype:
            points[col] = points[col].astype(str)
            shapes[col] = shapes[col].astype(str)

        # Subset to specified col values only; less filtering = faster plotting
        if col_order:
            points = points[points[col].isin(col_order)]
            shapes = shapes[shapes[col].isin(col_order)]

    # Remove unused categories in points
    for cat in points.columns:
        points[cat] = (
            points[cat].cat.remove_unused_categories()
            if points[cat].dtype == "category"
            else points[cat]
        )

    # Convert shapes to GeoDataFrames AFTER filtering
    shapes = geopandas.GeoDataFrame(shapes, geometry="cell_shape")

    # Get updated
    kws = dict(sharex=False, sharey=False)
    if isinstance(facet_kws, dict):
        kws.update(facet_kws)

    # https://stackoverflow.com/questions/32633322/changing-aspect-ratio-of-subplots-in-matplotlib
    g = sns.FacetGrid(
        points,
        col=col,
        hue=hue,
        legend_out=legend,
        palette=palette,
        hue_order=hue_order,
        col_wrap=col_wrap,
        col_order=col_order,
        height=height,
        aspect=1,
        margin_titles=False,
        **kws,
    )

    def hexbin(data, x, y, **kwargs):
        xmax = data[x].max()
        xmin = data[x].min()
        ymax = data[y].max()
        ymin = data[y].min()
        dx = xmax - xmin
        dy = ymax - ymin
        nx = kwargs['gridsize']
        ny = int((dy/dx) * nx)
        print(nx, ny)
        kwargs['gridsize'] = (nx, ny)
        plt.hexbin(data[x], data[y], extent=(xmin, xmax, ymin, ymax), **kwargs)
    
    if kind == "scatter":
        scatter_kws = dict(linewidth=0, s=5)
        scatter_kws.update(**kwargs)
        g.map_dataframe(sns.scatterplot, x="x", y="y", **scatter_kws)
    elif kind == "hist":
        hist_kws = dict(cmap="viridis", binwidth=15)
        hist_kws.update(**kwargs)
        g.map_dataframe(sns.histplot, x="x", y="y", **hist_kws)
    elif kind == "hex":
        hex_kws = dict(cmap="viridis", mincnt=1, linewidth=0, gridsize=100)
        hex_kws.update(**kwargs)
        g.map_dataframe(plt.hexbin, x="x", y="y", **hex_kws)

    if col:
        shapes = shapes.groupby(col)

        # Get max ax radius across groups
        ax_radii = []
        for k, ax in g.axes_dict.items():
            s = shapes.get_group(k)
            # Determine fixed radius of each subplot
            cell_bounds = s.bounds
            cell_maxw = cell_bounds["maxx"].max() - cell_bounds["minx"].min()
            cell_maxh = cell_bounds["maxy"].max() - cell_bounds["miny"].min()
            ax_radius = 1.1 * (max(cell_maxw, cell_maxh) / 2)
            ax_radii.append(ax_radius)

        ax_radius = max(ax_radii)

        for k, ax in tqdm(g.axes_dict.items()):
            s = shapes.get_group(k)
            shape_subplot(s, shape_names, ax_radius=ax_radius, ax=ax)

    else:
        shape_subplot(shapes, shape_names, ax=g.ax)

    if legend:
        g.add_legend()

    g.set_titles(template="")

    # box_aspect for Axes, aspect for data
    g.set(
        xticks=[],
        yticks=[],
        xlabel=None,
        ylabel=None,
        xmargin=0,
        ymargin=0,
        facecolor="black",
        box_aspect=1,
        aspect=1,
    )
    g.tight_layout()


def shape_subplot(data, shape_names, ax, ax_radius=None):
    # Gather all shapes and plot
    all_shapes = geopandas.GeoSeries(data[shape_names].values.flatten())
    all_shapes.plot(color=(0,0,0,0), edgecolor=(1, 1, 1, 0.8), lw=1, aspect=None, ax=ax)

    # Set axes boundaries to be square; make sure size of cells are relative to one another
    if ax_radius:
        s_bound = data.bounds
        centerx = np.mean([s_bound["minx"].min(), s_bound["maxx"].max()])
        centery = np.mean([s_bound["miny"].min(), s_bound["maxy"].max()])
        ax.set_xlim(centerx - ax_radius, centerx + ax_radius)
        ax.set_ylim(centery - ax_radius, centery + ax_radius)

    for spine in ax.spines.values():
        spine.set(edgecolor="white", linewidth=1)

    # Create scale bar
    scalebar = ScaleBar(
        0.1, "um", location="lower right", color="white", box_alpha=0, scale_loc="top"
    )
    ax.add_artist(scalebar)

    
def sig_samples(data, n=5):
    for f in data.uns['tensor_loadings'][TENSOR_DIM_NAMES[0]]:
        top_genes = (
            data.uns["tensor_loadings"]["genes"]
            .sort_values(f, ascending=False)
            .index.tolist()[:n]
        )

        top_cells = (
            data.uns["tensor_loadings"]["cells"]
            .sort_values(f, ascending=False)
            .index.tolist()[:n]
        )

        cellplot(
            data[top_cells, top_genes],
            fovs="all",
            kind="scatter",
            hue="gene",
            col="cell",
            height=2,
        )
        # plt.suptitle(f)