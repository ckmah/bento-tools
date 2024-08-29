import warnings
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

from typing import Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.axes
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import quantile_transform

from ._colors import red_light
from ._utils import savefig


def _quantiles(data: pd.DataFrame, x: str, **kwargs):
    """Plot quantiles on top of a stripplot.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with x column
    x : str
        Column to plot quantiles for
    """
    ax = plt.gca()

    ylims = ax.get_ylim()
    ymargin = 0.3 * (ylims[1] - ylims[0])
    quants = np.nanpercentile(data[x], [0, 25, 50, 75, 100])
    palette = sns.color_palette("red2blue", n_colors=len(quants) - 1)
    linecolor = sns.axes_style()["axes.edgecolor"]

    xys = [(q, ylims[0]) for q in quants[:-1]]
    widths = [quants[i + 1] - quants[i] for i in range(len(quants) - 1)]
    height = ymargin
    rects = [
        mpl.patches.Rectangle(
            xy,
            width=w,
            height=height,
            facecolor=c,
            alpha=0.8,
            linewidth=1,
            edgecolor=linecolor,
            clip_on=False,
        )
        for xy, w, c in zip(xys, widths, palette)
    ]

    for rect in rects:
        ax.add_patch(rect)


@savefig
def shape_stats(
    sdata,
    instance_key="cell_boundaries",
    nucleus_key="nucleus_boundaries",
    cols=[
        "cell_boundaries_area",
        "cell_boundaries_aspect_ratio",
        "cell_boundaries_density",
        "nucleus_boundaries_area",
        "nucleus_boundaries_aspect_ratio",
        "nucleus_boundaries_density",
    ],
    s=3,
    color="lightseagreen",
    alpha=0.3,
    rug=False,
    fname=None,
):
    """Plot shape statistic distributions for each cell.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    cols : list
        List of columns to plot
    groupby : str, optional
        Column in obs to groupby, by default None
    """
    cell_gdf = pd.DataFrame(
        sdata[instance_key].melt(
            value_vars=[c for c in cols if f"{instance_key}_" in c]
        )
    )
    nucleus_gdf = pd.DataFrame(
        sdata[nucleus_key].melt(value_vars=[c for c in cols if f"{nucleus_key}_" in c])
    )
    stats_long = pd.concat([cell_gdf, nucleus_gdf], ignore_index=True)
    # stats_long["quantile"] = stats_long.groupby("variable")["value"].transform(
    #     lambda x: quantile_transform(x.values.reshape(-1, 1), n_quantiles=100).flatten()
    # )

    stats_long["shape"] = stats_long["variable"].apply(lambda x: x.split("_")[0])
    stats_long["var"] = stats_long["variable"].apply(
        lambda x: "_".join(x.split("_")[2:])
    )
    # linecolor = sns.axes_style()["axes.edgecolor"]

    g = sns.FacetGrid(
        data=stats_long,
        row="var",
        col="shape",
        height=1.5,
        aspect=1.7,
        sharex=False,
        sharey=False,
        margin_titles=False,
    )
    g.map_dataframe(
        sns.kdeplot,
        x="value",
        color=color,
        linewidth=0,
        fill=True,
        alpha=alpha,
        rasterized=True,
    )
    if rug:
        g.map_dataframe(
            sns.rugplot,
            x="value",
            color=color,
            height=0.1,
            alpha=0.5,
            rasterized=True,
        )
    # g.map_dataframe(_quantiles, x="value")
    g.add_legend()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    for ax, var in zip(g.axes.flat, stats_long["variable"].unique()):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
        sns.despine(ax=ax, left=True)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    def plot_median(data, **kwargs):
        plt.axvline(data.median(), **kwargs)

    g.map(plot_median, "value", c=color, lw=1.5, zorder=3)


@savefig
def comp(
    sdata,
    groupby: Optional[str] = None,
    group_order: Optional[List[str]] = None,
    annotate: Optional[Union[bool, list[str]]] = None,
    min_count: int = 0,
    adjust: bool = True,
    palette: str = red_light,
    annot_color: Optional[str] = None,
    sizes: Tuple[int, int] = (5, 30),
    size_norm: Tuple[int, int] = (10, 100),
    dim_order: Optional[str] = None,
    legend: bool = True,
    height: int = 5,
    fname: Optional[str] = None,
):
    """
    Plot gene composition across set of shapes. Ideally, these shapes are non-overlapping.

    Parameters
    ----------
    sdata : SpatialData
        The spatial data to be plotted.
    groupby : str
        The column name in the data to group by.
    group_order : list
        The order of the groups for plotting.
    annotate : bool or list of str
        Whether to annotate the plot with gene names or a list of gene names to annotate.
    adjust : bool
        Whether to adjust the text positions, by default True.
    palette : str
        The color palette to use for the plot. Default is 'red_light'.
    annot_color : str
        The color to use for annotations. Default is None.
    sizes : tuple
        The minimum and maximum size of the points. Default is (5, 30).
    size_norm : tuple
        The normalization range for the point sizes. Default is (10, 100).
    dim_order : str
        The order of the dimensions for the plot. Default is None.
    legend : bool
        Whether to include a legend in the plot. Default is True.
    height : int
        The height of the plot. Default is 5.
    fname : str
        The filename to save the plot as. If None (default), the plot is not saved to file.
    """

    comp_key = f"{groupby}_comp_stats"
    if groupby and comp_key in sdata.tables["table"].uns.keys():
        comp_stats = sdata.tables["table"].uns[comp_key]
        if group_order is None:
            groups = list(comp_stats.keys())
        else:
            groups = group_order
        ngroups = len(groups)
        fig, axes = plt.subplots(1, ngroups, figsize=(ngroups * height * 1.1, height))
        if axes is not np.ndarray:
            axes = np.array([axes])

        # Plot each group separately
        for group, ax in zip(groups, axes.flat):
            group_comp = comp_stats[group]

            show_legend = False
            if legend and ax == axes.flat[-1]:
                show_legend = True

            _radviz(
                group_comp,
                annotate=annotate,
                min_count=min_count,
                adjust=adjust,
                palette=palette,
                annot_color=annot_color,
                sizes=sizes,
                size_norm=size_norm,
                dim_order=dim_order,
                legend=show_legend,
                ax=ax,
            )
            ax.set_title(group, fontsize=12)
    else:
        comp_key = "comp_stats"
        comp_stats = sdata.tables["table"].uns[comp_key]
        return _radviz(
            comp_stats,
            annotate=annotate,
            min_count=min_count,
            adjust=adjust,
            palette=palette,
            annot_color=annot_color,
            sizes=sizes,
            size_norm=size_norm,
            dim_order=dim_order,
            legend=legend,
        )


def _radviz(
    comp_stats: pd.DataFrame,
    annotate: Union[int, List[str]] = None,
    min_count: int = 0,
    adjust: bool = True,
    palette: str = red_light,
    annot_color: Optional[str] = None,
    sizes: Optional[Tuple[int, int]] = None,
    size_norm: Optional[Tuple[int, int]] = None,
    dim_order: Union[str, list, None] = "auto",
    legend: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
):
    """
    Plot a radviz plot of gene values across fields.

    Parameters
    ----------
    comp_stats : DataFrame
        Gene composition stats.
    annotate : int or list of str, optional
        Number of top genes to annotate or list of genes to annotate, by default None.
    adjust : bool, optional
        Whether to adjust the text positions, by default True.
    palette : str, optional
        Color palette, by default red_light.
    annot_color : str, optional
        The color to use for annotations, by default None.
    sizes : tuple, optional
        Size range for scatter plot, by default None.
    size_norm : tuple, optional
        Normalization range for the point sizes, by default None.
    dim_order : "auto", None, or list, optional
        Sort dimensions for more intuitive visualization, by default "auto".
        If "auto", sort dimensions by maximizing cosine similarity of adjacent
        dimensions. If None, do not sort dimensions. If list, use provided order.
    legend : bool, optional
        Whether to include a legend in the plot, by default True.
    ax : matplotlib.Axes, optional
        Axes to plot on, by default None.
    """
    with plt.rc_context({"font.size": 14}):
        # RADVIZ plot
        if not ax:
            figsize = (5, 5)
            plt.figure(figsize=figsize)
            ax = plt.gca()

        edgecolor = sns.axes_style()["axes.edgecolor"]

        kde_cmap = "binary" if edgecolor == "black" else "binary_r"

        # Infer annot_color from theme
        if annot_color is None:
            annot_color = edgecolor

        # Remove unexpressed genes
        ndims = comp_stats.columns.get_loc("logcounts") - 1
        dims = comp_stats.columns[: ndims + 1]
        stat_cols = comp_stats.columns[ndims + 1 :]
        comp_stats = comp_stats[comp_stats[dims].sum(axis=1) > 0]

        # Determine best dimension ordering by maximizing cosine similarity of adjacent dimensions
        if not dim_order:
            dim_order = dims
        elif dim_order == "auto":
            dim_order = _sort_dimensions(comp_stats[dims])
        elif isinstance(dim_order, list):
            dim_order = dim_order
        else:
            raise ValueError(f"Invalid dim_order: {dim_order}")

        comp_stats = comp_stats.reindex([*dim_order, *stat_cols], axis=1)

        # Plot the "circular" axis, labels and point positions
        comp_stats["_"] = ""
        pd.plotting.radviz(comp_stats[[*dim_order, "_"]], "_", s=0, ax=ax)
        ax.get_legend().remove()

        # Get points
        pts = []
        for c in ax.collections:
            pts.extend(c.get_offsets().data)

        pts = np.array(pts).reshape(-1, 2)
        xy = pd.DataFrame(pts, index=comp_stats.index)

        # Get vertices and origin
        center = ax.patches[0]
        vertices = ax.patches[1:]

        # Add polygon as background
        poly = plt.Polygon(
            [v.center for v in vertices],
            facecolor="none",
            edgecolor=edgecolor,
            zorder=1,
        )
        ax.add_patch(poly)

        # Add lines from origin to vertices
        for v in vertices:
            line_xy = np.array([center.center, v.center])
            ax.add_line(
                plt.Line2D(
                    line_xy[:, 0],
                    line_xy[:, 1],
                    linestyle=":",
                    linewidth=1,
                    color=edgecolor,
                    zorder=1,
                    alpha=0.4,
                )
            )
            v.remove()

        # Hide 2D axes
        ax.axis(False)

        # Point size ~ percent of cells in group
        cell_fraction = comp_stats["cell_fraction"]
        cell_fraction = cell_fraction.apply(lambda x: round(x, 1))
        size_key = "Fraction of cells (%)"
        xy[size_key] = cell_fraction

        # Hue ~ mean log2(count + 1)
        log_count = comp_stats["logcounts"]
        hue_key = "Mean log2(cnt + 1)"
        xy[hue_key] = log_count

        # Remove data points with invalid values (nan or inf)
        xy = xy.replace([np.inf, -np.inf], np.nan).dropna()

        sns.kdeplot(
            data=xy,
            x=0,
            y=1,
            shade=True,
            cmap=kde_cmap,
            zorder=0.9,
            ax=ax,
        )

        # Filter genes by min threshold
        xy_filt = xy[xy[hue_key] >= min_count]

        # Plot points
        sns.scatterplot(
            data=xy_filt,
            x=0,
            y=1,
            hue=hue_key,
            palette=palette,
            size=size_key,
            sizes=sizes,
            size_norm=size_norm,
            linewidth=0.5,
            # alpha=0.6,
            edgecolor="white",
            legend=legend,
            ax=ax,
        )
        scatter = ax.collections[0]

        if legend:
            plt.legend(bbox_to_anchor=[1.1, 1], fontsize=10, frameon=False)

        # Annotate top points
        if annotate:
            if isinstance(annotate, int):
                # Get top ranked genes by entropy
                from scipy.stats import entropy

                top_genes = (
                    comp_stats.loc[:, dims]
                    .apply(lambda gene_comp: entropy(gene_comp), axis=1)
                    .sort_values(ascending=True)
                    .index[:annotate]
                )
                top_xy = xy.loc[top_genes]
            else:
                # Parse list of genes
                top_xy = xy.loc[[g for g in annotate if g in xy.index]]

            # Plot top points
            sns.scatterplot(
                data=top_xy,
                x=0,
                y=1,
                hue=hue_key,
                palette=palette,
                size=size_key,
                sizes=sizes,
                size_norm=size_norm,
                linewidth=1,
                facecolor=None,
                edgecolor=annot_color,
                legend=False,
                ax=ax,
            )

            # Add text labels
            if annot_color == "black":
                stroke_color = "white"
            elif annot_color == "white":
                stroke_color = "black"
            else:
                stroke_color = "black"
            texts = [
                ax.text(
                    row[0],
                    row[1],
                    i,
                    fontsize=8,
                    weight="medium",
                    path_effects=[pe.withStroke(linewidth=2, foreground=stroke_color)],
                )
                for i, row in top_xy.iterrows()
            ]

            # Adjust text positions
            if adjust:
                print("Adjusting text positions...")
                adjust_text(
                    texts,
                    # expand=(2, 2),
                    objects=[scatter],
                    ax=ax,
                    # arrowstyle="-",
                    # color=edgecolor,
                    # lw=1,
                )


def _sort_dimensions(composition):
    sim = cosine_similarity(composition.T, composition.T)
    sim = pd.DataFrame(sim, index=composition.columns, columns=composition.columns)
    dim_order = sim.sample(3, random_state=11).index.tolist()

    # Insert dimensions greedily
    for dim in sim.columns:
        if dim in dim_order:
            continue

        insert_score = []
        for dim_i, dim_j in zip([dim_order[-1]] + dim_order[:-1], dim_order):
            insert_score.append(np.mean(sim.loc[dim, [dim_i, dim_j]]))

        insert_pos = np.argmax(insert_score)
        dim_order.insert(insert_pos, dim)
    return dim_order
