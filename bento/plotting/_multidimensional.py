import warnings

warnings.filterwarnings("ignore")

import matplotlib as mpl
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
    quants = np.percentile(data[x], [0, 25, 50, 75, 100])
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
def obs_stats(
    data,
    obs_cols=[
        "cell_area",
        "cell_aspect_ratio",
        "cell_density",
        "nucleus_area",
        "nucleus_aspect_ratio",
        "nucleus_density",
    ],
    s=3,
    alpha=0.3,
    rug=False,
    fname=None,
):
    """Plot shape statistic distributions for each cell.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    cols : list
        List of obs columns to plot
    groupby : str, optional
        Column in obs to groupby, by default None
    """
    stats_long = data.obs.melt(value_vars=obs_cols)
    stats_long["quantile"] = stats_long.groupby("variable")["value"].transform(
        lambda x: quantile_transform(x.values.reshape(-1, 1), n_quantiles=100).flatten()
    )

    stats_long["shape"] = stats_long["variable"].apply(lambda x: x.split("_")[0])
    stats_long["var"] = stats_long["variable"].apply(
        lambda x: "_".join(x.split("_")[1:])
    )

    linecolor = sns.axes_style()["axes.edgecolor"]

    g = sns.FacetGrid(
        data=stats_long,
        row="var",
        col="shape",
        height=1.2,
        aspect=2,
        sharex=False,
        sharey=False,
        margin_titles=True,
    )
    g.map_dataframe(
        sns.stripplot,
        x="value",
        color=linecolor,
        s=s,
        alpha=alpha,
        rasterized=True,
    )
    g.map_dataframe(_quantiles, x="value")
    g.add_legend()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    for ax, var in zip(g.axes.flat, stats_long["variable"].unique()):
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
        sns.despine(ax=ax, left=True)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    def plot_median(data, **kwargs):
        plt.axvline(data.median(), **kwargs)

    g.map(plot_median, "value", c=linecolor, lw=1.5, zorder=3)


@savefig
def flux_summary(
    data,
    groupby=None,
    group_order=None,
    annotate=None,
    adjust=True,
    palette=red_light,
    annot_color=None,
    sizes=(5, 30),
    size_norm=(10, 100),
    dim_order=None,
    legend=True,
    height=5,
    fname=None,
):
    """
    Plot RNAflux summary with a radviz plot describing gene embedding across flux clusters.
    """

    comp_key = f"{groupby}_comp_stats"
    if groupby and comp_key in data.uns.keys():
        comp_stats = data.uns[comp_key]
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
        return _radviz(
            comp_stats,
            annotate=annotate,
            adjust=adjust,
            palette=palette,
            annot_color=annot_color,
            sizes=sizes,
            size_norm=size_norm,
            dim_order=dim_order,
            legend=legend,
        )


def _radviz(
    comp_stats,
    annotate=None,
    adjust=True,
    palette=red_light,
    annot_color=None,
    sizes=None,
    size_norm=None,
    dim_order="auto",
    legend=True,
    ax=None,
):
    """Plot a radviz plot of gene values across fields.

    Parameters
    ----------
    comp_stats : DataFrame
        Gene composition stats
    palette : str, optional
        Color palette, by default None
    sizes : tuple, optional
        Size range for scatter plot, by default None
    size_norm : tuple, optional
        Size range for scatter plot, by default None
    dim_order : "auto", None, or list, optional
        Sort dimensions for more intuitive visualization, by default "auto".
        If "auto", sort dimensions by maximizing cosine similarity of adjacent
        dimensions. If None, do not sort dimensions. If list, use provided order.
    gridsize : int, optional
        Gridsize for hexbin plot, by default 20
    ax : matplotlib.Axes, optional
        Axes to plot on, by default None
    """
    with plt.rc_context({"font.size": 14}):
        # RADVIZ plot
        if not ax:
            figsize = (5, 5)
            plt.figure(figsize=figsize)
            ax = plt.gca()

        edgecolor = sns.axes_style()["axes.edgecolor"]

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
        size_key = "Fraction of cells\n in group (%)"
        xy[size_key] = cell_fraction

        # Hue ~ mean log2(count + 1)
        log_count = comp_stats["logcounts"]
        hue_key = "Mean log2(cnt + 1)\n in group"
        xy[hue_key] = log_count

        # Remove phantom points
        # ax.collections = ax.collections[1:]

        sns.kdeplot(
            data=xy,
            x=0,
            y=1,
            shade=True,
            cmap="binary",
            zorder=0.9,
            ax=ax,
        )

        # Plot points
        sns.scatterplot(
            data=xy,
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
                    expand_points=(2, 2),
                    add_objects=[scatter],
                    arrowprops=dict(arrowstyle="-", color=edgecolor, lw=1),
                    ax=ax,
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
