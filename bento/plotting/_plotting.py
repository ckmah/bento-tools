import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from sklearn.preprocessing import quantile_transform
from sklearn.metrics.pairwise import cosine_similarity
from adjustText import adjust_text

from ..geometry import get_points
from ..plotting._colors import red_light
from ._utils import savefig


from ..tools._composition import _get_compositions


def quantiles(data, x, **kwargs):
    ax = plt.gca()

    ylims = ax.get_ylim()
    ymargin = 0.3 * (ylims[1] - ylims[0])
    quants = np.percentile(data[x], [0, 1, 25, 50, 75, 99, 100])
    palette = sns.color_palette("red2blue", n_colors=len(quants) - 1)

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
            edgecolor="black",
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
    s=5,
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

    g = sns.FacetGrid(
        data=stats_long,
        row="variable",
        height=1.2,
        aspect=2,
        sharex=False,
        sharey=False,
        margin_titles=False,
    )
    g.map_dataframe(
        sns.stripplot,
        x="value",
        color="black",
        s=s,
        alpha=alpha,
        rasterized=True,
    )
    g.map_dataframe(quantiles, x="value")
    g.add_legend()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    for ax, var in zip(g.axes.flat, stats_long["variable"].unique()):
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
        sns.despine(ax=ax, left=True)
    g.set_titles(template="{row_name}", row_template="", col_template="")

    def plot_median(data, **kwargs):
        plt.axvline(data.median(), **kwargs)

    g.map(plot_median, "value", c="black", lw=1.5, zorder=3)


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

    fig, axs = plt.subplots(
        3, 3, figsize=(9, 9), gridspec_kw=dict(wspace=0.5, hspace=1)
    )
    kde_params = dict(color=color, shade=True, legend=False)

    with sns.axes_style("ticks"):
        sns.kdeplot(adata.obs["total_counts"], ax=axs.flat[0], **kde_params)
        sns.histplot(
            adata.X.flatten(),
            color=color,
            bins=20,
            log_scale=(False, True),
            ax=axs.flat[1],
        )
        sns.kdeplot(adata.obs["n_genes_by_counts"], ax=axs.flat[2], **kde_params)
        sns.kdeplot(adata.obs["cell_area"], ax=axs.flat[3], **kde_params)
        sns.kdeplot(adata.obs["cell_density"], ax=axs.flat[4], **kde_params)
        sns.kdeplot(
            adata.obs["nucleus_area"] / adata.obs["cell_area"],
            ax=axs.flat[5],
            **kde_params,
        )

        dual_colors = sns.light_palette(color, n_colors=2, reverse=True)
        no_nucleus_count = (adata.obs["nucleus_shape"] == None).sum()
        pie_values = [adata.n_obs - no_nucleus_count, no_nucleus_count]
        pie_percents = np.array(pie_values) / adata.n_obs * 100
        pie_labels = [
            f"Yes\n{pie_values[0]} ({pie_percents[0]:.1f}%)",
            f"No\n{pie_values[1]} ({pie_percents[1]:.1f}%)",
        ]
        axs.flat[6].pie(pie_values, labels=pie_labels, colors=dual_colors)
        pie_inner = plt.Circle((0, 0), 0.6, color="white")
        axs.flat[6].add_artist(pie_inner)

    titles = [
        "molecules / cell",
        "molecules / gene",
        "genes / cell",
        "cell area",
        "molecule density",
        "cell has nucleus",
    ]
    xlabels = [
        "Molecules",
        "Molecules",
        "Gene count",
        "Pixels",
        "Transcripts/pixel",
        "",
    ]

    for i, ax in enumerate(axs.flat):
        if i != 1:
            plt.setp(ax, xlabel=xlabels[i], ylabel="", yticks=[])
            sns.despine(left=True)
        else:
            plt.setp(ax, xlabel=xlabels[i], ylabel=None)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
        ax.set_title(titles[i], fontsize=12)
        #     0,
        #     1 - (1 / 12) * (2 * i + 1),
        #     titles[i],
        #     ha="right",
        #     va="center",
        #     fontsize=12,
        # )
        ax.grid(False)

    plt.tight_layout()


@savefig
def flux_summary(
    data,
    groupby=None,
    group_order=None,
    annotate=None,
    adjust=True,
    palette=red_light,
    annot_color="black",
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
    annot_color="black",
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
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            fig = ax.get_figure()

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
        pd.plotting.radviz(comp_stats[[*dims, "_"]], "_", s=20, ax=ax)
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
            edgecolor="black",
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
                    color="black",
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

        # Hue ~ mean log2(count = 1)
        log_count = comp_stats["logcounts"]
        hue_key = "Mean log2(cnt + 1)\n in group"
        xy[hue_key] = log_count

        # Remove phantom points
        del ax.collections[0]

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
            texts = [
                ax.text(
                    row[0],
                    row[1],
                    i,
                    fontsize=8,
                    weight="medium",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
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
                    arrowprops=dict(arrowstyle="-", color="black", lw=1),
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


@savefig
def plot(
    adata,
    points_key="points",
    kind="scatter",
    groupby="batch",
    hue=None,
    group_wrap=None,
    group_order=None,
    hue_order=None,
    theme="dark",
    palette=None,
    cmap=None,
    shape_names=["cell_shape", "nucleus_shape"],
    facecolors=None,
    edgecolors=None,
    linewidths=None,
    height=6,
    dx=0.1,
    units="um",
    legend=True,
    frameon=True,
    title=True,
    point_kws=dict(),
    graph_kws=dict(),
    shape_kws=dict(),
    fname=None,
):
    """
    Plot spatial data. This function wraps sns.FacetGrid to handle plotting multiple columns (no rows).

    Parameters
    ----------
    adata : AnnData
        Spatial formatted AnnData
    kind : {"hist", "scatter", "interpolate"}, optional
        Plotting method, by default "scatter"

    """

    # Set style
    if theme == "dark":
        style = "dark_background"
        facecolor = "black"
        edgecolor = "white"
        textcolor = "white"
    elif theme == "light":
        style = "default"
        facecolor = "white"
        edgecolor = "black"
        textcolor = "black"
    else:
        return ValueError("Theme must be 'dark' or 'light'.")

    if shape_names is None:
        shape_names = []

    # Convert shape_names to list
    shape_names = [shape_names] if isinstance(shape_names, str) else shape_names

    # Get obs attributes starting with shapes
    obs_attrs = list(shape_names)

    # Get points
    points = get_points(adata, key=points_key, asgeo=False)

    # Filter for genes
    if points_key == "points":
        points = points[points["gene"].isin(adata.var_names)]
        # Remove unused categories
        points["gene"] = (
            points["gene"].astype("category").cat.remove_unused_categories()
        )

    # Add functional enrichment if exists
    if "flow_fe" in adata.uns and adata.uns["flow_fe"].shape[0] == points.shape[0]:
        points[adata.uns["flow_fe"].columns] = adata.uns["flow_fe"].values

    # This feels weird here; refactor separate flow plotting?
    if kind == "interpolate":
        points[["vis1", "vis2", "vis3"]] = adata.uns["flow_vis"]

    # Include col if exists
    if groupby and (
        groupby == "cell" or (groupby in adata.obs.columns or groupby in points.columns)
    ):
        obs_attrs.append(groupby)

        # TODO bug, col typeerror
        if groupby not in points.columns:
            points = points.set_index("cell").join(adata.obs[[groupby]]).reset_index()
    else:
        groupby = None

    # Transfer obs hue to points
    if hue and hue in adata.obs.columns and hue not in points.columns:
        points = points.set_index("cell").join(adata.obs[[hue]]).reset_index()
        obs_attrs.append(hue)

    obs_attrs = list(set(obs_attrs))

    # Get shapes
    shapes = adata.obs.reset_index()[obs_attrs]
    if "cell_shape" in shapes.columns:
        shapes = shapes.set_geometry("cell_shape")

    if groupby:
        # Make sure col is same type across points and shapes
        # if points[col].dtype != shapes[col].dtype:
        points[groupby] = points[groupby].astype(str)
        shapes[groupby] = shapes[groupby].astype(str)

        # Subset to specified col values only; less filtering = faster plotting
        if group_order:
            points = points[points[groupby].isin(group_order)]
            shapes = shapes[shapes[groupby].isin(group_order)]

        group_names, pt_groups = zip(*points.groupby(groupby))
        group_names, shape_groups = zip(*shapes.groupby(groupby))
        # Get subplot grid shape
        if group_wrap is not None:
            ncols = group_wrap
            nrows = int(np.ceil(len(group_names) / group_wrap))
        else:
            ncols = len(group_names)
            nrows = 1
    else:
        group_names = [""]
        pt_groups = [points]
        shape_groups = [shapes]

        ncols = 1
        nrows = 1

    with plt.style.context(style):
        print("Plotting layers:")
        # https://stackoverflow.com/questions/32633322/changing-aspect-ratio-of-subplots-in-matplotlib
        fig_width = ncols * height
        fig_height = nrows * height
        figsize = (fig_width, fig_height)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if len(group_names) == 1:
            axes = [axes]
        else:
            if nrows > 1:
                axes = list(axes.flat)
            group_names, pt_groups = zip(*points.groupby(groupby))

        # Plot shapes if any
        if isinstance(shape_groups[0], gpd.GeoDataFrame):
            print("  + Shapes")

            # Get max radius across groups
            ax_radii = []
            for shape_group in shape_groups:
                # Determine fixed radius of each subplot
                cell_bounds = shape_group.bounds
                cell_maxw = cell_bounds["maxx"].max() - cell_bounds["minx"].min()
                cell_maxh = cell_bounds["maxy"].max() - cell_bounds["miny"].min()
                ax_radius = 1.05 * (max(cell_maxw, cell_maxh) / 2)
                ax_radii.append(ax_radius)

            ax_radius = max(ax_radii)

            for shape_group, ax in zip(shape_groups, axes):
                default_kws = dict()
                default_kws.update(shape_kws)

                if facecolors is None:
                    facecolors = [(0, 0, 0, 0)] * len(shape_names)

                if edgecolors is None:
                    edgecolors = [edgecolor] * len(shape_names)

                if linewidths is None:
                    linewidths = [0.5] * len(shape_names)

                _plot_shapes(
                    shape_group,
                    shape_names,
                    facecolors,
                    edgecolors,
                    linewidths,
                    legend,
                    ax,
                    **default_kws,
                )

                # Set axis limits
                xmin, xmax, ymin, ymax = ax.axis()
                xmin -= ax_radius * 0.02
                ymin -= ax_radius * 0.02
                xmax += ax_radius * 0.02
                ymax += ax_radius * 0.02
                xcenter = (xmax + xmin) / 2
                ycenter = (ymax + ymin) / 2
                ax.set_xlim(xcenter - ax_radius, xcenter + ax_radius)
                ax.set_ylim(ycenter - ax_radius, ycenter + ax_radius)

                # Mask outside cells
                rect_bound = gpd.GeoDataFrame(
                    geometry=[
                        Polygon(
                            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
                        )
                    ]
                )
                rect_bound.overlay(shape_group, how="difference").plot(
                    facecolor=facecolor, edgecolor=None, zorder=2, ax=ax
                )

        print("  + Transcripts")
        for ax, pt_group in zip(axes, pt_groups):
            show_legend = False
            if legend and ax == axes[-1]:
                show_legend = legend

            _plot_points(
                kind, hue, hue_order, palette, cmap, legend, ax, pt_group, **point_kws
            )

            # plot graph layer
            if kind == "graph":
                print("  + Graphs")
                default_kws = dict(
                    radius=20, edge_color=edgecolor, width=0.5, alpha=0.3, node_size=0
                )
                default_kws.update(graph_kws)
                _graphs(pt_group, ax, **default_kws)

        # Formatting subplots
        for ax, group_name in zip(axes, group_names):

            # Create scale bar
            scalebar = ScaleBar(
                dx,
                units,
                location="lower right",
                color=textcolor,
                box_alpha=0,
                scale_loc="top",
            )
            ax.add_artist(scalebar)

            # Add title
            if title:
                plt.text(
                    0.02,
                    0.98,
                    group_name,
                    ha="left",
                    va="top",
                    color=textcolor,
                    transform=ax.transAxes,
                )
            ax.spines[["top", "right", "bottom", "left"]].set_visible(frameon)
            ax.axis(frameon)

        # box_aspect for Axes, aspect for data
        if len(axes) > 1:
            box_aspect = 1
        else:
            box_aspect = None

        plt.setp(
            axes,
            xticks=[],
            yticks=[],
            xticklabels=[],
            yticklabels=[],
            xlabel=None,
            ylabel=None,
            xmargin=0,
            ymargin=0,
            facecolor=facecolor,
            box_aspect=box_aspect,
            aspect=1,
        )
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.setp(fig.patch, facecolor=facecolor)
        print("Done.")


def _plot_points(
    kind, hue, hue_order, palette, cmap, legend, ax, pt_group, **point_kws
):
    if kind == "scatter" or kind == "graph":
        scatter_kws = dict(linewidth=0, s=1)
        scatter_kws.update(point_kws)

        M = ax.transData.get_matrix()
        xscale = M[0, 0]
        yscale = M[1, 1]
        # has desired_data_width of width
        scatter_kws["s"] = (xscale * scatter_kws["s"]) ** 2

        # Matplotlib handle color
        if "c" in point_kws:
            collection = ax.scatter(x=pt_group["x"], y=pt_group["y"], **scatter_kws)
        # Use seaborn for hue mapping
        else:
            collection = sns.scatterplot(
                data=pt_group,
                x="x",
                y="y",
                hue=hue,
                hue_order=hue_order,
                palette=palette,
                legend=legend,
                ax=ax,
                **scatter_kws,
            ).collections[0]

    elif kind == "hist":
        hist_kws = dict(binwidth=15)
        hist_kws.update(**point_kws)
        collection = sns.histplot(
            data=pt_group,
            x="x",
            y="y",
            hue=hue,
            palette=palette,
            ax=ax,
            **hist_kws,
        ).collections[0]

    elif kind == "interpolate":
        kws = dict(method="linear")
        kws.update(**point_kws)
        _interpolate(
            points=pt_group,
            hue=hue,
            cmap=cmap,
            ax=ax,
            **kws,
        )
    else:
        raise ValueError(f"Invalid kind: {kind}")

    # Add colorbar to last subplot
    # if legend and kind in ["scatter", "hist", "graph"]:
    if legend:
        divider = make_axes_locatable(ax)
        if kind == "scatter" and "c" in point_kws:
            cax = divider.append_axes("bottom", size="5%", pad=0)
            plt.colorbar(collection, cax=cax, orientation="horizontal")
        elif kind == "interpolate" and hue:
            cax = divider.append_axes("bottom", size="5%", pad=0)
            plt.colorbar(ax.get_images()[0], cax=cax, orientation="horizontal")
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)


def _graphs(pt_group, ax, **graph_kws):

    # Convert points neighborhood adjacency list
    neighbor_index = (
        NearestNeighbors(radius=graph_kws["radius"], n_jobs=-1)
        .fit(pt_group[["x", "y"]])
        .radius_neighbors(pt_group[["x", "y"]], return_distance=False)
    )

    # Create networkx graph from adjacency list
    pt_graph = nx.Graph(dict(zip(range(len(neighbor_index)), neighbor_index)))
    pt_graph.remove_edges_from(nx.selfloop_edges(pt_graph))

    positions = dict(zip(pt_graph.nodes, pt_group[["x", "y"]].values))

    del graph_kws["radius"]
    collection = nx.draw_networkx_edges(
        pt_graph,
        pos=positions,
        ax=ax,
        **graph_kws,
    )
    collection.set_zorder(0.99)


def _plot_shapes(
    data,
    shape_names,
    facecolors,
    edgecolors,
    linewidths,
    legend,
    ax,
    **kwargs,
):
    # Gather all shapes and plot
    for shape_name, facecolor, edgecolor, linewidth in zip(
        shape_names, facecolors, edgecolors, linewidths
    ):
        zorder = 3
        if shape_name == "cell_shape":
            zorder = 3.5

        data.reset_index().set_geometry(shape_name).plot(
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            aspect=None,
            ax=ax,
            zorder=zorder,
            **kwargs,
        )


def _interpolate(points, hue, cmap, method, ax, **kwargs):

    if hue is None:
        components = points[["vis1", "vis2", "vis3"]].values
    else:
        components = points[hue].values.reshape(-1, 1)

    # Get subplot xy grid bounds
    minx, maxx = points["x"].min(), points["x"].max()
    miny, maxy = points["y"].min(), points["y"].max()

    # Infer step size by finding the smallest distance between the two smallest values of x
    unique_x = np.unique(points["x"])
    step = abs(unique_x[1] - unique_x[0])

    # Define grid coordinates
    grid_x, grid_y = np.mgrid[
        minx : maxx + step : step,
        miny : maxy + step : step,
    ]
    values = []

    # Interpolate values for each channel
    for cp in range(components.shape[1]):
        values.append(
            griddata(
                points[["x", "y"]].values,
                components[:, cp],
                (grid_x, grid_y),
                method=method,
                fill_value=0,
            ).T
        )

    values = np.stack(values, axis=-1)
    # interpolating can cause values to be outside of [0, 1] range; clip values for rgb images only
    if values.shape[2] > 1:
        values = np.clip(values, 0, 1)

    ax.imshow(
        values, extent=(minx, maxx, miny, maxy), origin="lower", cmap=cmap, **kwargs
    )
    ax.autoscale(False)


def sig_samples(data, rank, n_genes=5, n_cells=4, group_wrap=4, **kwargs):
    for f in data.uns[f"r{rank}_signatures"].columns:

        top_cells = (
            data.obsm[f"r{rank}_signatures"]
            .sort_values(f, ascending=False)
            .index.tolist()[:n_cells]
        )

        top_genes = (
            data.varm[f"r{rank}_signatures"]
            .sort_values(f, ascending=False)
            .index.tolist()[:n_genes]
        )

        plot(
            data[top_cells, top_genes],
            kind="scatter",
            hue="gene",
            groupby="cell",
            group_wrap=group_wrap,
            height=2,
            **kwargs,
        )
        plt.suptitle(f)
