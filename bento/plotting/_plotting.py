import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
from tqdm.auto import tqdm
from scipy.interpolate import griddata
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from sklearn.preprocessing import quantile_transform

from ._colors import red2blue_dark
from ..geometry import get_points
from ._utils import savefig


def quantiles(data, x, **kwargs):
    ax = plt.gca()

    ylims = ax.get_ylim()
    ymargin = 0.2 * (ylims[1] - ylims[0])
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
    fname=None,
):
    """Plot shape statistics for each cell. This is a wrapper around seaborn's pairplot.

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
        height=1,
        aspect=4,
        sharex=False,
        sharey=False,
        margin_titles=False,
    )
    g.map_dataframe(
        sns.stripplot,
        x="value",
        color="black",
        linewidth=0,
        s=2,
        alpha=0.3,
        rasterized=True,
    )
    g.map_dataframe(quantiles, x="value")
    g.add_legend()
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    for ax, var in zip(g.axes.flat, stats_long["variable"].unique()):
        # ax.spines["bottom"].set_position(("data", 0.3))
        ax.set_xlabel("")
        ax.set_ylabel(var, rotation=0, ha="right")
        ax.set_yticks([])
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
        sns.despine(ax=ax, left=True)
    g.set_titles(row_template="", col_template="")
    plt.subplots_adjust(hspace=0.03)

    def plot_mean(data, **kwargs):
        plt.axvline(data.mean(), **kwargs)

    g.map(plot_mean, "value", c="black", lw=0.5, zorder=3)


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
def plot(
    adata,
    kind="scatter",
    hue=None,
    col="batch",
    hue_order=None,
    col_wrap=None,
    col_order=None,
    height=3,
    theme="dark",
    palette=None,
    cmap=None,
    legend=True,
    frameon=True,
    point_kws=dict(),
    graph_kws=dict(),
    shape_kws=dict(),
    shape_names=["cell_shape", "nucleus_shape"],
    dx=0.1,
    units="um",
    title=True,
    fname=None,
):
    """
    Plot spatial data. This function wraps sns.FacetGrid to handle plotting multiple columns (no rows).

    Parameters
    ----------
    adata : AnnData
        Spatial formatted AnnData
    kind : {"hist", "scatter", "plot"}, optional
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
    if kind == "flow" and "flow_points" in adata.uns:
        points = adata.uns["flow_points"]
    else:
        points = get_points(adata, asgeo=False)

    # This feels weird here; refactor separate flow plotting?
    if kind == "flow":
        points[["flow1", "flow2", "flow3"]] = adata.uns["flow_embed"][:, :3]

        if "fe" in adata.uns:
            points[adata.uns["fe"].columns] = adata.uns["fe"].values

    # Include col if exists
    if col and (col == "cell" or (col in adata.obs.columns or col in points.columns)):
        obs_attrs.append(col)

        # TODO bug, col typeerror
        if col not in points.columns:
            points = points.set_index("cell").join(adata.obs[[col]]).reset_index()
    else:
        col = None

    # Transfer obs hue to points
    if hue and hue in adata.obs.columns and hue not in points.columns:
        points = points.set_index("cell").join(adata.obs[[hue]]).reset_index()
        obs_attrs.append(hue)

    obs_attrs = list(set(obs_attrs))

    # Get shapes
    shapes = adata.obs.reset_index()[obs_attrs]
    if "cell_shape" in shapes.columns:
        shapes = shapes.set_geometry("cell_shape")

    if col:
        # Make sure col is same type across points and shapes
        # if points[col].dtype != shapes[col].dtype:
        points[col] = points[col].astype(str)
        shapes[col] = shapes[col].astype(str)

        # Subset to specified col values only; less filtering = faster plotting
        if col_order:
            points = points[points[col].isin(col_order)]
            shapes = shapes[shapes[col].isin(col_order)]

        group_names, pt_groups = zip(*points.groupby(col))
        group_names, shape_groups = zip(*shapes.groupby(col))
        # Get subplot grid shape
        if col_wrap is not None:
            ncols = col_wrap
            nrows = int(np.ceil(len(group_names) / col_wrap))
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
            group_names, pt_groups = zip(*points.groupby(col))

        print("  + Transcripts")
        for ax, pt_group in zip(axes, pt_groups):
            if legend and ax == axes[-1]:
                legend = True

            _points(
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
                default_kws = dict(facecolor=(0, 0, 0, 0), edgecolor=edgecolor, lw=0.5)
                default_kws.update(shape_kws)
                _plot_shapes(
                    shape_group,
                    shape_names,
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


def _points(kind, hue, hue_order, palette, cmap, legend, ax, pt_group, **point_kws):
    if kind == "scatter" or kind == "graph":
        scatter_kws = dict(linewidth=0, s=1)
        scatter_kws.update(point_kws)
        # Matplotlib handle color
        if "c" in point_kws:
            collection = ax.scatter(x=pt_group["x"], y=pt_group["y"], **scatter_kws)
        # Convert hue hex colors to color column for matplotlib
        elif hue and str(pt_group[hue].values[0]).startswith("#"):
            scatter_kws["c"] = pt_group[hue]
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

    # TODO: currently only molecule points are passed to this function
    elif kind == "flow":
        flow_kws = dict(method="linear")
        flow_kws.update(**point_kws)
        _flow(
            points=pt_group,
            hue=hue,
            cmap=cmap,
            ax=ax,
            **flow_kws,
        )
    else:
        raise ValueError(f"Invalid kind: {kind}")

    # Add colorbar to last subplot
    if legend and kind in ["scatter", "hist", "graph"]:
        if kind == "scatter" and "c" in point_kws:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0)
            plt.colorbar(collection, cax=cax, orientation="horizontal")
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)


def _plot_shapes(
    data,
    shape_names,
    legend,
    ax,
    **kwargs,
):
    # Gather all shapes and plot
    for shape_name in shape_names:
        data.reset_index().set_geometry(shape_name).plot(
            aspect=None,
            ax=ax,
            zorder=3,
            **kwargs,
        )


def _flow(points, hue, cmap, method, ax, **kwargs):

    if hue is None:
        components = points[["flow1", "flow2", "flow3"]].values
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

    plt.suptitle(hue, fontsize=12)


def sig_samples(data, rank, n_genes=5, n_cells=4, col_wrap=4, **kwargs):
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
            col="cell",
            col_wrap=col_wrap,
            height=2,
            **kwargs,
        )
        plt.suptitle(f)
