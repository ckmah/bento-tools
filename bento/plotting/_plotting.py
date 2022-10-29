import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
from sklearn.preprocessing import quantile_transform
from tqdm.auto import tqdm
from scipy.interpolate import griddata

from ..preprocessing import get_points
from ._utils import savefig


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
def plot(
    adata,
    kind="scatter",
    col="batch",
    hue=None,
    col_wrap=None,
    height=3,
    aspect=1,
    theme="dark",
    cmap="viridis",
    palette=None,
    col_order=None,
    legend=True,
    frameon=True,
    subplot_kws=None,
    gridspec_kws=None,
    shape_names=["cell_shape", "nucleus_shape"],
    lw=1,
    dx=0.1,
    units="um",
    fname=None,
    **kwargs,
):
    """
    Plot spatial data. This function wraps sns.FacetGrid to handle plotting multiple columns (no rows).

    Parameters
    ----------
    adata : AnnData
        Spatial formatted AnnData
    kind : {"hist", "scatter", "hex"}, optional
        Approach for visualizing the data., by default "hist"
    col : str, optional
        Column variable, by default "batch"
    hue : str, optional
        Hue variable, by default None
    col_wrap : int, optional
        Wrap column variable, by default None
    sharex : bool, optional
        Share x axis, by default False
    sharey : bool, optional
        Share y axis, by default False
    height : int, optional
        Height of each facet, by default 3
    aspect : int, optional
        Aspect ratio of each facet, by default 1
    palette : str, optional
        Color palette, by default None
    col_order : list, optional
        Column order, by default None
    hue_order : list, optional
        Hue order, by default None
    dropna : bool, optional
        Drop NA values, by default False
    legend : bool, optional
        Show legend, by default True
    despine : bool, optional
        Remove spines, by default True
    margin_titles : bool, optional
        Show margin titles, by default False
    xlim : tuple, optional
        X axis limits, by default None
    ylim : tuple, optional
        Y axis limits, by default None
    subplot_kws : dict, optional
        Subplot keyword arguments, by default None
    gridspec_kws : dict, optional
        Gridspec keyword arguments, by default None
    shape_names : list, optional
        Shape names, by default ["cell_shape", "nucleus_shape"]
    lw : int, optional
        Line width of shapes, by default 1
    dx : float, optional
        Pixel size for scaling scalebar, by default 0.1
    units : str, optional
        Units of scalebar dx, by default "um"
    fname : str, optional
        Save the figure to specified filename, by default None
    """

    # Get points
    points = get_points(adata, asgeo=False)

    # Check theme
    if theme not in ["dark", "light"]:
        print("Theme must be 'dark' or 'light'.")
        return

    if theme == "dark":
        style = "dark_background"
        facecolor = "black"
        edgecolor = "white"
        textcolor = "white"
    else:
        style = "default"
        facecolor = "white"
        edgecolor = "black"
        textcolor = "black"

    # Add all shape_names if None
    if shape_names is None:
        shape_names = adata.obs.columns[adata.obs.columns.str.endswith("_shape")]

    # Convert shape_names to list
    shape_names = [shape_names] if isinstance(shape_names, str) else shape_names

    # Get obs attributes starting with shapes
    obs_attrs = list(shape_names)

    # Include col if exists
    if col and (col == "cell" or (col in adata.obs.columns or col in points.columns)):
        obs_attrs.append(col)

        if col not in points.columns:
            points = (
                points.set_index("cell")
                .join(adata.obs[[col]].reset_index(), on="cell")
                .reset_index()
            )
    else:
        col = None

    # Transfer obs hue to points
    if hue and hue in adata.obs.columns and hue not in points.columns:
        points = points.set_index("cell").join(adata.obs[[hue]]).reset_index()
        obs_attrs.append(hue)

    if (
        hue
        and "fe" in adata.uns
        and hue in adata.uns["fe"]
        and hue not in points.columns
    ):
        enrichment = adata.uns["fe"]
        points[hue] = enrichment[hue].values

    obs_attrs = list(set(obs_attrs))

    # Get shapes
    shapes = adata.obs.reset_index()[obs_attrs]
    shapes = gpd.GeoDataFrame(shapes, geometry="cell_shape")

    ncols = 1
    nrows = 1

    group_names = [""]
    pt_groups = [points]
    shape_groups = [shapes]

    if col:
        # Make sure col is same type across points and shapes
        if points[col].dtype != shapes[col].dtype:
            points[col] = points[col].astype(str)
            shapes[col] = shapes[col].astype(str)

        # Subset to specified col values only; less filtering = faster plotting
        if col_order:
            points = points[points[col].isin(col_order)]
            shapes = shapes[shapes[col].isin(col_order)]

        group_names, pt_groups = zip(*points.groupby(col))
        group_names, shape_groups = zip(*shapes.groupby(col))

        if col_wrap is not None:
            ncols = col_wrap
            nrows = int(np.ceil(len(group_names) / col_wrap))
        else:
            ncols = len(group_names)

    with plt.style.context(style):

        # https://stackoverflow.com/questions/32633322/changing-aspect-ratio-of-subplots-in-matplotlib
        fig_width = ncols * height * aspect
        fig_height = nrows * height
        figsize = (fig_width, fig_height)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if len(group_names) == 1:
            axes = [axes]
        else:
            if nrows > 1:
                axes = list(axes.flat)
            group_names, pt_groups = zip(*points.groupby(col))

        for ax, pt_group in zip(axes, pt_groups):

            if kind == "scatter":
                scatter_kws = dict(linewidths=0, marker="o", cmap=cmap, s=1)
                scatter_kws.update(**kwargs)
                if hue:
                    scatter_kws["c"] = pt_group[hue]
                collection = ax.scatter(x=pt_group["x"], y=pt_group["y"], **scatter_kws)

            elif kind == "hist":
                hist_kws = dict(cmap=cmap, binwidth=15)
                hist_kws.update(**kwargs)
                collection = sns.histplot(
                    data=pt_group, x="x", y="y", ax=ax, **hist_kws
                )

            elif kind == "hex":
                hex_kws = dict(cmap=cmap, mincnt=1, linewidth=0, gridsize=100)
                hex_kws.update(**kwargs)
                collection = ax.hexbin(x=pt_group["x"], y=pt_group["y"], **hex_kws)

            # Add colorbar
            if legend and kind in ["scatter", "hist", "hex"]:
                if kind == "scatter" and (hue == "embed_color" or not hue):
                    pass
                else:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="5%", pad=0)
                    fig.colorbar(collection, cax=cax, orientation="horizontal")

        # Defer to geopandas plot for cell coloring
        if kind == "cell":
            cell_hue = hue
        # Disable otherwise
        else:
            cell_hue = None
            cmap = None

        if len(shape_groups) > 0:

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

            # Add loading bar if more than 1 group
            if len(axes) == 0:
                shape_zip = zip(shape_groups, axes)
            else:
                shape_zip = tqdm(zip(shape_groups, axes), total=len(shape_groups))

            for shape_group, ax in shape_zip:
                _plot_shapes(
                    shape_group,
                    shape_names,
                    lw,
                    dx,
                    units,
                    hue=cell_hue,
                    cmap=cmap,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    textcolor=textcolor,
                    legend=legend,
                    ax=ax,
                    ax_radius=ax_radius,
                    **kwargs,
                )

        else:
            _plot_shapes(
                shape_group,
                shape_names,
                lw,
                dx,
                units,
                hue=cell_hue,
                cmap=cmap,
                facecolor=facecolor,
                edgecolor=edgecolor,
                textcolor=textcolor,
                legend=legend,
                ax=axes[0],
                **kwargs,
            )

        # Formatting subplots

        for ax, group_name in zip(axes, group_names):
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
            box_aspect=1,
            aspect=1,
        )
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.setp(fig.patch, facecolor=facecolor)


@savefig
def flow(
    adata,
    col="batch",
    hue=None,
    col_wrap=None,
    height=3,
    aspect=1,
    method="cubic",
    resolution="high",
    theme="dark",
    cmap="viridis",
    col_order=None,
    frameon=True,
    subplot_kws=None,
    gridspec_kws=None,
    shape_names=["cell_shape", "nucleus_shape"],
    lw=1,
    dx=0.1,
    units="um",
    fname=None,
    **kwargs,
):
    """
    Plot RNAFlow `bento.tl.flow` gradients.

    Parameters
    ----------
    adata : AnnData
        Spatial formatted AnnData
    col : str, optional
        Column variable, by default "batch"
    hue : str, optional
        Hue variable, by default None
    col_wrap : int, optional
        Wrap column variable, by default None
    height : int, optional
        Height of each facet, by default 3
    palette : str, optional
        Color palette, by default None
    col_order : list, optional
        Column order, by default None
    frameon : bool, optional
        Show frame, by default True
    subplot_kws : dict, optional
        Subplot keyword arguments, by default None
    gridspec_kws : dict, optional
        Gridspec keyword arguments, by default None
    shape_names : list, optional
        Shape names, by default ["cell_shape", "nucleus_shape"]
    lw : int, optional
        Line width of shapes, by default 1
    dx : float, optional
        Pixel size for scaling scalebar, by default 0.1
    units : str, optional
        Units of scalebar dx, by default "um"
    fname : str, optional
        Save the figure to specified filename, by default None
    """

    # Get points
    points = adata.uns["flow_points"]

    # Check theme
    if theme not in ["dark", "light"]:
        print("Theme must be 'dark' or 'light'.")
        return

    if theme == "dark":
        style = "dark_background"
        facecolor = "black"
        edgecolor = "white"
        textcolor = "white"
    else:
        style = "default"
        facecolor = "white"
        edgecolor = "black"
        textcolor = "black"

    # Add all shape_names if None
    if shape_names is None:
        shape_names = adata.obs.columns[adata.obs.columns.str.endswith("_shape")]

    # Convert shape_names to list
    shape_names = [shape_names] if isinstance(shape_names, str) else shape_names

    # Get obs attributes starting with shapes
    obs_attrs = list(shape_names)

    # Include col if exists
    if col and (col == "cell" or (col in adata.obs.columns or col in points.columns)):
        obs_attrs.append(col)

        if col not in points.columns:
            points = points.set_index("cell").join(adata.obs[[col]]).reset_index()
    else:
        col = None

    # TODO refactor fe
    if (
        hue
        and "fe" in adata.uns
        and hue in adata.uns["fe"]
        and hue not in points.columns
    ):
        enrichment = adata.uns["fe"]
        points[hue] = enrichment[hue].values

    obs_attrs = list(set(obs_attrs))

    # Get shapes
    shapes = adata.obs.reset_index()[obs_attrs]
    shapes = gpd.GeoDataFrame(shapes, geometry="cell_shape")

    # Scale first 3 components of embedding
    flow_embed = adata.uns["flow_pca"]
    colors_rgb = quantile_transform(flow_embed[:, :3])
    points[["c1", "c2", "c3"]] = colors_rgb

    ncols = 1
    nrows = 1

    group_names = [""]
    pt_groups = [points]
    shape_groups = [shapes]

    if col:
        # Make sure col is same type across points and shapes
        if points[col].dtype != shapes[col].dtype:
            points[col] = points[col].astype(str)
            shapes[col] = shapes[col].astype(str)

        # Subset to specified col values only; less filtering = faster plotting
        if col_order:
            points = points[points[col].isin(col_order)]
            shapes = shapes[shapes[col].isin(col_order)]

        group_names, pt_groups = zip(*points.groupby(col))
        group_names, shape_groups = zip(*shapes.groupby(col))

        if col_wrap is not None:
            ncols = col_wrap
            nrows = int(np.ceil(len(group_names) / col_wrap))
        else:
            ncols = len(group_names)

    with plt.style.context(style):

        # https://stackoverflow.com/questions/32633322/changing-aspect-ratio-of-subplots-in-matplotlib
        fig_width = ncols * height * aspect
        fig_height = nrows * height
        figsize = (fig_width, fig_height)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if len(group_names) == 1:
            axes = [axes]
        else:
            if nrows > 1:
                axes = list(axes.flat)
            group_names, pt_groups = zip(*points.groupby(col))

        for ax, pt_group in zip(axes, pt_groups):
            # art_kws.update(**kwargs)
            _flow(
                data=pt_group, x="x", y="y", resolution=resolution, method=method, ax=ax
            )

            # TODO Add colorbar for enrichment stuff
            # if legend and kind in ["scatter", "hist", "hex"]:
            #     if kind == "scatter" and (hue == "embed_color" or not hue):
            #         pass
            #     else:
            #         divider = make_axes_locatable(ax)
            #         cax = divider.append_axes("bottom", size="5%", pad=0)
            #         fig.colorbar(collection, cax=cax, orientation="horizontal")

        if len(shape_groups) > 0:

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

            # Add loading bar if more than 1 group
            if len(axes) == 0:
                shape_zip = zip(shape_groups, axes)
            else:
                shape_zip = tqdm(zip(shape_groups, axes), total=len(shape_groups))

            for shape_group, ax in shape_zip:
                _plot_shapes(
                    shape_group,
                    shape_names,
                    lw,
                    dx,
                    units,
                    hue=None,
                    cmap=None,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    textcolor=textcolor,
                    legend=False,
                    ax=ax,
                    ax_radius=ax_radius,
                    **kwargs,
                )

        else:
            _plot_shapes(
                shape_group,
                shape_names,
                lw,
                dx,
                units,
                hue=None,
                cmap=None,
                facecolor=facecolor,
                edgecolor=edgecolor,
                textcolor=textcolor,
                legend=False,
                ax=axes[0],
                **kwargs,
            )

        # Formatting subplots
        for ax, group_name in zip(axes, group_names):
            # subplot titles
            plt.text(
                0.02,
                0.98,
                group_name,
                ha="left",
                va="top",
                color=textcolor,
                transform=ax.transAxes,
            )
            # axis splines
            ax.spines[["top", "right", "bottom", "left"]].set_visible(frameon)
            ax.axis(frameon)

        # box_aspect for Axes, aspect for data
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
            box_aspect=1,
            aspect=1,
        )
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.setp(fig.patch, facecolor=facecolor)


def _plot_shapes(
    data,
    shape_names,
    lw,
    dx,
    units,
    hue,
    cmap,
    facecolor,
    edgecolor,
    textcolor,
    legend,
    ax,
    ax_radius=None,
    **kwargs,
):
    if legend and hue and hue != "embed_color":
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0)
        legend = True
    else:
        cax = None
        legend = False

    # Gather all shapes and plot
    for shape_name in shape_names:
        data.reset_index().set_geometry(shape_name).plot(
            column=hue,
            cmap=cmap,
            facecolor=(0, 0, 0, 0),
            edgecolor=edgecolor,
            lw=lw,
            aspect=None,
            ax=ax,
            legend=legend,
            cax=cax,
            legend_kwds={"orientation": "horizontal"},
            **kwargs,
        )

    # Set axes boundaries to be square; make sure size of cells are relative to one another
    if ax_radius:
        s_bound = data.bounds
        centerx = np.mean([s_bound["minx"].min(), s_bound["maxx"].max()])
        centery = np.mean([s_bound["miny"].min(), s_bound["maxy"].max()])
        ax.set_xlim(centerx - ax_radius, centerx + ax_radius)
        ax.set_ylim(centery - ax_radius, centery + ax_radius)

        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        rect_bound = gpd.GeoDataFrame(
            geometry=[Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])]
        )

        # Mask outside cells
        rect_bound.overlay(data, how="difference").plot(
            ax=ax, facecolor=facecolor, edgecolor=edgecolor, lw=lw
        )

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


def _flow(data, x, y, resolution, method, ax, **kwargs):

    components = data[["c1", "c2", "c3"]].values

    # Get subplot xy grid bounds
    minx, maxx = data[x].min(), data[x].max()
    miny, maxy = data[y].min(), data[y].max()

    if resolution == "high":
        pixels_per_step = 1
    elif resolution == "low":
        pixels_per_step = 10

    # Define grid coordinates
    grid_x, grid_y = np.mgrid[
        minx : maxx : ((maxx - minx) / pixels_per_step) + 0j,
        miny : maxy : ((maxy - miny) / pixels_per_step) + 0j,
    ]
    values = []

    # Interpolate values for each channel
    for cp in range(components.shape[1]):
        values.append(
            griddata(
                data[[x, y]].values,
                components[:, cp],
                (grid_x, grid_y),
                method=method,
                fill_value=0,
            ).T
        )

    values = np.stack(values, axis=-1)
    # interpolating can cause values to be outside of [0, 1] range
    values = np.clip(values, 0, 1)

    ax.imshow(values, extent=(minx, maxx, miny, maxy), origin="lower")
    ax.autoscale(False)


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
