import warnings

from spatialdata import SpatialData

warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib.patches as mplp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection, PathCollection, PolyCollection
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path
from shapely.geometry import Polygon

from .._utils import get_points, get_shape
from ._colors import red2blue, red2blue_dark
from ._layers import _hist, _kde, _polygons, _raster, _scatter
from ._utils import savefig, setup_ax, polytopatch


def _prepare_points_df(
    sdata, points_key, instance_key, sync, semantic_vars=None, hue=None, hue_order=None
):
    """
    Prepare points DataFrame for plotting. This function will concatenate the appropriate semantic variables as columns to points data.
    """
    points = get_points(sdata, points_key=points_key, astype="pandas", sync=sync)
    cols = list(set(["x", "y", instance_key]))

    if semantic_vars is None or len(semantic_vars) == 0:
        return points[cols]
    else:
        vars = [v for v in semantic_vars if v is not None]
    cols.extend(vars)

    cols = list(set(cols))

    if hue_order is not None:
        points = points[points[hue].isin(hue_order)]
        points[hue] = pd.Categorical(points[hue], categories=hue_order)

    # Add semantic variables to points; priority: points, obs, points metadata
    for var in vars:
        if var in points.columns:
            continue
        elif var in sdata.shapes:
            points[var] = (
                sdata.shapes[var].reindex(points[instance_key].values)[var].values
            )
        else:
            raise ValueError(f"Variable {var} not found in points or obs")

    return points[cols]


@savefig
@setup_ax
def points(
    sdata: SpatialData,
    points_key="transcripts",
    instance_key="cell_boundaries",
    hue=None,
    hue_order=None,
    size=None,
    style=None,
    shapes=None,
    hide_outside=True,
    title=None,
    dx=0.1,
    units="um",
    square=False,
    axis_visible=False,
    frame_visible=True,
    ax=None,
    shapes_kws=dict(),
    fname=None,
    **kwargs,
):
    """
    Plot points scatter.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object
    hue : str, optional
        Variable name to color points by, by default None
    hue_order : list, optional
        Order of hue levels, by default None
    size : str, optional
        Variable name to size points by, by default None
    style : str, optional
        Variable name to style points by, by default None
    shapes : list, optional
        List of shape names to plot, by default None. If None, will plot cell and nucleus shapes by default.
    hide_outside : bool, optional
        Whether to hide molecules outside of cells, by default True
    title : str, optional
        Title of plot, by default None
    dx : float, optional
        Size of scalebar in units, by default 0.1
    units : str, optional
        Units of scalebar, by default "um"
    square : bool, optional
        Whether to make axis square, by default False
    axis_visible : bool, optional
        Whether to show axis, by default False
    frame_visible : bool, optional
        Whether to show frame, by default True
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None. If None, will use current axis.
    sync_shapes : bool, optional
        Whether to synchronize shapes with points, by default True
    shapes_kws : dict, optional
        Keyword arguments for shapes, by default {}
    fname : str, optional
        Filename to save figure to, by default None. If None, will not save figure.
    """

    points = _prepare_points_df(
        sdata,
        points_key=points_key,
        instance_key=instance_key,
        sync=hide_outside,
        semantic_vars=[hue, size, style],
        hue=hue,
        hue_order=hue_order,
    )

    if ax is None:
        ax = plt.gca()

    _scatter(points, hue=hue, size=size, style=style, ax=ax, **kwargs)
    _shapes(sdata, shapes=shapes, hide_outside=hide_outside, ax=ax, **shapes_kws)

    return ax


@savefig
@setup_ax
def density(
    sdata,
    points_key="transcripts",
    instance_key="cell_boundaries",
    kind="hist",
    hue=None,
    hue_order=None,
    shapes=None,
    hide_outside=True,
    axis_visible=False,
    frame_visible=True,
    title=None,
    dx=0.1,
    units="um",
    square=False,
    ax=None,
    shape_kws=dict(),
    fname=None,
    **kwargs,
):
    """
    Plot points as 2D density.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object
    kind : str, optional
        Type of density plot, by default "hist". Options: "hist", "kde"
    hue : str, optional
        Variable name to color points by, by default None
    hue_order : list, optional
        Order of hue levels, by default None
    shapes : list, optional
        List of shape names to plot, by default None. If None, will plot cell and nucleus shapes by default.
    hide_outside : bool, optional
        Whether to hide molecules outside of cells, by default True
    title : str, optional
        Title of plot, by default None
    dx : float, optional
        Size of scalebar in units, by default 0.1
    units : str, optional
        Units of scalebar, by default "um"
    square : bool, optional
        Whether to make axis square, by default False
    axis_visible : bool, optional
        Whether to show axis, by default False
    frame_visible : bool, optional
        Whether to show frame, by default True
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None. If None, will use current axis.
    sync_shapes : bool, optional
        Whether to synchronize shapes with points, by default True
    shape_kws : dict, optional
        Keyword arguments for shapes, by default {}
    fname : str, optional
        Filename to save figure to, by default None. If None, will not save figure.
    """

    points = _prepare_points_df(
        sdata,
        points_key=points_key,
        instance_key=instance_key,
        sync=hide_outside,
        semantic_vars=[hue],
        hue=hue,
        hue_order=hue_order,
    )

    if ax is None:
        ax = plt.gca()

    if kind == "hist":
        _hist(points, hue=hue, ax=ax, **kwargs)
    elif kind == "kde":
        _kde(points, hue=hue, ax=ax, **kwargs)

    _shapes(
        sdata,
        shapes=shapes,
        instance_key=instance_key,
        hide_outside=hide_outside,
        ax=ax,
        **shape_kws,
    )


@savefig
@setup_ax
def shapes(
    sdata,
    shapes=None,
    color=None,
    color_style="outline",
    hide_outside=True,
    dx=0.1,
    units="um",
    axis_visible=False,
    frame_visible=True,
    title=None,
    square=False,
    ax=None,
    fname=None,
    **kwargs,
):
    """Plot shape layers.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData
    shapes : list, optional
        List of shapes to plot, by default None. If None, will plot cell and nucleus shapes by default.
    color : str, optional
        Color name, by default None. If None, will use default theme color.
    color_style : "outline" or "fill"
        Whether to color the outline or fill of the shape, by default "outline".
    hide_outside : bool, optional
        Whether to hide molecules outside of cells, by default True.
    dx : float, optional
        Size of scalebar in units, by default 0.1.
    units : str, optional
        Units of scalebar, by default "um".
    axis_visible : bool, optional
        Whether to show axis, by default False.
    frame_visible : bool, optional
        Whether to show frame, by default True.
    title : str, optional
        Title of plot, by default None.
    square : bool, optional
        Whether to make axis square, by default False.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None. If None, will use current axis.
    sync_shapes : bool, optional
        Whether to synchronize shapes with points, by default True.
    fname : str, optional
        Filename to save figure to, by default None. If None, will not save figure.
    """

    if ax is None:
        ax = plt.gca()

    _shapes(
        sdata,
        shapes=shapes,
        color=color,
        color_style=color_style,
        hide_outside=hide_outside,
        ax=ax,
        **kwargs,
    )


def _shapes(
    sdata,
    instance_key="cell_boundaries",
    nucleus_key="nucleus_boundaries",
    shapes=None,
    color=None,
    color_style="outline",
    hide_outside=True,
    ax=None,
    **kwargs,
):
    if shapes is None:
        shapes = [instance_key, nucleus_key]

    if shapes and not isinstance(shapes, list):
        shapes = [shapes]

    shape_names = [name for name in shapes if name in sdata.shapes.keys()]
    missing_names = [name for name in shapes if name not in sdata.shapes.keys()]

    if len(missing_names) > 0:
        warnings.warn("Shapes not found in data: " + ", ".join(missing_names))

    geo_kws = dict(lw=0.5)
    if color_style == "outline":
        geo_kws["edgecolor"] = color if color else sns.axes_style()["axes.edgecolor"]
        geo_kws["facecolor"] = (0, 0, 0, 0)
    elif color_style == "fill":
        geo_kws["facecolor"] = color
        geo_kws["edgecolor"] = sns.axes_style()["axes.edgecolor"]
    geo_kws.update(**kwargs)

    for name in shape_names:
        if name == instance_key:
            geo_kws.update(zorder=2.001)
        else:
            geo_kws.update(zorder=2)
        _polygons(
            sdata,
            name,
            sync=hide_outside,
            ax=ax,
            **geo_kws,
        )

    # Hide extracellular area
    if hide_outside and instance_key in shape_names:
        instance_poly = gpd.GeoDataFrame(get_shape(sdata, shape_key=instance_key))
        bounds = instance_poly.total_bounds

        axes_poly = Polygon(
            (
                [bounds[0], bounds[1]],
                [bounds[0], bounds[3]],
                [bounds[2], bounds[3]],
                [bounds[2], bounds[1]],
            )
        )
        buffer = 0.01 * (bounds[2] - bounds[0])
        axes_poly = gpd.GeoDataFrame(
            geometry=[axes_poly.buffer(buffer, cap_style="square")]
        )

        # Take difference between instance_key shapes and buffered bounding box
        mask_poly = axes_poly.overlay(instance_poly, how="difference")

        for poly in mask_poly.geometry.values:
            if isinstance(poly, Polygon):
                mask_polys = [polytopatch(poly)]
            else:
                mask_polys = [polytopatch(p) for p in poly.geoms]
            patches = PatchCollection(
                mask_polys,
                facecolor=sns.axes_style()["axes.facecolor"],
                zorder=2.0001,
            )
            ax.add_collection(patches)


@savefig
@setup_ax
def flux(
    sdata,
    instance_key="cell_boundaries",
    alpha=True,
    res=1,
    shapes=None,
    hide_outside=True,
    axis_visible=False,
    frame_visible=True,
    title=None,
    dx=0.1,
    units="um",
    square=False,
    ax=None,
    shape_kws=dict(),
    fname=None,
    **kwargs,
):
    """Plot colorized representation of RNAflux embedding.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData
    res : float, optional
        Resolution of flux, by default 1
    shapes : list, optional
        List of shapes to plot, by default None. If None, will plot cell and nucleus shapes by default.
    hide_outside : bool, optional
        Whether to hide molecules outside of cells, by default True.
    axis_visible : bool, optional
        Whether to show axis, by default False.
    frame_visible : bool, optional
        Whether to show frame, by default True.
    title : str, optional
        Title of plot, by default None.
    dx : float, optional
        Size of scalebar in units, by default 0.1.
    units : str, optional
        Units of scalebar, by default "um".
    square : bool, optional
        Whether to make axis square, by default False.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None. If None, will use current axis.
    sync_shapes : bool, optional
        Whether to synchronize shapes with points, by default True.
    shape_kws : dict, optional
        Keyword arguments for shapes, by default {}.
    fname : str, optional
        Filename to save figure to, by default None. If None, will not save figure.
    """

    if ax is None:
        ax = plt.gca()

    _raster(
        sdata,
        alpha=alpha,
        points_key=f"{instance_key}_raster",
        res=res,
        color="flux_color",
        ax=ax,
        **kwargs,
    )
    _shapes(
        sdata,
        shapes=shapes,
        instance_key=instance_key,
        hide_outside=hide_outside,
        ax=ax,
        **shape_kws,
    )


@savefig
@setup_ax
def fe(
    sdata,
    gs,
    instance_key="cell_boundaries",
    alpha=False,
    res=1,
    shapes=None,
    cmap=None,
    cbar=True,
    hide_outside=True,
    axis_visible=False,
    frame_visible=True,
    title=None,
    dx=0.1,
    units="um",
    square=False,
    ax=None,
    shape_kws=dict(),
    fname=None,
    **kwargs,
):
    """Plot spatial heatmap of flux enrichment scores.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData
    gs : str
        Gene set name
    res : float, optional
        Resolution of flux, by default 1
    shapes : list, optional
        List of shape names to plot, by default None. If None, will plot cell and nucleus shapes by default.
    cmap : str, optional
        Colormap, by default None. If None, will use red2blue colormap.
    cbar : bool, optional
        Whether to show colorbar, by default True
    hide_outside : bool, optional
        Whether to hide molecules outside of cells, by default True.
    axis_visible : bool, optional
        Whether to show axis, by default False.
    frame_visible : bool, optional
        Whether to show frame, by default True.
    title : str, optional
        Title of plot, by default None.
    dx : float, optional
        Size of scalebar in units, by default 0.1.
    units : str, optional
        Units of scalebar, by default "um".
    square : bool, optional
        Whether to make axis square, by default False.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None. If None, will use current axis.
    sync_shapes : bool, optional
        Whether to synchronize shapes with points, by default True.
    shape_kws : dict, optional
        Keyword arguments for shapes, by default {}.
    fname : str, optional
        Filename to save figure to, by default None. If None, will not save figure.
    """

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        if sns.axes_style()["axes.facecolor"] == "white":
            cmap = red2blue
        elif sns.axes_style()["axes.facecolor"] == "black":
            cmap = red2blue_dark

    _raster(
        sdata,
        alpha=alpha,
        points_key=f"{instance_key}_raster",
        res=res,
        color=gs,
        cmap=cmap,
        cbar=cbar,
        ax=ax,
        **kwargs,
    )
    _shapes(
        sdata,
        shapes=shapes,
        instance_key=instance_key,
        hide_outside=hide_outside,
        ax=ax,
        **shape_kws,
    )
    # _shapes(sdata, instance_key=instance_key, hide_outside=hide_outside, ax=ax, **shape_kws)


@savefig
@setup_ax
def fluxmap(
    sdata,
    palette="tab10",
    instance_key="cell_boundaries",
    hide_outside=True,
    axis_visible=False,
    frame_visible=True,
    title=None,
    dx=0.1,
    units="um",
    square=False,
    ax=None,
    fname=None,
    **kwargs,
):
    """Plot fluxmap shapes in different colors. Wrapper for :func:`bt.pl.shapes()`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    palette : str or dict, optional
        Color palette, by default "tab10". If dict, will use dict to map shape names to colors.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None. If None, will use current axis.
    fname : str, optional
        Filename to save figure to, by default None. If None, will not save figure.
    """

    if not np.any([s.startswith("fluxmap") for s in sdata.shapes.keys()]):
        print("Please run bento.tl.fluxmap() first.")
        return

    if ax is None:
        ax = plt.gca()

    # Map shape names to colors
    if isinstance(palette, dict):
        colormap = palette
    else:
        fluxmap_shapes = [s for s in sdata.shapes.keys() if s.startswith("fluxmap")]
        fluxmap_shapes.sort()
        colors = sns.color_palette(palette, n_colors=len(fluxmap_shapes))
        colormap = dict(zip(fluxmap_shapes, colors))

    shape_kws = dict(color_style="fill", linewidth=0)
    shape_kws.update(kwargs)

    # Cycle through colormap and plot shapes
    for shape, color in colormap.items():
        _shapes(
            sdata,
            shapes=shape,
            instance_key=instance_key,
            color=color,
            hide_outside=False,
            ax=ax,
            **shape_kws,
        )

    # Create legend denoting each fluxmap as different color
    patches = [Patch(color=c, label=label) for label, c in colormap.items()]
    # for patch in patches:
    #     patch.set_marker('s')
    #     patch.set_markersize(20)

    ax.legend(
        patches,
        colormap.keys(),
        loc="upper left",
        bbox_to_anchor=[1, 1],
        frameon=False,
    )

    # Plot base cell and nucleus shapes
    _shapes(sdata, instance_key=instance_key, ax=ax)
