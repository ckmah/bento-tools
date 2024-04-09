import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar

from ..geometry import get_points
from ._layers import _raster, _scatter, _hist, _kde, _polygons
from ._utils import savefig
from ._colors import red2blue, red2blue_dark

def _prepare_points_df(sdata, semantic_vars=None, hue=None, hue_order=None):
    """
    Prepare points DataFrame for plotting. This function will concatenate the appropriate semantic variables as columns to points data.
    """
    points = get_points(sdata, astype="pandas")
    cols = list(set(["x", "y", "cell"]))

    if semantic_vars is None or len(semantic_vars) == 0:
        return points[cols]
    else:
        vars = [v for v in semantic_vars if v is not None]
    cols.extend(vars)

    if hue_order is not None:
        points = points[points[hue].isin(hue_order)]

    # Add semantic variables to points; priority: points, obs, points metadata
    for var in vars:
        if var in points.columns:
            continue
        elif var in sdata.shapes:
            points[var] = sdata.shapes[var].reindex(points["cell"].values)[var].values
        else:
            raise ValueError(f"Variable {var} not found in points or obs")
    
    return points[cols]

def _setup_ax(
    ax=None,
    dx=0.1,
    units="um",
    square=False,
    axis_visible=False,
    frame_visible=True,
    **kwargs,
):
    """Setup axis for plotting. TODO make decorator?"""
    if ax is None:
        ax = plt.gca()

    # Infer font color from theme
    edgecolor = sns.axes_style()["axes.edgecolor"]

    scalebar = ScaleBar(
        dx,
        units,
        location="lower right",
        box_alpha=0,
        color=edgecolor,
        frameon=False,
        scale_loc="top",
    )
    ax.add_artist(scalebar)

    ax_kws = dict(aspect=1, box_aspect=None)

    if not axis_visible:
        ax_kws.update(
            dict(
                xticks=[],
                yticks=[],
                xticklabels=[],
                yticklabels=[],
                ylabel=None,
                xlabel=None,
                xmargin=0.01,
                ymargin=0.01,
            )
        )

    if square:
        ax_kws["box_aspect"] = 1

    ax_kws.update(kwargs)
    plt.setp(ax, **ax_kws)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(frame_visible)

    return ax

@savefig
def points(
    sdata,
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
    sync_shapes=True,
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
    
    ax = _setup_ax(
        ax=ax,
        dx=dx,
        units=units,
        square=square,
        axis_visible=axis_visible,
        frame_visible=frame_visible,
        title=title,
    )
    points = _prepare_points_df(sdata, semantic_vars=[hue, size, style], hue=hue, hue_order=hue_order)
    _scatter(points, hue=hue, size=size, style=style, ax=ax, **kwargs)
    _shapes(sdata, shapes=shapes, hide_outside=hide_outside, ax=ax, sync_shapes=sync_shapes, **shapes_kws)

@savefig
def density(
    sdata,
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
    sync_shapes=True,
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

    ax = _setup_ax(
        ax=ax,
        dx=dx,
        units=units,
        square=square,
        axis_visible=axis_visible,
        frame_visible=frame_visible,
        title=title,
    )

    points = _prepare_points_df(sdata, semantic_vars=[hue], hue=hue, hue_order=hue_order)
    if kind == "hist":
        _hist(points, hue=hue, ax=ax, **kwargs)
    elif kind == "kde":
        _kde(points, hue=hue, ax=ax, **kwargs)

    _shapes(sdata, shapes=shapes, hide_outside=hide_outside, ax=ax, sync_shapes=sync_shapes, **shape_kws)

@savefig
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
    sync_shapes=True,
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

    ax = _setup_ax(
        ax=ax,
        dx=dx,
        units=units,
        square=square,
        axis_visible=axis_visible,
        frame_visible=frame_visible,
        title=title,
    )

    if shapes and not isinstance(shapes, list):
        shapes = [shapes]

    _shapes(
        sdata,
        shapes=shapes,
        color=color,
        color_style=color_style,
        hide_outside=hide_outside,
        ax=ax,
        sync_shapes=sync_shapes,
        **kwargs,
    )


def _shapes(
    sdata,
    shapes=None,
    color=None,
    color_style="outline",
    hide_outside=True,
    ax=None,
    sync_shapes=True,
    **kwargs,
):
    if shapes is None:
        shapes = ["cell", "nucleus"]

    shape_names = []
    for s in shapes:
        if str(s).endswith("_boundaries"):
            shape_names.append(s)
        else:
            shape_names.append(f"{s}_boundaries")

    # Save list of names to remove if not in data.obs
    shape_names = [name for name in shape_names if name in sdata.shapes.keys()]
    missing_names = [name for name in shape_names if name not in sdata.shapes.keys()]

    if len(missing_names) > 0:
        warnings.warn("Shapes not found in data: " + ", ".join(missing_names))

    geo_kws = dict(edgecolor="none", facecolor="none")
    if color_style == "outline":
        geo_kws["edgecolor"] = color
        geo_kws["facecolor"] = "none"
    elif color_style == "fill":
        geo_kws["facecolor"] = color
        geo_kws["edgecolor"] = "black"
    geo_kws.update(**kwargs)

    for name in shape_names:
        hide = False
        if name == "cell_boundaries" and hide_outside:
            hide = True

        _polygons(
            sdata,
            name,
            hide_outside=hide,
            ax=ax,
            sync_shapes=sync_shapes,
            **geo_kws,
        )

@savefig
def flux(
    sdata,
    res=0.05,
    shapes=None,
    hide_outside=True,
    axis_visible=False,
    frame_visible=True,
    title=None,
    dx=0.1,
    units="um",
    square=False,
    ax=None,
    sync_shapes=True,
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
        Resolution of fluxmap, by default 0.05
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

    ax = _setup_ax(
        ax=ax,
        dx=dx,
        units=units,
        square=square,
        axis_visible=axis_visible,
        frame_visible=frame_visible,
        title=title,
    )

    _raster(sdata, res=res, color="flux_color", ax=ax, **kwargs)
    _shapes(sdata, shapes=shapes, hide_outside=hide_outside, ax=ax, sync_shapes=sync_shapes, **shape_kws)

@savefig
def fe(
    sdata,
    gs,
    res=0.05,
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
    sync_shapes=True,
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
        Resolution of fluxmap, by default 0.05
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

    ax = _setup_ax(
        ax=ax,
        dx=dx,
        units=units,
        square=square,
        axis_visible=axis_visible,
        frame_visible=frame_visible,
        title=title,
    )

    if cmap is None:
        if sns.axes_style()["axes.facecolor"] == "white":
            cmap = red2blue
        elif sns.axes_style()["axes.facecolor"] == "black":
            cmap = red2blue_dark

    _raster(sdata, res=res, color=gs, cmap=cmap, cbar=cbar, ax=ax, **kwargs)
    _shapes(sdata, shapes=shapes, hide_outside=hide_outside, ax=ax, sync_shapes=sync_shapes, **shape_kws)

@savefig
def fluxmap(
    sdata,
    palette="tab10",
    hide_outside=True,
    axis_visible=False,
    frame_visible=True,
    title=None,
    dx=0.1,
    ax=None,
    sync_shapes=False,
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

    # Plot fluxmap shapes
    if isinstance(palette, dict):
        colormap = palette
    else:
        fluxmap_shapes = [s for s in sdata.shapes.keys() if s.startswith("fluxmap")]
        fluxmap_shapes.sort()
        colors = sns.color_palette(palette, n_colors=len(fluxmap_shapes))
        colormap = dict(zip(fluxmap_shapes, colors))

    shape_kws = dict(color_style="fill")
    shape_kws.update(kwargs)

    for s, c in colormap.items():
        shapes(
            sdata,
            shapes=s,
            color=c,
            hide_outside=hide_outside,
            axis_visible=axis_visible,
            frame_visible=frame_visible,
            title=title,
            dx=dx,
            ax=ax,
            sync_shapes=sync_shapes,
            **shape_kws,
        )
        
    # Plot base cell and nucleus shapes
    shapes(sdata, ax=ax, fname=fname)