import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar

from ..geometry import get_points
from ._layers import _raster, _scatter, _hist, _kde, _polygons
from ._utils import savefig
from .._utils import sync
from ._colors import red2blue, red2blue_dark


def _prepare_points_df(adata, semantic_vars=None):
    """
    Prepare points DataFrame for plotting. This function will concatenate the appropriate semantic variables as columns to points data.
    """
    points = get_points(adata, key="points")
    cols = list(set(["x", "y", "cell"]))

    if semantic_vars is None or len(semantic_vars) == 0:
        return points[cols]
    else:
        vars = [v for v in semantic_vars if v is not None]
    cols.extend(vars)

    # Add semantic variables to points; priority: points, obs, points metadata
    for var in vars:
        if var in points.columns:
            continue
        elif var in adata.obs.columns:
            points[var] = adata.obs.reindex(points["cell"].values)[var].values
        elif var in adata.uns["point_sets"]["points"]:
            if len(adata.uns[var].shape) > 1:
                raise ValueError(f"Variable {var} is not 1-dimensional")
            points[var] = adata.uns[var]
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
    data,
    batch=None,
    hue=None,
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
    # Default use first obs batch
    if batch is None:
        batch = data.obs["batch"].iloc[0]
    adata = data[data.obs["batch"] == batch]
    title = f"batch {batch}" if not title else title

    ax = _setup_ax(
        ax=ax,
        dx=dx,
        units=units,
        square=square,
        axis_visible=axis_visible,
        frame_visible=frame_visible,
        title=title,
    )

    points = _prepare_points_df(adata, semantic_vars=[hue, size, style])
    _scatter(points, hue=hue, size=size, style=style, ax=ax, **kwargs)
    _shapes(adata, shapes=shapes, hide_outside=hide_outside, ax=ax, **shapes_kws)


@savefig
def density(
    data,
    batch=None,
    kind="hist",
    hue=None,
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
    # Default use first obs batch
    if batch is None:
        batch = data.obs["batch"].iloc[0]
    adata = data[data.obs["batch"] == batch]
    title = f"batch {batch}" if title is None else title

    ax = _setup_ax(
        ax=ax,
        dx=dx,
        units=units,
        square=square,
        axis_visible=axis_visible,
        frame_visible=frame_visible,
        title=title,
    )

    points = _prepare_points_df(adata, semantic_vars=[hue])
    if kind == "hist":
        _hist(points, hue=hue, ax=ax, **kwargs)
    elif kind == "kde":
        _kde(points, hue=hue, ax=ax, **kwargs)

    _shapes(adata, shapes=shapes, hide_outside=hide_outside, ax=ax, **shape_kws)


@savefig
def flux(
    data,
    batch=None,
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
    shape_kws=dict(),
    fname=None,
    **kwargs,
):
    # Default use first obs batch
    if batch is None:
        batch = data.obs["batch"].iloc[0]
    adata = data[data.obs["batch"] == batch]
    title = f"batch {batch}" if not title else title

    ax = _setup_ax(
        ax=ax,
        dx=dx,
        units=units,
        square=square,
        axis_visible=axis_visible,
        frame_visible=frame_visible,
        title=title,
    )

    _raster(adata, res=res, color="flux_color", ax=ax, **kwargs)
    _shapes(adata, shapes=shapes, hide_outside=hide_outside, ax=ax, **shape_kws)


@savefig
def fe(
    data,
    gs,
    batch=None,
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
    shape_kws=dict(),
    fname=None,
    **kwargs,
):
    # Default use first obs batch
    if batch is None:
        batch = data.obs["batch"].iloc[0]
    adata = data[data.obs["batch"] == batch]
    sync(adata)
    title = f"batch {batch}" if not title else title

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

    _raster(adata, res=res, color=gs, cmap=cmap, cbar=cbar, ax=ax, **kwargs)
    _shapes(adata, shapes=shapes, hide_outside=hide_outside, ax=ax, **shape_kws)


@savefig
def shapes(
    data,
    batch=None,
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
    # Default use first obs batch
    if batch is None:
        batch = data.obs["batch"].iloc[0]
    adata = data[data.obs["batch"] == batch]
    title = f"batch {batch}" if not title else title

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
        adata,
        shapes=shapes,
        color=color,
        color_style=color_style,
        hide_outside=hide_outside,
        ax=ax,
        **kwargs,
    )


def _shapes(
    data,
    shapes=None,
    color=None,
    color_style="outline",
    hide_outside=True,
    ax=None,
    **kwargs,
):
    """Plot layer(s) of shapes.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    shapes : list, optional
        List of shapes to plot, by default None. If None, will plot cell and nucleus shapes by default.
    color : str, optional
        Color name, by default None. If None, will use default theme color.
    color_style : "outline" or "fill"
        Whether to color the outline or fill of the shape, by default "outline".
    hide_outside : bool, optional
        Whether to hide molecules outside of cells, by default True.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None. If None, will use current axis.
    """
    if shapes is None:
        shapes = ["cell", "nucleus"]

    shape_names = []
    for s in shapes:
        if str(s).endswith("_shape"):
            shape_names.append(s)
        else:
            shape_names.append(f"{s}_shape")

    # Save list of names to remove if not in data.obs
    shape_names = [name for name in shape_names if name in data.obs.columns]
    missing_names = [name for name in shape_names if name not in data.obs.columns]

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
        if name == "cell_shape" and hide_outside:
            hide = True

        _polygons(
            data,
            name,
            hide_outside=hide,
            ax=ax,
            **geo_kws,
        )


def fluxmap(
    data,
    batch=None,
    palette="tab10",
    hide_outside=True,
    ax=None,
    fname=None,
    **kwargs,
):
    """Plot fluxmap shapes in different colors. Wrapper for :func:`bt.pl.shapes()`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    batch : str, optional
        Batch to plot, by default None. If None, will use first batch.
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
        fluxmap_shapes = [s for s in data.obs.columns if s.startswith("fluxmap")]
        fluxmap_shapes.sort()
        colors = sns.color_palette(palette, n_colors=len(fluxmap_shapes))
        colormap = dict(zip(fluxmap_shapes, colors))

    shape_kws = dict(color_style="fill")
    shape_kws.update(kwargs)

    for s, c in colormap.items():
        shapes(
            data,
            batch=batch,
            shapes=s,
            color=c,
            hide_outside=hide_outside,
            ax=ax,
            **shape_kws,
        )

    # Plot base cell and nucleus shapes
    shapes(data, batch=batch, ax=ax, fname=fname)
