import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mplp
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
from shapely.geometry import Polygon, MultiPolygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import PatchCollection

from .._utils import get_points, get_shape, get_points_metadata
from ._utils import polytopatch


def _scatter(points, ax, hue=None, size=None, style=None, **kwargs):
    semantic_vars = list(set([hue, size, style]))
    semantic_vars = (
        None if semantic_vars == [] else [v for v in semantic_vars if v is not None]
    )

    if ax is None:
        ax = plt.gca()

    scatter_kws = dict(s=2, linewidth=0, zorder=1, rasterized=True)
    scatter_kws.update(kwargs)

    # Let matplotlib scatter handle color if it's in hex format
    if hue and str(points[hue].iloc[0]).startswith("#"):
        scatter_kws["c"] = points[hue]
        hue = None

    sns.scatterplot(
        data=points, x="x", y="y", hue=hue, size=size, style=style, ax=ax, **scatter_kws
    )


def _hist(points, ax, hue=None, **kwargs):
    hist_kws = dict(zorder=1)
    hist_kws.update(kwargs)

    sns.histplot(data=points, x="x", y="y", hue=hue, ax=ax, **hist_kws)


def _kde(points, ax, hue=None, **kwargs):
    kde_kws = dict(zorder=1, fill=True)
    kde_kws.update(kwargs)

    sns.kdeplot(data=points, x="x", y="y", hue=hue, ax=ax, **kde_kws)


def _polygons(sdata, shape, ax, hue=None, sync=True, **kwargs):
    """Plot shapes with GeoSeries plot function."""
    shapes = gpd.GeoDataFrame(geometry=get_shape(sdata, shape, sync=sync))

    style_kwds = dict(lw=0.5)
    # If hue is specified, use it to color faces
    if hue:
        df = (
            shapes.reset_index()
            .merge(
                sdata.shapes[shape], how="left", left_on="geometry", right_on="geometry"
            )
            .set_index("index")
        )
        if hue == "cell":
            shapes[hue] = df.index
        else:
            shapes[hue] = df.reset_index()[hue].values
        style_kwds["facecolor"] = sns.axes_style()["axes.edgecolor"]
        style_kwds["edgecolor"] = (
            "none"  # let GeoDataFrame plot function handle facecolor
        )

    style_kwds.update(kwargs)

    patches = []
    # Manually create patches for each polygon; GeoPandas plot function is slow
    for poly in shapes["geometry"].values:
        if isinstance(poly, Polygon):
            patches.append(polytopatch(poly))
        elif isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                patches.append(polytopatch(p))

    # Add patches to axes
    patches = PatchCollection(patches, **style_kwds)
    ax.add_collection(patches)


def _raster(
    sdata, res, color, points_key, alpha, pthreshold=None, cbar=False, ax=None, **kwargs
):
    """Plot gradient."""

    if ax is None:
        ax = plt.gca()

    #
    points = get_points(sdata, points_key=points_key, astype="pandas", sync=True)
    step = 1 / res
    color_values = (
        get_points_metadata(sdata, metadata_keys=color, points_key=points_key)[color]
        .replace("", np.nan)
        .values
    )

    # Infer value format and convert values to rgb
    # Handle color names and (r, g, b) tuples with matplotlib
    v1 = color_values[0]
    if isinstance(v1, str) or (
        isinstance(v1, tuple) and v1.min() >= 0 and v1.max() <= 1
    ):
        if alpha:
            rgb = np.array([mpl.colors.to_rgba(c) for c in color_values])
        else:
            rgb = np.array([mpl.colors.to_rgb(c) for c in color_values])
    else:
        rgb = color_values.reshape(-1, 1)

    # Get subplot xy grid bounds
    minx, maxx = points["x"].min(), points["x"].max()
    miny, maxy = points["y"].min(), points["y"].max()

    # Define grid coordinates
    grid_x, grid_y = np.mgrid[
        minx : maxx + step : step,
        miny : maxy + step : step,
    ]

    values = []
    for channel in range(rgb.shape[1]):
        values.append(
            griddata(
                points[["x", "y"]].values,
                rgb[:, channel],
                (grid_x, grid_y),
                method="nearest",
                fill_value=0,
            ).T
        )
    img = np.stack(values, axis=-1)

    img_kws = dict(interpolation="none", zorder=1)
    img_kws.update(kwargs)

    im = ax.imshow(img, extent=(minx, maxx, miny, maxy), origin="lower", **img_kws)
    ax.autoscale(False)

    if cbar:
        cax = inset_axes(ax, width="20%", height="4%", loc="upper right", borderpad=1.5)
        cbar = plt.colorbar(im, orientation="horizontal", cax=cax)
        # cbar.ax.tick_params(axis="x", direction="in", pad=-12)
