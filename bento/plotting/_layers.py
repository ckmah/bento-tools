import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
from shapely.geometry import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..geometry import get_points, get_shape, get_points_metadata


def _scatter(points, ax, hue=None, size=None, style=None, **kwargs):
    semantic_vars = list(set([hue, size, style]))
    semantic_vars = (
        None if semantic_vars == [] else [v for v in semantic_vars if v is not None]
    )

    if ax is None:
        ax = plt.gca()

    scatter_kws = dict(s=2, c="grey", linewidth=0)
    scatter_kws.update(kwargs)

    # Let matplotlib scatter handle color if it's in hex format
    if hue and str(points[hue].iloc[0]).startswith("#"):
        scatter_kws["c"] = points[hue]
        hue = None

    sns.scatterplot(
        data=points, x="x", y="y", hue=hue, size=size, style=style, ax=ax, **scatter_kws
    )


def _hist(points, ax, hue=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    hist_kws = dict()
    hist_kws.update(kwargs)

    sns.histplot(data=points, x="x", y="y", hue=hue, ax=ax, **hist_kws)


def _kde(points, ax, hue=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    kde_kws = dict(zorder=1, fill=True)
    kde_kws.update(kwargs)

    sns.kdeplot(data=points, x="x", y="y", hue=hue, ax=ax, **kde_kws)


def _polygons(adata, shape, ax, hue=None, hide_outside=False, **kwargs):
    """Plot shapes with GeoSeries plot function."""

    shapes = gpd.GeoDataFrame(geometry=get_shape(adata, shape))

    edge_color = "none"
    face_color = "none"

    # If hue is specified, use it to color faces
    if hue:
        shapes[hue] = adata.obs.reset_index()[hue].values
        edge_color = sns.axes_style()["axes.edgecolor"]
        face_color = "none"  # let GeoDataFrame plot function handle facecolor

    style_kwds = dict(
        linewidth=0.5, edgecolor=edge_color, facecolor=face_color, zorder=2
    )
    style_kwds.update(kwargs)
    shapes.plot(ax=ax, column=hue, **style_kwds)

    if hide_outside:
        # get axes limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # get min range
        # min_range = min(xmax - xmin, ymax - ymin)
        # buffer_size = 0.0 * (min_range)

        # Create shapely polygon from axes limits
        axes_poly = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(
                [Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])]
            )
            # .buffer(0)
        )
        axes_poly.overlay(shapes, how="difference").plot(
            ax=ax,
            linewidth=0,
            facecolor=sns.axes_style()["axes.facecolor"],
            zorder=1.99,
        )


def _raster(adata, res, color, points_key="cell_raster", cbar=False, ax=None, **kwargs):
    """Plot gradient."""

    if ax is None:
        ax = plt.gca()

    points = get_points(adata, key=points_key)
    step = 1 / res
    color_values = get_points_metadata(adata, metadata_key=color, points_key=points_key)
    # Infer value format and convert values to rgb
    # Handle color names and (r, g, b) tuples with matplotlib
    v1 = color_values[0]
    if isinstance(v1, str) or (
        isinstance(v1, tuple) and v1.min() >= 0 and v1.max() <= 1
    ):
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

    img_kws = dict(interpolation="none")
    img_kws.update(kwargs)

    im = ax.imshow(img, extent=(minx, maxx, miny, maxy), origin="lower", **img_kws)
    ax.autoscale(False)

    if cbar:
        cax = inset_axes(ax, width="20%", height="4%", loc="upper right", borderpad=1.5)
        cbar = plt.colorbar(im, orientation="horizontal", cax=cax)
        # cbar.ax.tick_params(axis="x", direction="in", pad=-12)
