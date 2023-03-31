from typing import Callable, Dict, List, Union

import matplotlib.path as mplPath
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial import distance, distance_matrix
from shapely.geometry import MultiPolygon, Point
from tqdm.auto import tqdm

from .._utils import sync, track
from ..geometry import get_points, get_shape


def _area(data: AnnData, shape_name: str):
    """Compute the area of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    shape_name : str
        Key in `data.obs` that contains the shape information.

    Fields
    ------
        .obs['{shape}_area'] : float
            Area of each polygon
    """

    # Calculate pixel-wise area
    area = get_shape(data, shape_name).area

    shape_prefix = shape_name.split("_")[0]
    data.obs[f"{shape_prefix}_area"] = area


def _poly_aspect_ratio(poly):
    """Compute the aspect ratio of the minimum rotated rectangle that contains a polygon."""
    # get coordinates of min bounding box vertices around polygon
    x, y = poly.minimum_rotated_rectangle.exterior.coords.xy

    # get length of bound box sides
    edge_length = (
        Point(x[0], y[0]).distance(Point(x[1], y[1])),
        Point(x[1], y[1]).distance(Point(x[2], y[2])),
    )

    # length = longest side, width = shortest side
    length, width = max(edge_length), min(edge_length)

    # return long / short ratio
    return length / width


def _aspect_ratio(data: AnnData, shape_name: str):
    """Compute the aspect ratio of the minimum rotated rectangle that contains each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    shape_name : str
        Key in `data.obs` that contains the shape information.

    Fields
    ------
        .obs['{shape}_aspect_ratio'] : float
            Ratio of major to minor axis for each polygon
    """

    ar = get_shape(data, shape_name).apply(_poly_aspect_ratio)
    shape_prefix = shape_name.split("_")[0]
    data.obs[f"{shape_prefix}_aspect_ratio"] = ar


def _bounds(data, shape_name):
    """Compute the minimum and maximum coordinate values that bound each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    shape_name : str
        Key in `data.obs` that contains the shape information.

    Fields
    ------
        .obs['{shape}_minx'] : float
            x-axis lower bound of each polygon
        .obs['{shape}_miny'] : float
            y-axis lower bound of each polygon
        .obs['{shape}_maxx'] : float
            x-axis upper bound of each polygon
        .obs['{shape}_maxy'] : float
            y-axis upper bound of each polygon
    """

    bounds = get_shape(data, shape_name).bounds

    shape_prefix = shape_name.split("_")[0]
    data.obs[f"{shape_prefix}_minx"] = bounds["minx"]
    data.obs[f"{shape_prefix}_miny"] = bounds["miny"]
    data.obs[f"{shape_prefix}_maxx"] = bounds["maxx"]
    data.obs[f"{shape_prefix}_maxy"] = bounds["maxy"]


# TODO move to point_features
def _density(data, shape_name):
    """Compute the RNA density of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    shape_name : str
        Key in `data.obs` that contains the shape information.

    Fields
    ------
        .obs['{shape}_density'] : float
            Density (molecules / shape area) of each polygon
    """

    shape_prefix = shape_name.split("_")[0]
    count = get_points(data).query(f"{shape_prefix} != '-1'")["cell"].value_counts()
    _area(data, shape_name)

    data.obs[f"{shape_prefix}_density"] = count / data.obs[f"{shape_prefix}_area"]


def _opening(data, proportion):
    """Compute the opening (morphological) of distance d for each cell.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData

    Returns
    -------
    data : anndata.AnnData
        Returns `data` if `copy=True`, otherwise adds fields to `data`:

        `obs['cell_open_{d}_shape']` : Polygons
            Ratio of long / short axis for each polygon in `obs['cell_shape']`
    """

    shape_name = f"cell_open_{proportion}_shape"

    _radius(data, "cell_shape")

    cells = get_shape(data, "cell_shape")
    d = proportion * data.obs["cell_radius"]

    # Opening
    data.obs[shape_name] = cells.buffer(-d).buffer(d)


def _second_moment_polygon(centroid, pts):
    """
    Calculate second moment of points with centroid as reference.

    Parameters
    ----------
    centroid : [1 x 2] float
    pts : [n x 2] float
    """
    centroid = np.array(centroid).reshape(1, 2)
    radii = distance.cdist(centroid, pts)
    second_moment = np.sum(radii * radii / len(pts))
    return second_moment


def _second_moment(data, shape_name):
    """Compute the second moment of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData

    Returns
    -------
    data : anndata.AnnData
        Returns `data` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_moment']` : float
            The second moment for each polygon
    """

    shape_prefix = shape_name.split("_")[0]
    if f"{shape_prefix}_raster" not in data.obs:
        _raster(data, shape_name)

    rasters = data.obs[f"{shape_prefix}_raster"]
    shape_centroids = get_shape(data, shape_name).centroid

    moments = [
        _second_moment_polygon(np.array(centroid.xy).reshape(1, 2), r)
        for centroid, r in zip(shape_centroids, rasters)
    ]

    data.obs[f"{shape_prefix}_moment"] = moments


def _raster_polygon(poly, step=1):
    """
    Generate a grid of points contained within the poly. The points lie on
    a 2D grid, with vertices spaced step distance apart.
    """
    minx, miny, maxx, maxy = [int(i) for i in poly.bounds]
    x, y = np.meshgrid(
        np.arange(minx, maxx + step, step=step),
        np.arange(miny, maxy + step, step=step),
    )
    x = x.flatten()
    y = y.flatten()
    xy = np.array([x, y]).T

    poly_cell_mask = np.ones(xy.shape[0], dtype=bool)

    # Add all points within the polygon; handle MultiPolygons
    if isinstance(poly, MultiPolygon):
        for p in poly:
            poly_path = mplPath.Path(np.array(p.exterior.xy).T)
            poly_cell_mask = poly_cell_mask & poly_path.contains_points(xy)
    else:
        poly_path = mplPath.Path(np.array(poly.exterior.xy).T)
        poly_cell_mask = poly_path.contains_points(xy)
    xy = xy[poly_cell_mask]

    # Make sure at least a single point is returned
    if xy.shape[0] == 0:
        return np.array(poly.centroid.xy).reshape(1, 2)
    return xy


def _raster(data, shape_name, step=1):
    """Generate a grid of points contained within each shape. The points lie on
    a 2D grid, with vertices spaced `step` distance apart.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData

    Returns
    -------
    data : anndata.AnnData
        Returns `data` if `copy=True`, otherwise adds fields to `data`:

        `uns['{shape}_raster']` : np.array
            Long DataFrame of points annotated by shape from `obs['{shape_name}']`
    """

    raster = data.obs[f"{shape_name}"].apply(
        lambda poly: _raster_polygon(poly, step=step)
    )

    shape_prefix = shape_name.split("_")[0]
    raster_all = []
    for s, r in raster.items():
        raster_df = pd.DataFrame(r, columns=["x", "y"])
        raster_df[shape_prefix] = s
        raster_all.append(raster_df)

    # Add raster to data.obs as 2d array per cell (for point_features compatibility)
    data.obs[f"{shape_prefix}_raster"] = [df[["x", "y"]].values for df in raster_all]

    # Add raster to data.uns as long dataframe (for flux compatibility)
    raster_all = pd.concat(raster_all).reset_index(drop=True)
    data.uns[f"{shape_prefix}_raster"] = raster_all


def _perimeter(data, shape_name):
    """Compute the perimeter of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData

    Returns
    -------
    data : anndata.AnnData
        Returns `data` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_perimeter']` : np.array
            Perimeter of each polygon
    """

    shape_prefix = shape_name.split("_")[0]
    data.obs[f"{shape_prefix}_perimeter"] = get_shape(data, shape_name).length


def _radius(data, shape_name):
    """Compute the radius of each cell.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData

    Returns
    -------
    data : anndata.AnnData
        Returns `data` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_radius']` : np.array
            Radius of each polygon in `obs['cell_shape']`
    """

    shapes = get_shape(data, shape_name)

    # Get average distance from boundary to centroid
    shape_radius = shapes.apply(
        lambda c: distance.cdist(
            np.array(c.centroid).reshape(1, 2), np.array(c.exterior.xy).T
        ).mean()
    )

    shape_prefix = shape_name.split("_")[0]
    data.obs[f"{shape_prefix}_radius"] = shape_radius


def _span(data, shape_name):
    """Compute the length of the longest diagonal of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData

    Returns
    -------
    data : anndata.AnnData
        Returns `data` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_span']` : float
            Length of longest diagonal for each polygon
    """

    def get_span(poly):
        shape_coo = np.array(poly.coords.xy).T
        return int(distance_matrix(shape_coo, shape_coo).max())

    span = get_shape(data, shape_name).exterior.apply(get_span)

    shape_prefix = shape_name.split("_")[0]
    data.obs[f"{shape_prefix}_span"] = span


def list_shape_features():
    """Return a DataFrame of available shape features. Pulls descriptions from function docstrings.

    Returns
    -------
    list
        List of available shape features.
    """

    # Get shape feature descriptions from docstrings
    df = dict()
    for k, v in shape_features.items():
        description = v.__doc__.split("Parameters")[0].strip()
        df[k] = description

    return df


shape_features = dict(
    area=_area,
    aspect_ratio=_aspect_ratio,
    bounds=_bounds,
    density=_density,
    perimeter=_perimeter,
    radius=_radius,
    raster=_raster,
    second_moment=_second_moment,
    span=_span,
)


def obs_stats(
    data: AnnData,
    feature_names: List[str] = ["area", "aspect_ratio", "density"],
    copy=False,
):
    """Compute features for each cell shape. Convenient wrapper for `bento.tl.shape_features`.
    See list of available features in `bento.tl.shape_features`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    feature_names : list
        List of features to compute. See list of available features in `bento.tl.shape_features`.
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    data : anndata.AnnData
        Returns `data` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_{feature}']` : np.array
            Feature of each polygon
    """
    adata = data.copy() if copy else data

    # Compute features
    analyze_shapes(adata, "cell_shape", feature_names, copy=copy)
    if "nucleus_shape" in adata.obs.columns:
        analyze_shapes(adata, "nucleus_shape", feature_names, copy=copy)

    return adata if copy else None


@track
def analyze_shapes(
    data: AnnData,
    shape_names: Union[str, List[str]],
    feature_names: Union[str, List[str]],
    feature_kws: Dict[str, Dict] = None,
    progress: bool = True,
    copy: bool = False,
):
    """Analyze features of shapes.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    shape_names : list of str
        List of shapes to analyze.
    feature_names : list of str
        List of features to analyze.
    feature_kws : dict, optional (default: None)
        Keyword arguments for each feature.
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : AnnData
        See specific feature function docs for fields added.

    """
    adata = data.copy() if copy else data

    # Cast to list if not already
    if isinstance(shape_names, str):
        shape_names = [shape_names]

    # Cast to list if not already
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    # Generate feature x shape combinations
    combos = [(f, s) for f in feature_names for s in shape_names]

    # Set up progress bar
    if progress:
        combos = tqdm(combos)

    # Analyze each feature x shape combination
    for feature, shape in combos:
        if feature_kws and feature in feature_kws:
            shape_features[feature](adata, shape, **feature_kws[feature])
        else:
            shape_features[feature](adata, shape)

    return adata if copy else None


def register_shape_feature(name: str, func: Callable):
    """Register a shape feature function. The function should take an AnnData object and a shape name as input.
    The function should add the feature to the AnnData object as a column in AnnData.obs. This should be done in place and not return anything.

    Parameters
    ----------
    name : str
        Name of the feature function.
    func : function
        Function that takes an AnnData object and a shape name as arguments.
    """
    shape_features[name] = func

    # TODO perform some checks on the function

    print(f"Registered shape feature '{name}' to `bento.tl.shape_features`.")
