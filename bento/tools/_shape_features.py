import warnings

from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")


from typing import Callable, Dict, List, Union

import matplotlib.path as mplPath
import numpy as np
import pandas as pd
from scipy.spatial import distance, distance_matrix
from shapely.geometry import MultiPolygon, Point
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from tqdm.auto import tqdm

from .._utils import get_points, get_shape, set_shape_metadata


def area(sdata: SpatialData, shape_key: str, recompute: bool = False):
    """
    Compute the area of each shape.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    shape_key : str
        Key in `sdata.shapes[shape_key]` that contains the shape information.
    recompute : bool, optional
        If True, forces the computation of the area even if it already exists in the shape metadata. 
        If False (default), the computation is skipped if the area already exists.

    Returns
    ------
    .shapes[shape_key]['{shape}_area'] : float
        Area of each polygon
    """

    feature_key = f"{shape_key}_area"
    if feature_key in sdata.shapes[shape_key].columns and not recompute:
        return

    # Calculate pixel-wise area
    area = get_shape(sdata=sdata, shape_key=shape_key, sync=False).area
    set_shape_metadata(sdata=sdata, shape_key=shape_key, metadata=area, column_names=feature_key)


def _poly_aspect_ratio(poly):
    """Compute the aspect ratio of the minimum rotated rectangle that contains a polygon."""

    if not poly:
        return np.nan

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


def aspect_ratio(sdata: SpatialData, shape_key: str, recompute: bool = False):
    """Compute the aspect ratio of the minimum rotated rectangle that contains each shape.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    shape_key : str
        Key in `sdata.shapes[shape_key]` that contains the shape information.

    Fields
    ------
        .shapes[shape_key]['{shape}_aspect_ratio'] : float
            Ratio of major to minor axis for each polygon
    """

    feature_key = f"{shape_key}_aspect_ratio"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    ar = get_shape(sdata, shape_key, sync=False).apply(_poly_aspect_ratio)
    set_shape_metadata(sdata=sdata, shape_key=shape_key, metadata=ar, column_names=feature_key)


def bounds(sdata: SpatialData, shape_key: str, recompute: bool = False):
    """Compute the minimum and maximum coordinate values that bound each shape.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    shape_key : str
        Key in `sdata.shapes[shape_key]` that contains the shape information.

    Returns
    ------
        .shapes[shape_key]['{shape}_minx'] : float
            x-axis lower bound of each polygon
        .shapes[shape_key]['{shape}_miny'] : float
            y-axis lower bound of each polygon
        .shapes[shape_key]['{shape}_maxx'] : float
            x-axis upper bound of each polygon
        .shapes[shape_key]['{shape}_maxy'] : float
            y-axis upper bound of each polygon
    """

    feat_names = ["minx", "miny", "maxx", "maxy"]
    feature_keys = [
        f"{shape_key}_{k}" for k in feat_names
    ]
    if (
        all([k in sdata.shapes[shape_key].keys() for k in feature_keys])
        and not recompute
    ):
        return

    bounds = get_shape(sdata, shape_key, sync=False).bounds

    set_shape_metadata(sdata=sdata, shape_key=shape_key, metadata=bounds[feat_names], column_names=feature_keys)


def density(sdata: SpatialData, shape_key: str, recompute: bool = False):
    """Compute the RNA density of each shape.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    shape_key : str
        Key in `sdata.shapes[shape_key]` that contains the shape information.

    Returns
    ------
        .shapes[shape_key]['{shape}_density'] : float
            Density (molecules / shape area) of each polygon
    """

    
    feature_key = f"{shape_key}_density"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    count = (
        get_points(sdata, astype="dask", sync=False)
        .query(f"{shape_key} != 'None'")[shape_key]
        .value_counts()
        .compute()
    )
    area(sdata, shape_key)

    set_shape_metadata(
        sdata=sdata, 
        shape_key=shape_key, 
        metadata=count / sdata.shapes[shape_key][f"{shape_key}_area"], 
        column_names=feature_key
    )


def opening(sdata: SpatialData, shape_key: str, proportion: float, recompute: bool = False):
    """Compute the opening (morphological) of distance d for each cell.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData

    Returns
    -------
        .shapes[shape_key]['cell_open_{d}_shape'] : Polygons
            Ratio of long / short axis for each polygon in `.shapes[shape_key]['cell_boundaries']`
    """

    feature_key = f"{shape_key}_open_{proportion}_shape"

    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    radius(sdata, shape_key)

    shapes = get_shape(sdata, shape_key, sync=False)
    d = proportion * sdata.shapes[shape_key][f"{shape_key}_radius"]
    set_shape_metadata(sdata=sdata, shape_key=shape_key, metadata=shapes.buffer(-d).buffer(d), column_names=feature_key)


def _second_moment_polygon(centroid, pts):
    """
    Calculate second moment of points with centroid as reference.

    Parameters
    ----------
    centroid : 2D Point object
    pts : [n x 2] float
    """

    if not centroid or not isinstance(pts, np.ndarray):
        return
    centroid = np.array(centroid.coords).reshape(1, 2)
    radii = distance.cdist(centroid, pts)
    second_moment = np.sum(radii * radii / len(pts))
    return second_moment


def second_moment(sdata: SpatialData, shape_key: str, recompute: bool = False):
    """Compute the second moment of each shape.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData

    Returns
    -------
        .shapes[shape_key]['{shape}_moment'] : float
            The second moment for each polygon
    """

    
    feature_key = f"{shape_key}_moment"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    raster(sdata, shape_key, recompute=recompute)

    rasters = sdata.shapes[shape_key][f"{shape_key}_raster"]
    shape_centroids = get_shape(sdata, shape_key, sync=False).centroid

    moments = [
        _second_moment_polygon(centroid, r)
        for centroid, r in zip(shape_centroids, rasters)
    ]

    set_shape_metadata(sdata=sdata, shape_key=shape_key, metadata=moments, column_names=feature_key)

def _raster_polygon(poly, step=1):
    """
    Generate a grid of points contained within the poly. The points lie on
    a 2D grid, with vertices spaced step distance apart.
    """
    if not poly:
        return
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


def raster(
    sdata: SpatialData,
    shape_key: str,
    points_key: str = "transcripts",
    step: int = 1,
    recompute: bool = False,
):
    """Generate a grid of points contained within each shape. The points lie on
    a 2D grid, with vertices spaced `step` distance apart.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData

    Returns
    -------
        .shapes[shape_key]['{shape}_raster'] : np.array
            Long DataFrame of points annotated by shape from `.shapes[shape_key]['{shape_key}']`
    """

    
    feature_key = f"{shape_key}_raster"

    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    shapes = get_shape(sdata, shape_key, sync=False)
    raster = shapes.apply(lambda poly: _raster_polygon(poly, step=step))

    raster_all = []
    for s, r in raster.items():
        raster_df = pd.DataFrame(r, columns=["x", "y"])
        raster_df[shape_key] = s
        raster_all.append(raster_df)

    # Add raster to sdata.shapes as 2d array per cell (for point_features compatibility)
    set_shape_metadata(
        sdata=sdata, 
        shape_key=shape_key, 
        metadata=[df[["x", "y"]].values for df in raster_all], 
        column_names=feature_key
    )

    # Add raster to sdata.points as long dataframe (for flux compatibility)
    raster_all = pd.concat(raster_all).reset_index(drop=True)
    transform = sdata.points[points_key].attrs
    sdata.points[feature_key] = PointsModel.parse(
        raster_all, coordinates={"x": "x", "y": "y"}
    )
    sdata.points[feature_key].attrs = transform


def perimeter(sdata: SpatialData, shape_key: str, recompute: bool = False):
    """Compute the perimeter of each shape.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData

    Returns
    -------
        `.shapes[shape_key]['{shape}_perimeter']` : np.array
            Perimeter of each polygon
    """

    
    feature_key = f"{shape_key}_perimeter"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    set_shape_metadata(
        sdata=sdata, 
        shape_key=shape_key, 
        metadata=get_shape(sdata, shape_key, sync=False).length, 
        column_names=feature_key
    )


def radius(sdata: SpatialData, shape_key: str, recompute: bool = False):
    """Compute the radius of each cell.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData

    Returns
    -------
        .shapes[shape_key]['{shape}_radius'] : np.array
            Radius of each polygon in `obs['cell_shape']`
    """

    
    feature_key = f"{shape_key}_radius"
    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    shapes = get_shape(sdata, shape_key, sync=False)

    # Get average distance from boundary to centroid
    shape_radius = shapes.apply(_shape_radius)
    set_shape_metadata(sdata=sdata, shape_key=shape_key, metadata=shape_radius, column_names=feature_key)


def _shape_radius(poly):
    if not poly:
        return np.nan

    return distance.cdist(
        np.array(poly.centroid.coords).reshape(1, 2), np.array(poly.exterior.xy).T
    ).mean()


def span(sdata: SpatialData, shape_key: str, recompute: bool = False):
    """Compute the length of the longest diagonal of each shape.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData

    Returns
    -------
        .shapes[shape_key]['{shape}_span'] : float
            Length of longest diagonal for each polygon
    """

    
    feature_key = f"{shape_key}_span"

    if feature_key in sdata.shapes[shape_key].keys() and not recompute:
        return

    def get_span(poly):
        if not poly:
            return np.nan

        shape_coo = np.array(poly.coords.xy).T
        return int(distance_matrix(shape_coo, shape_coo).max())

    span = get_shape(sdata, shape_key, sync=False).exterior.apply(get_span)
    set_shape_metadata(sdata=sdata, shape_key=shape_key, metadata=span, column_names=feature_key)
    

def list_shape_features():
    """Return a dictionary of available shape features and their descriptions.

    Returns
    -------
    dict
        A dictionary where keys are shape feature names and values are their corresponding descriptions.
    """

    # Get shape feature descriptions from docstrings
    df = dict()
    for k, v in shape_features.items():
        description = v.__doc__.split("Parameters")[0].strip()
        df[k] = description

    return df


shape_features = dict(
    area=area,
    aspect_ratio=aspect_ratio,
    bounds=bounds,
    density=density,
    opening=opening,
    perimeter=perimeter,
    radius=radius,
    raster=raster,
    second_moment=second_moment,
    span=span,
)


def shape_stats(
    sdata: SpatialData,
    feature_names: List[str] = ["area", "aspect_ratio", "density"],
):
    """Compute descriptive stats for cells. Convenient wrapper for `bento.tl.shape_features`.
    See list of available features in `bento.tl.shape_features`.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    feature_names : list
        List of features to compute. See list of available features in `bento.tl.shape_features`.

    Returns
    -------
        .shapes['cell_boundaries']['cell_boundaries_{feature}'] : np.array
            Feature of each polygon
    """

    # Compute features
    analyze_shapes(sdata, "cell_boundaries", feature_names)
    if "nucleus_boundaries" in sdata.shapes.keys():
        analyze_shapes(sdata, "nucleus_boundaries", feature_names)


def analyze_shapes(
    sdata: SpatialData,
    shape_keys: Union[str, List[str]],
    feature_names: Union[str, List[str]],
    feature_kws: Dict[str, Dict] = None,
    recompute: bool = False,
    progress: bool = True,
):
    """Analyze features of shapes.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData
    shape_keys : list of str
        List of shapes to analyze.
    feature_names : list of str
        List of features to analyze.
    feature_kws : dict, optional (default: None)
        Keyword arguments for each feature.

    Returns
    -------
    sdata : SpatialData
        See specific feature function docs for fields added.

    """

    # Cast to list if not already
    if isinstance(shape_keys, str):
        shape_keys = [shape_keys]

    # Cast to list if not already
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    # Generate feature x shape combinations
    combos = [(f, s) for f in feature_names for s in shape_keys]

    # Set up progress bar
    if progress:
        combos = tqdm(combos)

    # Analyze each feature x shape combination
    for feature, shape in combos:
        kws = dict(recompute=recompute)
        if feature_kws and feature in feature_kws:
            kws.update(feature_kws[feature])

        shape_features[feature](sdata, shape, **kws)



def register_shape_feature(name: str, func: Callable):
    """Register a shape feature function. The function should take an SpatialData object and a shape name as input.
       The function should add the feature to the SpatialData object as a column in SpatialData.table.obs. 
       This should be done in place and not return anything.

    Parameters
    ----------
    name : str
        Name of the feature function.
    func : function
        Function that takes a SpatialData object and a shape name as arguments.
    """
    shape_features[name] = func

    # TODO perform some checks on the function

    print(f"Registered shape feature '{name}' to `bento.tl.shape_features`.")
