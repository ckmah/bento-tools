import geopandas as gpd
import ipywidgets as widgets
import matplotlib.path as mplPath
import numpy as np
from ipywidgets import interact_manual
from scipy.spatial import distance, distance_matrix
from shapely.geometry import Point
from tqdm.auto import tqdm

from .._utils import track
from ..preprocessing import get_points, set_points


# @track
def analyze_shapes(data, shape_names, feature_names, progress=True, copy=False):
    """Analyze features of shapes.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    shape_names : list of str
        List of shapes to analyze.
    feature_names : list of str
        List of features to analyze.
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    _type_
        _description_
    """
    adata = data.copy() if copy else data

    # Cast to list if not already
    if isinstance(shape_names, str):
        shape_names = [shape_names]
    elif isinstance(shape_names, tuple):
        shape_names = list(set(shape_names))

    # Cast to list if not already
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    elif isinstance(feature_names, tuple):
        feature_names = list(set(feature_names))

    # Generate feature x shape combinations
    combos = [(f, s) for f in feature_names for s in shape_names]

    # Set up progress bar
    if progress:
        combos = tqdm(combos, desc="Analyzing shapes")

    # Analyze each feature x shape combination
    for feature, shape in combos:
        feature_functions[feature](adata, shape)

    return adata if copy else None


def analyze_shapes_ui(data, progress=True):
    """Interactive UI for shape analysis.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    progress : bool, optional
        Show progress bars, by default True
    """
    shape_names = [col for col in data.obs.columns if "shape" in col]
    shape_names.sort()

    if len(shape_names) == 0:
        raise ValueError("No shape columns found in `data.obs`")

    w = interact_manual(
        analyze_shapes,
        data=widgets.fixed(data),
        shape_names=widgets.SelectMultiple(
            options=shape_names,
            description="Shapes:",
            disabled=False,
        ),
        feature_names=widgets.SelectMultiple(
            options=list(feature_functions.keys()),
            description="Features:",
            disabled=False,
        ),
        progress=widgets.fixed(progress),
        copy=widgets.fixed(False)
    )

    return w


def _area(data, shape_name, copy=False):
    """Compute the area of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_area']` : np.array
            Area of each polygon in `obs['{shape_name}']`
    """
    adata = data.copy() if copy else data

    # Calculate pixel-wise area
    # TODO: unit scale?
    area = gpd.GeoSeries(adata.obs[shape_name]).area

    shape_prefix = shape_name.split("_")[0]
    adata.obs[f"{shape_prefix}_area"] = area

    return adata if copy else None


def _poly_aspect_ratio(poly):
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


def _aspect_ratio(data, shape_name, copy=False):
    """Compute the aspect ratio of the minimum rotated rectangle that contains each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_aspect_ratio']` : np.array
            Ratio of long / short axis for each polygon in `obs['{shape_name}']`
    """

    adata = data.copy() if copy else data

    ar = adata.obs[shape_name].apply(lambda poly: _poly_aspect_ratio(poly))
    shape_prefix = shape_name.split("_")[0]
    adata.obs[f"{shape_prefix}_aspect_ratio"] = ar

    return adata if copy else None


def _bounds(data, shape_name, copy=False):
    """Compute the minimum and maximum coordinate values that bound each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_minx']` : float
            x-axis lower bound of each polygon in `obs['{shape_name}']`
        `obs['{shape}_miny']` : float
            y-axis lower bound of each polygon in `obs['{shape_name}']`
        `obs['{shape}_maxx']` : float
            x-axis upper bound of each polygon in `obs['{shape_name}']`
        `obs['{shape}_maxy']` : float
            y-axis upper bound of each polygon in `obs['{shape_name}']`
    """
    adata = data.copy() if copy else data

    bounds = gpd.GeoSeries(data=adata.obs[shape_name]).bounds

    shape_prefix = shape_name.split("_")[0]
    adata.obs[f"{shape_prefix}_minx"] = bounds["minx"]
    adata.obs[f"{shape_prefix}_miny"] = bounds["miny"]
    adata.obs[f"{shape_prefix}_maxx"] = bounds["maxx"]
    adata.obs[f"{shape_prefix}_maxy"] = bounds["maxy"]

    return adata if copy else None


def _density(data, shape_name, copy=False):
    """Compute the RNA density of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_density']` : np.array
            Density (total cell counts / shape area) of each polygon in `obs['{shape_name}']`
    """
    adata = data.copy() if copy else data

    shape_prefix = shape_name.split("_")[0]
    set_points(adata)
    count = get_points(adata).query(f"{shape_prefix} != '-1'").groupby("cell").size()
    _area(adata, shape_name)

    adata.obs[f"{shape_prefix}_density"] = count / adata.obs[f"{shape_prefix}_area"]

    return adata if copy else None


@track
def _opening(data, proportion, overwrite=False, copy=False):
    """Compute the opening (morphological) of distance d for each cell.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['cell_open_{d}_shape']` : Polygons
            Ratio of long / short axis for each polygon in `obs['cell_shape']`
    """
    adata = data.copy() if copy else data

    shape_name = f"cell_open_{proportion}_shape"

    if not overwrite and shape_name in adata.obs.columns:
        return adata if copy else None

    # Compute cell radius as needed
    cell_radius.__wrapped__(adata)

    cells = gpd.GeoSeries(adata.obs["cell_shape"])
    d = proportion * data.obs["cell_radius"]

    # Opening
    adata.obs[shape_name] = cells.buffer(-d).buffer(d)

    return adata if copy else None


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


def _second_moment(data, shape_name, copy=False):
    """Compute the second moment of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_moment']` : float
            The second moment for each polygon in `obs['{shape_name}']`
    """
    adata = data.copy() if copy else data

    shape_prefix = shape_name.split("_")[0]
    if f"{shape_prefix}_raster" not in adata.obs:
        _raster(adata, shape_name)

    rasters = adata.obs[f"{shape_prefix}_raster"]
    shape_centroids = gpd.GeoSeries(adata.obs[shape_name]).centroid

    moments = [
        _second_moment_polygon(np.array(centroid.xy).reshape(1, 2), r)
        for centroid, r in zip(shape_centroids, rasters)
    ]

    adata.obs[f"{shape_prefix}_moment"] = moments

    return adata if copy else None


def _raster_polygon(poly):
    """
    Generate a grid of points contained within the poly. The points lie on
    a 2D grid, with vertices spaced 1 unit apart.
    """
    minx, miny, maxx, maxy = poly.bounds
    x, y = np.meshgrid(
        np.arange(minx, maxx, step=float(1)),
        np.arange(miny, maxy, step=float(1)),
    )
    x = x.flatten()
    y = y.flatten()
    xy = np.array([x, y]).T
    poly_path = mplPath.Path(np.array(poly.exterior.xy).T)
    poly_cell_mask = poly_path.contains_points(xy)
    xy = xy[poly_cell_mask]
    return xy


def _raster(data, shape_name, copy=False):
    """Generate a grid of points contained within each shape. The points lie on
    a 2D grid, with vertices spaced 1 unit apart.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_raster']` : np.array
            2D array of grid points for each polygon in `obs['{shape_name}']`
    """
    adata = data.copy() if copy else data

    raster = adata.obs[f"{shape_name}"].apply(_raster_polygon)
    shape_prefix = shape_name.split("_")[0]
    adata.obs[f"{shape_prefix}_raster"] = raster

    return adata if copy else None


def _perimeter(data, shape_name, copy=False):
    """Compute the perimeter of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_perimeter']` : np.array
            Perimeter of each polygon in `obs['{shape_name}']`
    """
    adata = data.copy() if copy else data

    shape_prefix = shape_name.split("_")[0]
    adata.obs[f"{shape_prefix}_perimeter"] = gpd.GeoSeries(adata.obs[shape_name]).length

    return adata if copy else None


def _radius(data, shape_name, copy=False):
    """Compute the radius of each cell.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_radius']` : np.array
            Radius of each polygon in `obs['cell_shape']`
    """
    adata = data.copy() if copy else data

    shapes = gpd.GeoSeries(adata.obs[shape_name])

    # Get average distance from boundary to centroid
    shape_radius = shapes.apply(
        lambda c: distance.cdist(
            np.array(c.centroid).reshape(1, 2), np.array(c.exterior.xy).T
        ).mean()
    )

    shape_prefix = shape_name.split("_")[0]
    adata.obs[f"{shape_prefix}_radius"] = shape_radius

    return adata if copy else None


def _span(data, shape_name, copy=False):
    """Compute the length of the longest diagonal of each shape.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `obs['{shape}_span']` : float
            Length of longest diagonal for each polygon in `obs['{shape_name}']`
    """
    adata = data.copy() if copy else data

    def get_span(poly):
        shape_coo = np.array(poly.coords.xy).T
        return int(distance_matrix(shape_coo, shape_coo).max())

    span = gpd.GeoSeries(data=adata.obs[shape_name]).exterior.apply(get_span)

    shape_prefix = shape_name.split("_")[0]
    adata.obs[f"{shape_prefix}_span"] = span

    return adata if copy else None


@track
def nucleus_area_ratio(data, copy=False):
    adata = data.copy() if copy else data

    cell_area.__wrapped__(adata)
    nucleus_area.__wrapped__(adata)
    adata.obs["nucleus_area_ratio"] = adata.obs["nucleus_area"] / adata.obs["cell_area"]

    return adata if copy else None


@track
def nucleus_offset(data, copy=False):
    adata = data.copy() if copy else data

    cell_centroid = gpd.GeoSeries(adata.obs["cell_shape"]).centroid
    nucleus_centroid = gpd.GeoSeries(adata.obs["nucleus_shape"]).centroid

    cell_radius.__wrapped__(adata)
    offset = cell_centroid.distance(nucleus_centroid, align=False)
    offset = offset.apply(abs)

    adata.obs["nucleus_offset"] = offset

    return adata if copy else None


@track
def is_nuclear(data, shape_name, overwrite=False, copy=False):
    """
    Check if shape_name is contained within the nucleus.
    TODO speed up with sjoin
    """
    adata = data.copy() if copy else data

    shape_prefix = shape_name.split("_shape")[0]
    if not overwrite and f"{shape_prefix}_in_nucleus" in adata.obs.columns:
        return adata if copy else None

    if shape_name == "nucleus_shape":
        adata.obs["nucleus_in_nucleus"] = True
    else:
        shapes = gpd.GeoSeries(data.obs[shape_name])
        nuclei = gpd.GeoSeries(data.obs["nucleus_shape"])

        shape_in_nucleus = shapes.within(nuclei)
        adata.obs[f"{shape_prefix}_in_nucleus"] = shape_in_nucleus

    return adata if copy else None


feature_functions = dict(
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
