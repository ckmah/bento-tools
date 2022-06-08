import geopandas as gpd
import matplotlib.path as mplPath
import numpy as np
from scipy.spatial import distance, distance_matrix
from shapely.geometry import Point
from tqdm.auto import tqdm

from .._utils import track


@track
def analyze_cells(data, features, copy=False):
    adata = data.copy() if copy else data

    if not isinstance(features, list):
        features = [features]

    features = list(set(features))

    for f in tqdm(features):
        cell_features[f].__wrapped__(adata)

    return adata if copy else None


@track
def cell_span(data, copy=False):
    adata = data.copy() if copy else data

    def get_span(poly):
        shape_coo = np.array(poly.coords.xy).T
        return int(distance_matrix(shape_coo, shape_coo).max())

    span = gpd.GeoSeries(data=adata.obs["cell_shape"]).exterior.apply(get_span)

    adata.obs["cell_span"] = span

    return adata if copy else None


@track
def cell_bounds(data, copy=False):
    adata = data.copy() if copy else data

    bounds = gpd.GeoSeries(data=adata.obs["cell_shape"]).bounds
    adata.obs["cell_minx"] = bounds["minx"]
    adata.obs["cell_miny"] = bounds["miny"]
    adata.obs["cell_maxx"] = bounds["maxx"]
    adata.obs["cell_maxy"] = bounds["maxy"]

    return adata if copy else None


@track
def cell_moments(data, copy=False):
    adata = data.copy() if copy else data

    if "cell_raster" not in adata.obs:
        raster_cell(adata)

    cell_rasters = adata.obs["cell_raster"]
    shape_centroids = gpd.GeoSeries(adata.obs["cell_shape"]).centroid
    cell_moments = [
        _second_moment(np.array(centroid.xy).reshape(1, 2), cell_raster)
        for centroid, cell_raster in zip(shape_centroids, cell_rasters)
    ]

    adata.obs["cell_moment"] = cell_moments

    return adata if copy else None


def _second_moment(centroid, pts):
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


def _raster_polygon(poly):
    """
    Rasterize polygon and return list of coordinates in body of polygon.
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


@track
def raster_cell(data, copy=False):
    adata = data.copy() if copy else data

    raster = adata.obs["cell_shape"].apply(_raster_polygon)
    adata.obs["cell_raster"] = raster

    return adata if copy else None


def _aspect_ratio(poly):
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


@track
def cell_aspect_ratio(data, copy=False):

    adata = data.copy() if copy else data

    ar = adata.obs["cell_shape"].apply(lambda poly: _aspect_ratio(poly))
    adata.obs["cell_aspect_ratio"] = ar

    return adata if copy else None


@track
def cell_density(data, copy=False):
    adata = data.copy() if copy else data

    cell_area.__wrapped__(adata)

    count = adata.X.sum(axis=1)
    adata.obs["cell_density"] = count / adata.obs["cell_area"]

    return adata if copy else None


@track
def cell_area(data, copy=False):
    adata = data.copy() if copy else data

    # Calculate pixel-wise area
    # TODO: unit scale?
    area = gpd.GeoSeries(adata.obs["cell_shape"]).area
    adata.obs["cell_area"] = area

    return adata if copy else None


@track
def cell_perimeter(data, copy=False):
    adata = data.copy() if copy else data

    adata.obs["cell_perimeter"] = gpd.GeoSeries(adata.obs["cell_shape"]).length

    return adata if copy else None


@track
def cell_radius(data, overwrite=False, copy=False):
    """
    Calculate the mean cell radius.
    """
    adata = data.copy() if copy else data

    if not overwrite and "cell_radius" in adata.obs.columns:
        return adata if copy else None

    cells = gpd.GeoSeries(adata.obs["cell_shape"])

    # Get average distance of cell boundary to centroid
    cell_radius = cells.apply(
        lambda c: distance.cdist(
            np.array(c.centroid).reshape(1, 2), np.array(c.exterior.xy).T
        ).mean()
    )
    adata.obs["cell_radius"] = cell_radius

    return adata if copy else None


@track
def cell_morph_open(data, proportion, overwrite=False, copy=False):
    """
    Perform opening (morphological) of distance d on cell_shape.
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
def nucleus_area(data, copy=False):
    adata = data.copy() if copy else data

    adata.obs["nucleus_area"] = gpd.GeoSeries(adata.obs["nucleus_shape"]).area

    return adata if copy else None


@track
def nucleus_aspect_ratio(data, copy=False):
    adata = data.copy() if copy else data

    ar = adata.obs["nucleus_shape"].apply(lambda poly: _aspect_ratio(poly))
    adata.obs["nucleus_aspect_ratio"] = ar

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


cell_features = dict(
    cell_span=cell_span,
    cell_bounds=cell_bounds,
    cell_moments=cell_moments,
    raster_cell=raster_cell,
    cell_aspect_ratio=cell_aspect_ratio,
    cell_density=cell_density,
    cell_area=cell_area,
    cell_perimeter=cell_perimeter,
    cell_radius=cell_radius,
    cell_morph_open=cell_morph_open,
)
