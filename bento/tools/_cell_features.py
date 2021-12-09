import geopandas as gpd
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point

from .._utils import track


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
def cell_area(data, copy=False):
    adata = data.copy() if copy else data

    # Calculate pixel-wise area
    # TODO: unit scale?
    area = gpd.GeoSeries(adata.obs["cell_shape"]).area
    adata.obs["cell_area"] = area

    return adata if copy else None


@track
def nucleus_area_ratio(data, copy=False):
    adata = data.copy() if copy else data

    cell_area(adata)
    nucleus_area(adata)
    adata.obs["nucleus_area_ratio"] = adata.obs["nucleus_area"] / adata.obs["cell_area"]

    return adata if copy else None


@track
def nucleus_offset(data, copy=False):
    adata = data.copy() if copy else data

    cell_centroid = gpd.GeoSeries(adata.obs["cell_shape"]).centroid
    nucleus_centroid = gpd.GeoSeries(adata.obs["nucleus_shape"]).centroid

    cell_radius(adata)
    offset = (
        cell_centroid.distance(nucleus_centroid, align=False)
    )
    offset = offset.apply(abs)

    adata.obs["nucleus_offset"] = offset

    return adata if copy else None


@track
def cell_perimeter(data, copy=False):
    adata = data.copy() if copy else data

    adata.obs["cell_perimeter"] = gpd.GeoSeries(adata.obs["cell_shape"]).length

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
def is_nuclear(data, shape_name, overwrite=False, copy=False):
    """
    Check if shape_name is contained within the nucleus.
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
    cell_radius(adata)

    cells = gpd.GeoSeries(adata.obs["cell_shape"])
    d = proportion * data.obs["cell_radius"]

    # Opening
    adata.obs[shape_name] = cells.buffer(-d).buffer(d)

    return adata if copy else None
