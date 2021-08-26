import geopandas as gpd
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point


def cell_aspect_ratio(data, copy=False):

    adata = data.copy() if copy else data

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

    ar = adata.obs["cell_shape"].apply(lambda poly: _aspect_ratio(poly))
    adata.obs["aspect-ratio"] = ar

    return adata if copy else None


def cell_area(data, copy=False):
    adata = data.copy() if copy else data

    if not overwrite and 'cell_area' in adata.obs.columns:
        return adata if copy else None
    
    # Calculate pixel-wise area
    # TODO: unit scale?
    area = gpd.GeoSeries(adata.obs["cell_shape"]).area
    adata.obs["cell_area"] = area

    return adata if copy else None


def cell_radius(data, overwrite=False, copy=False):
    """
    Calculate the mean cell radius.
    """
    adata = data.copy() if copy else data
    
    if not overwrite and 'cell_radius' in adata.obs.columns:
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


def is_nuclear(data, shape_name, overwrite=False, copy=False):
    """
    Check if shape_name is contained within the nucleus.
    """
    adata = data.copy() if copy else data
    
    shape_prefix = shape_name.split('_shape')[0]
    if not overwrite and f"{shape_prefix}_in_nucleus" in adata.obs.columns:
        return adata if copy else None

    if shape_name == 'nucleus_shape':
        adata.obs["nucleus_in_nucleus"] = True
    else:
        shapes = gpd.GeoSeries(data.obs[shape_name])
        nuclei = gpd.GeoSeries(data.obs['nucleus_shape'])

        shape_in_nucleus = shapes.within(nuclei)
        adata.obs[f"{shape_prefix}_in_nucleus"] = shape_in_nucleus

    return adata if copy else None
