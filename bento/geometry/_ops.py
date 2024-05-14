# Geometric operations for SpatialData ShapeElements wrapping GeoPandas GeoDataFrames.


from functools import singledispatch

import geopandas as gpd
import numpy as np
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import ShapesModel

from .._utils import get_feature_key, get_instance_key
from ..io import prep


def overlay(
    sdata: SpatialData,
    s1: str,
    s2: str,
    name: str,
    how: str = "intersection",
    make_valid: bool = True,
):
    """Overlay two shape elements in a SpatialData object and store the result as a new shape element.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    s1 : str
        Name of the first shape element
    s2 : str
        Name of the second shape element
    how : str
        Type of overlay operation to perform. Options are "intersection", "union", "difference", "symmetric_difference", by default "intersection"
    make_valid : bool
        If True, correct invalid geometries with GeoPandas, by default True
    instance_key : str
        Name of the shape element to use as the instance for indexing, by default "cell_boundaries". If None, no indexing is performed.

    Returns
    -------
    SpatialData
        A new SpatialData object with the resulting shapes from the overlay operation.
    """
    shape1 = sdata[s1]
    shape2 = sdata[s2]

    new_shape = shape1.overlay(shape2, how=how, make_valid=make_valid)
    new_shape.attrs = {}

    transform = shape1.attrs
    sdata.shapes[name] = ShapesModel.parse(new_shape)
    sdata.shapes[name].attrs = transform

    sdata = prep(
        sdata,
        shape_keys=[name],
        instance_key=get_instance_key(sdata),
        feature_key=get_feature_key(sdata),
    )



@singledispatch
def labels_to_shapes(labels: np.ndarray, attrs: dict, bg_value: int = 0):
    """
    Given a labeled 2D image, convert encoded pixels as Polygons and return a SpatialData verified GeoPandas DataFrame.

    Parameters
    ----------
    labels : np.ndarray
        Labeled 2D image where each pixel is encoded with an integer value.
    attrs : dict
        Dictionary of attributes to set for the SpatialData object.
    bg_value : int, optional
        Value of the background pixels, by default 0

    Returns
    -------
    GeoPandas DataFrame
        GeoPandas DataFrame containing the polygons extracted from the labeled image.
    
    """
    import rasterio as rio
    import shapely.geometry

    # Extract polygons from labeled image
    contours = rio.features.shapes(labels)
    polygons = np.array([(shapely.geometry.shape(p), v) for p, v in contours])
    shapes = gpd.GeoDataFrame(
        polygons[:, 1], geometry=gpd.GeoSeries(polygons[:, 0]).T, columns=["id"]
    )
    shapes = shapes[shapes["id"] != bg_value] # Ignore background
    
    # Validate for SpatialData
    sd_shape = ShapesModel.parse(shapes)
    sd_shape.attrs = attrs
    return sd_shape


@labels_to_shapes.register(SpatialImage)
def _(labels: SpatialImage, attrs: dict, bg_value: int = 0):
    """
    Given a labeled 2D image, convert encoded pixels as Polygons and return a SpatialData verified GeoPandas DataFrame.

    Parameters
    ----------
    labels : SpatialImage
        Labeled 2D image where each pixel is encoded with an integer value.
    attrs : dict
        Dictionary of attributes to set for the SpatialData object.
    bg_value : int, optional
        Value of the background pixels, by default 0
    
    Returns
    -------
    GeoPandas DataFrame
        GeoPandas DataFrame containing the polygons extracted from the labeled image.
    """
        
    # Convert spatial_image.SpatialImage to np.ndarray
    labels = labels.values
    return labels_to_shapes(labels, attrs, bg_value)