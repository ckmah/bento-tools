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
        SpatialData object containing the shape elements
    s1 : str
        Name of the first shape element
    s2 : str
        Name of the second shape element
    name : str
        Name of the new shape element to be created
    how : str, optional
        Type of overlay operation to perform. Options are "intersection", "union", "difference", "symmetric_difference", by default "intersection"
    make_valid : bool, optional
        If True, correct invalid geometries with GeoPandas, by default True

    Returns
    -------
    None
        The function modifies the input SpatialData object in-place
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
