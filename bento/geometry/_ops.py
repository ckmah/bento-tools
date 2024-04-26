# Geometric operations for SpatialData ShapeElements wrapping GeoPandas GeoDataFrames.


from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation

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
