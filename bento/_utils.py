
from spatialdata import SpatialData


def get_instance_key(sdata: SpatialData):
    """
    Returns the instance key for the spatial data object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.

    Returns
    -------
    instance_key : str
        Key for the shape that will be used as the instance for all indexing. Usually the cell shape.
    """
    try:
        return sdata.points["transcripts"].attrs["spatialdata_attrs"]["instance_key"]
    except KeyError:
        raise KeyError("Instance key attribute not found in spatialdata object.")
    

def get_feature_key(sdata: SpatialData):
    """
    Returns the feature key for the spatial data object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.

    Returns
    -------
    feature_key : str
        Key for the feature name in the points DataFrame
    """
    try:
        return sdata.points["transcripts"].attrs["spatialdata_attrs"]["feature_key"]
    except KeyError:
        raise KeyError("Feature key attribute not found in spatialdata object.")