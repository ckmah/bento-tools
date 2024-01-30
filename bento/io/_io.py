from typing import List
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import ShapesModel
import anndata
from shapely import wkt

# from .._utils import sc_format
from ..geometry import sindex_points, sjoin_shapes

def format_sdata(
    sdata: SpatialData, points_key: str = "transcripts", cell_boundaries_key: str = "cell_boundaries", shape_names: List[str] = ["cell_boundaries", "nucleus_boundaries"]
) -> SpatialData:
    """Converts shape indices to strings and indexes points to shapes and add as columns to `data.points[point_key]`.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    point_key : str
        Key for points DataFrame in `sdata.points`
    shape_names : str, list
        List of shape names to index points to

    Returns
    -------
    SpatialData
        .shapes[shape_name]: Updated shapes GeoDataFrame with string index
        .points[point_key]: Updated points DataFrame with boolean column for each shape
    """
    
    # Renames geometry column of shape element to match shape name
    # Changes indices to strings
    for shape in sdata.shapes.keys():
        shape_gpd = sdata.shapes[shape]
        if shape == cell_boundaries_key:
            shape_gpd[shape] = shape_gpd['geometry']
        if type(shape_gpd.index[0]) != str:
            shape_gpd.index = shape_gpd.index.astype(str, copy = False)
        sdata.shapes[shape] = ShapesModel.parse(shape_gpd)
    
    # sindex points and sjoin shapes if they have not been indexed or joined
    point_sjoin = []
    shape_sjoin = []
    for shape_name in shape_names:
        if shape_name.split("_")[0] not in sdata.points[points_key].columns:
            point_sjoin.append(shape_name)
        if shape_name != cell_boundaries_key and shape_name not in sdata.shapes[cell_boundaries_key].columns:
            shape_sjoin.append(shape_name)

    if len(point_sjoin) != 0:
        sdata = sindex_points(sdata=sdata, shape_names=point_sjoin, points_key=points_key)
    if len(shape_sjoin) != 0:
        sdata = sjoin_shapes(sdata=sdata, shape_names=shape_sjoin)

    return sdata
