from typing import List
import warnings

warnings.filterwarnings("ignore")

from spatialdata._core.spatialdata import SpatialData

from ..geometry import sindex_points, sjoin_shapes

def format_sdata(
    sdata: SpatialData, points_key: str, shape_names: List[str]
) -> SpatialData:
    """Converts shape indices to strings and indexes points to shapes and add as columns to `data.points[points_key]`.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    points_key : str
        Key for points DataFrame in `data.points`
    shape_names : str, list
        List of shape names to index points to
    copy : bool, optional
        Whether to return a copy the SpatialData object. Default False.
    Returns
    -------
    SpatialData
        .shapes[shape_name]: Updated shapes GeoDataFrame with string index
        .points[points_key]: Updated points DataFrame with boolean column for each shape
    """
    # Renames geometry column of shape element to match shape name
    # Changes indices to strings
    for shape in sdata.shapes.keys():
        shape_gpd = sdata.shapes[shape]
        if shape_gpd.geometry.name != shape:
            shape_gpd.rename_geometry(shape, inplace=True)
        if type(shape_gpd.index[0]) != str:
            shape_gpd.index = shape_gpd.index.astype(str, copy = False)
    
    # sindex points and sjoin shapes if they have not been indexed or joined
    point_sjoin = []
    shape_sjoin = []
    for shape_name in shape_names:
        if shape_name.split("_")[0] not in sdata.points[points_key].columns:
            point_sjoin.append(shape_name)
        if shape_name != "cell_boundaries" and shape_name not in sdata.shapes["cell_boundaries"].columns:
            shape_sjoin.append(shape_name)

    if len(point_sjoin) != 0:
        sdata = sindex_points(sdata, points_key, shape_names)
    if len(shape_sjoin) != 0:
        sdata = sjoin_shapes(sdata, shape_sjoin)

    return sdata
