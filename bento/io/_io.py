import warnings
from typing import List

warnings.filterwarnings("ignore")

from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import ShapesModel

# from .._utils import sc_format
from ..geometry import sindex_points, sjoin_shapes


def format_sdata(
    sdata: SpatialData,
    points_key: str = "transcripts",
    cell_shape_key: str = "cell_boundaries",
    shape_names: List[str] = ["cell_boundaries", "nucleus_boundaries"],
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
        .points[point_key]: Updated points DataFrame with string index for each shape
    """

    # Renames geometry column of shape element to match shape name
    # Changes indices to strings
    for shape_name, shape_gdf in sdata.shapes.items():
        if shape_name == cell_shape_key:
            shape_gdf[shape_name] = shape_gdf["geometry"]
        if isinstance(shape_gdf.index[0], str):
            shape_gdf.index = shape_gdf.index.astype(str, copy=False)
        sdata.shapes[shape_name] = ShapesModel.parse(shape_gdf)

    # sindex points and sjoin shapes if they have not been indexed or joined
    point_sjoin = []
    shape_sjoin = []
    for shape_name in shape_names:
        if shape_name.split("_")[0] not in sdata.points[points_key].columns:
            point_sjoin.append(shape_name)
        if (
            shape_name != cell_shape_key
            and shape_name not in sdata.shapes[cell_shape_key].columns
        ):
            shape_sjoin.append(shape_name)

    if len(point_sjoin) > 0:
        sdata = sindex_points(
            sdata=sdata, shape_names=point_sjoin, points_key=points_key
        )
    if len(shape_sjoin) > 0:
        sdata = sjoin_shapes(sdata=sdata, cell_shape_key=cell_shape_key, shape_names=shape_sjoin)

    return sdata
