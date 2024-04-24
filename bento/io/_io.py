from pathlib import Path
import warnings
from typing import List, Optional, Union

warnings.filterwarnings("ignore")

import spatialdata as sd
from spatialdata import SpatialData
from spatialdata.models import TableModel

from ..geometry import sjoin_points, sjoin_shapes
import zarr


def read_zarr(
    store: Union[str, Path, zarr.Group],
    selection: Optional[tuple[str]] = None,
    points_key: str = "transcripts",
    feature_key: str = "feature_name",
    instance_key: str = "cell_boundaries",
    shape_keys: List[str] = ["cell_boundaries", "nucleus_boundaries"],
) -> SpatialData:
    sdata = sd.read_zarr(store, selection=selection)
    return prep(sdata)


def prep(
    sdata: SpatialData,
    points_key: str = "transcripts",
    feature_key: str = "feature_name",
    instance_key: str = "cell_boundaries",
    shape_keys: List[str] = ["cell_boundaries", "nucleus_boundaries"],
) -> SpatialData:
    """Computes spatial indices for elements in SpatialData to enable usage of bento-tools.

    Specifically, this function indexes points to shapes and joins shapes to the instance shape. It also computes a count table for the points.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    points_key : str
        Key for points DataFrame in `sdata.points`
    feature_key : str
        Key for the feature name in the points DataFrame
    instance_key : str
        Key for the shape that will be used as the instance for all indexing. Usually the cell shape.
    shape_keys : str, list
        List of shape names to index points to

    Returns
    -------
    SpatialData
        .shapes[shape_key]: Updated shapes GeoDataFrame with string index
        .points[points_key]: Updated points DataFrame with string index for each shape
    """

    # Renames geometry column of shape element to match shape name
    # Changes indices to strings
    for shape_key, shape_gdf in sdata.shapes.items():
        if shape_key == instance_key:
            shape_gdf[shape_key] = shape_gdf["geometry"]
        shape_gdf.index = shape_gdf.index.astype(str, copy=False)

    # sindex points and sjoin shapes if they have not been indexed or joined
    point_sjoin = []
    shape_sjoin = []

    for shape_key in shape_keys:
        # Compile list of shapes that need to be indexed to points
        if shape_key not in sdata.points[points_key].columns:
            point_sjoin.append(shape_key)
        # Compile list of shapes that need to be joined to instance shape
        if (
            shape_key != instance_key
            and shape_key not in sdata.shapes[instance_key].columns
        ):
            shape_sjoin.append(shape_key)

    if len(point_sjoin) > 0:
        print("Indexing points...")
        sdata = sjoin_points(sdata=sdata, points_key=points_key, shape_keys=point_sjoin)
    if len(shape_sjoin) > 0:
        print("Indexing shapes...")
        sdata = sjoin_shapes(
            sdata=sdata, instance_key=instance_key, shape_keys=shape_sjoin
        )

    # Recompute count table
    print("Aggregating counts...")
    table = TableModel.parse(
        sdata.aggregate(
            values=points_key,
            instance_key=instance_key,
            by=instance_key,
            value_key=feature_key,
            aggfunc="count",
        ).table
    )

    try:
        del sdata.table
    except KeyError:
        pass
    
    sdata.table = table
    # Set instance key to cell_shape_key for all points and table
    sdata.points[points_key].attrs["spatialdata_attrs"]["instance_key"] = instance_key
    sdata.points[points_key].attrs["spatialdata_attrs"]["feature_key"] = feature_key

    print("Done.")

    return sdata
