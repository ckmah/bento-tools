from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import dask.dataframe as dd
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from shapely.geometry import Polygon

from .._utils import sync_points


def sindex_points(
    sdata: SpatialData,
    points_key: str = "transcripts",
    shape_names: List[str] = ["cell_boundaries"],
):
    """Index points to shapes and add as columns to `data.points[points_key]`. Only supports 2D points for now.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    points_key : str
        Key for points DataFrame in `sdata.points`
    shape_names : str, list
        List of shape names to index points to

    Returns
    -------
    sdata : SpatialData
        .points[points_key]: Updated points DataFrame with string index for each shape
    """

    if isinstance(shape_names, str):
        shape_names = [shape_names]

    # Grab all shape GeoDataFrames to index points to
    shape_gpds = {}
    for shape in shape_names:
        shape_gpds[shape] = sdata.shapes[shape]

    # Grab points as GeoDataFrame
    points = sdata.points[points_key].compute()
    attrs = sdata.points[points_key].attrs
    points_gpd = gpd.GeoDataFrame(
        points, geometry=gpd.points_from_xy(points.x, points.y), copy=True
    )

    # Index points to shapes
    for shape in shape_gpds:
        shape_gpd = shape_gpds[shape]
        shape_gpd.index = shape_gpd.index.astype(str)
        sjoined_points = gpd.sjoin(
            points_gpd, shape_gpd, how="left", predicate="intersects"
        )
        sjoined_points = sjoined_points[~sjoined_points.index.duplicated(keep="last")]
        sjoined_points.loc[sjoined_points["index_right"].isna(), "index_right"] = ""
        points[shape] = sjoined_points["index_right"].astype("category")

    sdata.points[points_key] = PointsModel.parse(
        points, coordinates={"x": "x", "y": "y"}
    )
    sdata.points[points_key].attrs = attrs

    return sdata


def sjoin_shapes(sdata: SpatialData, instance_key: str, shape_names: List[str]):
    """Adds polygon indexes to sdata.shapes[instance_key][shape_name] for point feature analysis

    Parameters
    ----------
    sdata : SpatialData
        Spatially formatted SpatialData
    instance_key : str
        Key for the shape that will be used as the instance for all indexing. Usually the cell shape.
    shape_names : str or list of str
        Names of the shapes to add.

    Returns
    -------
    sdata : SpatialData
        .shapes[cell_shape_key][shape_name]
    """

    # Cast to list if not already
    if isinstance(shape_names, str):
        shape_names = [shape_names]

    # Check if shapes are already indexed to instance_key shape
    missing_shape_names = set(shape_names) - set(sdata.shapes[instance_key].columns)

    if len(missing_shape_names) == 0:
        return sdata
    else:
        shape_names = missing_shape_names

    sjoined_shapes = sdata.shapes[instance_key]
    transform = sdata.shapes[instance_key].attrs

    # sjoin shapes to instance_key shape
    for shape in shape_names:
        shape_gdf = gpd.GeoDataFrame(geometry=sdata.shapes[shape]["geometry"])
        sjoined_shapes = sjoined_shapes.sjoin(
            shape_gdf, how="left", predicate="contains"
        )

        # save shape index as column in instance_key shape
        id_col = f"{shape}_id"
        sjoined_shapes.rename(columns={"index_right": id_col}, inplace=True)


    # Add to sdata.shapes
    sdata.shapes[instance_key] = ShapesModel.parse(sjoined_shapes)
    sdata.shapes[instance_key].attrs = transform

    return sdata


def get_shape(sdata: SpatialData, shape_name: str, sync: bool = True) -> gpd.GeoSeries:
    """Get a GeoSeries of Polygon objects from an SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    shape_name : str
        Name of shape column in sdata.shapes
    sync : bool
        Whether to retrieve shapes synced to cell shape. Default True.

    Returns
    -------
    GeoSeries
        GeoSeries of Polygon objects
    """
    instance_key = sdata.table.uns["spatialdata_attrs"]["instance_key"]
    if sync and shape_name != instance_key:
        check_shape_sync(sdata, shape_name, instance_key)
        return sdata.shapes[instance_key][shape_name]

    if shape_name not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_name} not found in sdata.shapes")

    return sdata.shapes[shape_name].geometry


def check_shape_sync(sdata, shape_name, instance_key):
    """
    Check if a shape is synced to instance_key shape in a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to check.
    shape_name : str
        The name of the shape to check.
    instance_key : str
        The instance key of the shape to check.

    Raises
    ------
    ValueError
        If the shape is not synced to instance_key shape.
    """
    if (
        shape_name != instance_key
        and shape_name not in sdata.shapes[instance_key].columns
    ):
        raise ValueError(
            f"Shape {shape_name} not synced to cell_boundaries. Run bento.io.format_sdata() to setup SpatialData object for bento-tools."
        )


def set_shape_metadata(
    sdata: SpatialData,
    shape_name: str,
    metadata: Union[pd.Series, pd.DataFrame],
):
    """Write metadata in SpatialData shapes element as column(s). Aligns metadata index to shape index.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    shape_name : str
        Name of element in sdata.shapes
    metadata : pd.Series, pd.DataFrame
        Metadata to set for shape
    """
    if shape_name not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_name} not found in sdata.shapes")

    # Set metadata as columns in sdata.shape[shape_name]
    if isinstance(metadata, pd.Series):
        metadata = pd.DataFrame(metadata)
    sdata.shapes[shape_name][metadata.columns] = metadata
    return sdata


def get_points(
    sdata: SpatialData, points_key: str = "transcripts", astype: str = "pandas"
) -> Union[pd.DataFrame, dd.DataFrame, gpd.GeoDataFrame]:
    """Get points DataFrame synced to AnnData object.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object
    key : str, optional
        Key for `data.points` to use, by default "transcripts"
    astype : str, optional
        Whether to return a 'pandas' DataFrame, 'dask' DataFrame, or 'geopandas' GeoDataFrame, by default "pandas"

    Returns
    -------
    DataFrame or GeoDataFrame
        Returns `data.points[key]` as a `[Geo]DataFrame` or 'Dask DataFrame'
    """
    if astype not in ["pandas", "dask", "geopandas"]:
        raise ValueError(
            f"astype must be one of ['dask', 'pandas', 'geopandas'], not {astype}"
        )

    points = sync_points(sdata).points[points_key]

    if astype == "pandas":
        return points.compute()
    elif astype == "dask":
        return points
    elif astype == "geopandas":
        points = points.compute()
        return gpd.GeoDataFrame(
            points, geometry=gpd.points_from_xy(points.x, points.y), copy=True
        )


def get_points_metadata(
    sdata: SpatialData, metadata_key: str, points_key: str = "transcripts"
):
    """Get points metadata synced to SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    metadata_key : str
        Key for `sdata.points[points_key][key]` to use
    points_key : str, optional
        Key for `sdata.points` to use, by default "transcripts"

    Returns
    -------
    Series
        Returns `data.uns[key][metadata_key]` as a `Series`
    """
    metadata = sync_points(sdata).points[points_key][metadata_key].compute()
    return metadata
