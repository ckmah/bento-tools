from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import dask.dataframe as dd
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from shapely.geometry import Polygon


def sjoin_points(
    sdata: SpatialData,
    points_key: str,
    shape_keys: List[str],
):
    """Index points to shapes and add as columns to `data.points[points_key]`. Only supports 2D points for now.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    points_key : str
        Key for points DataFrame in `sdata.points`
    shape_keys : str, list
        List of shape names to index points to

    Returns
    -------
    sdata : SpatialData
        .points[points_key]: Updated points DataFrame with string index for each shape
    """

    if isinstance(shape_keys, str):
        shape_keys = [shape_keys]

    # Grab all shape GeoDataFrames to index points to
    query_shapes = {}
    for shape in shape_keys:
        query_shapes[shape] = gpd.GeoDataFrame(geometry=sdata.shapes[shape].geometry)

    # Grab points as GeoDataFrame
    points = get_points(sdata, points_key, astype="geopandas", sync=False)

    # Index points to shapes
    for shape_key, shape in query_shapes.items():
        shape = query_shapes[shape_key]
        shape.index.name = None
        shape.index = shape.index.astype(str)

        points = points.sjoin(shape, how="left", predicate="intersects")
        points = points[~points.index.duplicated(keep="last")]
        points.loc[points["index_right"].isna(), "index_right"] = ""
        points.rename(columns={"index_right": shape_key}, inplace=True)

        set_points_metadata(sdata, points_key, points[shape_key])
    
    return sdata


def sjoin_shapes(
        sdata: SpatialData, 
        instance_key: str, 
        shape_keys: List[str]
):
    """Adds polygon indexes to sdata.shapes[instance_key][shape_key] for point feature analysis

    Parameters
    ----------
    sdata : SpatialData
        Spatially formatted SpatialData
    instance_key : str
        Key for the shape that will be used as the instance for all indexing. Usually the cell shape.
    shape_keys : str or list of str
        Names of the shapes to add.

    Returns
    -------
    sdata : SpatialData
        .shapes[cell_shape_key][shape_key]
    """

    # Cast to list if not already
    if isinstance(shape_keys, str):
        shape_keys = [shape_keys]

    # Check if shapes are already indexed to instance_key shape
    shape_keys = (
        set(shape_keys) - set(sdata.shapes[instance_key].columns) - set(instance_key)
    )

    if len(shape_keys) == 0:
        return sdata

    parent_shape = sdata.shapes[instance_key]

    # sjoin shapes to instance_key shape
    for shape_key in shape_keys:
        child_shape = gpd.GeoDataFrame(geometry=sdata.shapes[shape_key]["geometry"])
        parent_shape = parent_shape.sjoin(child_shape, how="left", predicate="contains")
        parent_shape = parent_shape[~parent_shape.index.duplicated(keep="last")]
        parent_shape.loc[parent_shape["index_right"].isna(), "index_right"] = ""
        parent_shape = parent_shape.astype({"index_right": "category"})

        # save shape index as column in instance_key shape
        parent_shape.rename(columns={"index_right": shape_key}, inplace=True)
        set_shape_metadata(sdata, shape_key=instance_key, metadata=parent_shape[shape_key])

        # Add instance_key shape index to shape
        parent_shape.index.name = "parent_index"
        instance_index = parent_shape.reset_index().set_index(shape_key)["parent_index"]
        instance_index.name = instance_key
        instance_index.index.name = None
        instance_index = instance_index[instance_index.index != ""]

        set_shape_metadata(sdata, shape_key=shape_key, metadata=instance_index)

    return sdata

def get_points(
    sdata: SpatialData,
    points_key: str = "transcripts",
    astype: str = "pandas",
    sync: bool = True,
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
    if points_key not in sdata.points.keys():
        raise ValueError(f"Points key {points_key} not found in sdata.points")

    if astype not in ["pandas", "dask", "geopandas"]:
        raise ValueError(
            f"astype must be one of ['dask', 'pandas', 'geopandas'], not {astype}"
        )

    points = sdata.points[points_key]

    # Sync points to instance_key
    if sync:
        _check_points_sync(sdata, points_key)
        instance_key = points.attrs["spatialdata_attrs"]["instance_key"]

        point_index = sdata.points[points_key][instance_key]
        valid_points = point_index != ""
        points = points[valid_points]

    if astype == "pandas":
        return points.compute()
    elif astype == "dask":
        return points
    elif astype == "geopandas":
        points = points.compute()
        return gpd.GeoDataFrame(
            points, geometry=gpd.points_from_xy(points.x, points.y), copy=True
        )
    
def get_shape(sdata: SpatialData, shape_key: str, sync: bool = True) -> gpd.GeoSeries:
    """Get a GeoSeries of Polygon objects from an SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    shape_key : str
        Name of shape column in sdata.shapes
    sync : bool
        Whether to retrieve shapes synced to cell shape. Default True.

    Returns
    -------
    GeoSeries
        GeoSeries of Polygon objects
    """
    instance_key = sdata.table.uns["spatialdata_attrs"]["instance_key"]

    # Make sure shape exists in sdata.shapes
    if shape_key not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_key} not found in sdata.shapes")

    if sync and shape_key != instance_key:
        _check_shape_sync(sdata, shape_key, instance_key)
        shape_index = sdata.shapes[shape_key][instance_key]
        valid_shapes = shape_index != ""
        return sdata.shapes[shape_key][valid_shapes].geometry

    return sdata.shapes[shape_key].geometry

def get_points_metadata(
    sdata: SpatialData,
    metadata_keys: Union[str, List[str]],
    points_key: str = "transcripts",
    astype="pandas",
):
    """Get points metadata.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    metadata_keys : str or list of str
        Key(s) for `sdata.points[points_key][key]` to use
    points_key : str, optional
        Key for `sdata.points` to use, by default "transcripts"
    astype : str, optional
        Whether to return a 'pandas' Series or 'dask' DataFrame, by default "pandas"

    Returns
    -------
    pd.DataFrame or dd.DataFrame
        Returns `sdata.points[points_key][metadata_keys]` as a `pd.DataFrame` or `dd.DataFrame`
    """
    if points_key not in sdata.points.keys():
        raise ValueError(f"Points key {points_key} not found in sdata.points")
    if astype not in ["pandas", "dask"]:
        raise ValueError(
            f"astype must be one of ['dask', 'pandas'], not {astype}"
        )
    if isinstance(metadata_keys, str):
        metadata_keys = [metadata_keys]
    for key in metadata_keys:
        if key not in sdata.points[points_key].columns:
            raise ValueError(f"Metadata key {key} not found in sdata.points[{points_key}]")

    metadata = sdata.points[points_key][metadata_keys]

    if astype == "pandas":
        return metadata.compute()
    elif astype == "dask":
        return metadata
    
def get_shape_metadata(
    sdata: SpatialData,
    metadata_keys: Union[str, List[str]],
    shape_key: str = "transcripts",
):
    """Get shape metadata.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    metadata_keys : str or list of str
        Key(s) for `sdata.shapes[shape_key][key]` to use
    shape_key : str
        Key for `sdata.shapes` to use, by default "transcripts"

    Returns
    -------
    pd.Dataframe
        Returns `sdata.shapes[shape_key][metadata_keys]` as a `pd.DataFrame`
    """
    if shape_key not in sdata.shapes.keys():
        raise ValueError(f"Shape key {shape_key} not found in sdata.shapes")
    if isinstance(metadata_keys, str):
        metadata_keys = [metadata_keys]
    for key in metadata_keys:
        if key not in sdata.shapes[shape_key].columns:
            raise ValueError(f"Metadata key {key} not found in sdata.shapes[{shape_key}]")

    return sdata.shapes[shape_key][metadata_keys]

def set_points_metadata(
    sdata: SpatialData,
    points_key: str,
    metadata: Union[List, pd.Series, pd.DataFrame],
    column_names: Optional[Union[str, List[str]]] = None,
):
    """Write metadata in SpatialData points element as column(s). Aligns metadata index to shape index.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    points_key : str
        Name of element in sdata.points
    metadata : pd.Series, pd.DataFrame
        Metadata to set for points. Index must be a (sub)set of points index.
    column_names : str or list of str, optional
        Name of column(s) to set. If None, use metadata column name(s), by default None
    """
    if points_key not in sdata.points.keys():
        raise ValueError(f"{points_key} not found in sdata.points")
    
    if isinstance(metadata, list):
        metadata = pd.Series(metadata, index=sdata.points[points_key].index)

    if isinstance(metadata, pd.Series):
        metadata = pd.DataFrame(metadata)

    if column_names is not None:
        if isinstance(column_names, str):
            column_names = [column_names]
        metadata = metadata.rename(columns={metadata.columns[0]: column_names[0]})
        
    sdata.points[points_key] = sdata.points[points_key].reset_index(drop=True)
    for name, series in metadata.iteritems():
        series = series.fillna("")
        metadata_series = dd.from_pandas(series, npartitions=sdata.points[points_key].npartitions).reset_index(drop=True)
        sdata.points[points_key][name] = metadata_series

def set_shape_metadata(
    sdata: SpatialData,
    shape_key: str,
    metadata: Union[List, pd.Series, pd.DataFrame],
    column_names: Optional[Union[str, List[str]]] = None,
):
    """Write metadata in SpatialData shapes element as column(s). Aligns metadata index to shape index.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    shape_key : str
        Name of element in sdata.shapes
    metadata : pd.Series, pd.DataFrame
        Metadata to set for shape. Index must be a (sub)set of shape index.
    column_names : str or list of str, optional
        Name of column(s) to set. If None, use metadata column name(s), by default None
    """
    if shape_key not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_key} not found in sdata.shapes")
    
    if isinstance(metadata, list):
        metadata = pd.Series(metadata, index=sdata.shapes[shape_key].index)

    if isinstance(metadata, pd.Series):
        metadata = pd.DataFrame(metadata)

    if column_names is not None:
        if isinstance(column_names, str):
            column_names = [column_names]
        metadata = metadata.rename(columns={metadata.columns[0]: column_names[0]})

    sdata.shapes[shape_key].loc[:, metadata.columns] = metadata.reindex(
        sdata.shapes[shape_key].index
    ).fillna("")

def _check_points_sync(sdata, points_key):
    """
    Check if points are synced to instance_key shape in a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to check.
    points_key : str
        The name of the points to check.

    Raises
    ------
    ValueError
        If the points are not synced to instance_key shape.
    """
    points = sdata.points[points_key]
    if points.attrs["spatialdata_attrs"]["instance_key"] not in points.columns:
        raise ValueError(
            f"Points {points_key} not synced to instance_key shape element. Run bento.io.format_sdata() to setup SpatialData object for bento-tools."
        )

def _check_shape_sync(sdata, shape_key, instance_key):
    """
    Check if a shape is synced to instance_key shape in a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to check.
    shape_key : str
        The name of the shape to check.
    instance_key : str
        The instance key of the shape to check.

    Raises
    ------
    ValueError
        If the shape is not synced to instance_key shape.
    """
    if (
        shape_key != instance_key
        and shape_key not in sdata.shapes[instance_key].columns
    ):
        raise ValueError(
            f"Shape {shape_key} not synced to instance_key shape element. Run bento.io.format_sdata() to setup SpatialData object for bento-tools."
        )
