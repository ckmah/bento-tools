from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import dask.dataframe as dd
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from shapely.geometry import Polygon


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
    query_shapes = {}
    for shape in shape_names:
        query_shapes[shape] = gpd.GeoDataFrame(geometry=sdata.shapes[shape].geometry)

    # Grab points as GeoDataFrame
    points = get_points(sdata, points_key, astype="geopandas", sync=False)
    attrs = sdata.points[points_key].attrs

    # Index points to shapes
    for shape_name, shape in query_shapes.items():
        shape = query_shapes[shape_name]
        shape.index.name = None
        shape.index = shape.index.astype(str)
        points = points.sjoin(shape, how="left", predicate="intersects")
        points = points[~points.index.duplicated(keep="last")]
        points.loc[points["index_right"].isna(), "index_right"] = ""
        points.rename(columns={"index_right": shape_name}, inplace=True)

    points = pd.DataFrame(points.drop(columns="geometry"))

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
    shape_names = (
        set(shape_names) - set(sdata.shapes[instance_key].columns) - set(instance_key)
    )

    if len(shape_names) == 0:
        return sdata

    parent_shape = sdata.shapes[instance_key]
    attrs = sdata.shapes[instance_key].attrs

    # sjoin shapes to instance_key shape
    for shape in shape_names:
        child_shape = gpd.GeoDataFrame(geometry=sdata.shapes[shape]["geometry"])
        parent_shape = parent_shape.sjoin(child_shape, how="left", predicate="contains")
        parent_shape = parent_shape[~parent_shape.index.duplicated(keep="last")]
        parent_shape.loc[parent_shape["index_right"].isna(), "index_right"] = ""
        parent_shape = parent_shape.astype({"index_right": "category"})

        # save shape index as column in instance_key shape
        # id_col = f"{shape}_id"
        parent_shape.rename(columns={"index_right": shape}, inplace=True)

        # Add instance_key shape index to shape
        parent_shape.index.name = "parent_index"
        instance_index = parent_shape.reset_index().set_index(shape)["parent_index"]
        instance_index.name = instance_key # index name = shape, column name = instance_key
        instance_index.index.name = None
        instance_index = instance_index[instance_index.index != ""]

        set_shape_metadata(sdata, shape_name=shape, metadata=instance_index)

    # Add to sdata.shapes
    sdata.shapes[instance_key] = ShapesModel.parse(parent_shape)
    sdata.shapes[instance_key].attrs = attrs

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
            f"Shape {shape_name} not synced to instance_key shape element. Run bento.io.format_sdata() to setup SpatialData object for bento-tools."
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
        Metadata to set for shape. Index must be a (sub)set of shape index.
    """
    if shape_name not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_name} not found in sdata.shapes")

    # Set metadata as columns in sdata.shape[shape_name]
    if isinstance(metadata, pd.Series):
        metadata = pd.DataFrame(metadata)

    sdata.shapes[shape_name].loc[:, metadata.columns] = metadata.reindex(
        sdata.shapes[shape_name].index
    )
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
    if astype not in ["pandas", "dask", "geopandas"]:
        raise ValueError(
            f"astype must be one of ['dask', 'pandas', 'geopandas'], not {astype}"
        )

    points = sdata.points[points_key]

    # Sync points to instance_key
    if sync:
        check_points_sync(sdata, points_key)

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
    sdata: SpatialData,
    metadata_keys: Union[str, List[str]],
    points_key: str = "transcripts",
    astype="pandas",
):
    """Get points metadata synced to SpatialData object.

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
    Series
        Returns `data.uns[key][metadata_key]` as a `Series`
    """
    metadata = sdata.points[points_key][metadata_keys]

    if astype == "pandas":
        return metadata.compute()
    elif astype == "dask":
        return metadata


def sync_points(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    feature_key: str = "",
) -> SpatialData:
    """
    Sync existing point sets and associated metadata with sdata.table.obs_names and sdata.table.var_names

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    points_key : str, optional
        Key for points DataFrame in `sdata.points`, by default "transcripts"
    instance_key : str, optional
        Key for the shape that will be used as the instance for all indexing. Usually the cell shape.
    feature_key : str, optional
        Key for the feature corresponding to table var names. Usually the gene name.
    """

    # Iterate over point sets
    for point_key in sdata.points:
        points = get_points(sdata, point_key, astype="dask")

        # Subset for cells
        cells = sdata.table.obs_names.tolist()
        in_cells = points[instance_key].isin(cells)

        # Subset for genes
        in_genes = [True] * points.shape[0]
        if feature_key and feature_key in points.columns:
            genes = sdata.table.var_names.tolist()
            in_genes = points[feature_key].isin(genes)

        # Combine boolean masks
        valid_mask = (in_cells & in_genes).values

        # Sync points using mask
        points = points.loc[valid_mask]

        # Remove unused categories for categorical columns
        for col in points.columns:
            if points[col].dtype == "category":
                points[col].cat.remove_unused_categories(inplace=True)

        # Update points in sdata
        attrs = sdata.points[points_key].attrs
        sdata.points[points_key] = PointsModel.parse(
            points.reset_index(drop=True), coordinates={"x": "x", "y": "y"}
        )
        sdata.points[points_key].attrs = attrs

        return sdata


def check_points_sync(sdata, points_key):
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
    if points.attrs["instance_key"] not in points.columns:
        raise ValueError(
            f"Points {points_key} not synced to instance_key shape element. Run bento.io.format_sdata() to setup SpatialData object for bento-tools."
        )
