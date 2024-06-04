# Geometric operations for SpatialData ShapeElements wrapping GeoPandas GeoDataFrames.
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from dask import dataframe as dd
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel


def filter_by_gene(
    sdata: SpatialData,
    min_count: int = 10,
    points_key: str = "transcripts",
    feature_key: str = "feature_name",
):
    """
    Filters out genes with low expression from the spatial data object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    threshold : int
        Minimum number of counts for a gene to be considered expressed.
        Keep genes where at least {threshold} molecules are detected in at least one cell.
    points_key : str
        key for points element that holds transcript coordinates
    feature_key : str
        Key for gene instances

    Returns
    -------
    sdata : SpatialData
        .points[points_key] is updated to remove genes with low expression.
        .table is updated to remove genes with low expression.
    """
    gene_filter = (sdata.table.X >= min_count).sum(axis=0) > 0
    filtered_table = sdata.table[:, gene_filter]

    filtered_genes = list(sdata.table.var_names.difference(filtered_table.var_names))
    points = get_points(sdata, points_key=points_key, astype="pandas", sync=False)
    points = points[~points[feature_key].isin(filtered_genes)]
    points[feature_key] = points[feature_key].cat.remove_unused_categories()

    transform = sdata[points_key].attrs
    points = PointsModel.parse(
        dd.from_pandas(points, npartitions=1), coordinates={"x": "x", "y": "y"}
    )
    points.attrs = transform
    sdata.points[points_key] = points

    try:
        del sdata.table
    except KeyError:
        pass
    sdata.table = TableModel.parse(filtered_table)

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
    sync : bool, optional
        Whether to set and retrieve points synced to instance_key shape. Default True.

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

    # Sync points to instance_key
    if sync:
        _sync_points(sdata, points_key)

    points = sdata.points[points_key]

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
        Whether to set and retrieve shapes synced to cell shape. Default True.

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
        _sync_shapes(sdata, shape_key, instance_key)
        shape_index = sdata.shapes[shape_key][instance_key]
        valid_shapes = shape_index != ""
        return sdata.shapes[shape_key][valid_shapes].geometry

    return sdata.shapes[shape_key].geometry


def get_points_metadata(
    sdata: SpatialData,
    metadata_keys: Union[List[str], str],
    points_key: str,
    astype: str = "pandas",
) -> Union[pd.DataFrame, dd.DataFrame]:
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
        raise ValueError(f"astype must be one of ['dask', 'pandas'], not {astype}")
    if isinstance(metadata_keys, str):
        metadata_keys = [metadata_keys]
    for key in metadata_keys:
        if key not in sdata.points[points_key].columns:
            raise ValueError(
                f"Metadata key {key} not found in sdata.points[{points_key}]"
            )

    metadata = sdata.points[points_key][metadata_keys]

    if astype == "pandas":
        return metadata.compute()
    elif astype == "dask":
        return metadata


def get_shape_metadata(
    sdata: SpatialData,
    metadata_keys: Union[List[str], str],
    shape_key: str,
) -> pd.DataFrame:
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
    pd.DataFrame
        Returns `sdata.shapes[shape_key][metadata_keys]` as a `pd.DataFrame`
    """
    if shape_key not in sdata.shapes.keys():
        raise ValueError(f"Shape key {shape_key} not found in sdata.shapes")
    if isinstance(metadata_keys, str):
        metadata_keys = [metadata_keys]
    for key in metadata_keys:
        if key not in sdata.shapes[shape_key].columns:
            raise ValueError(
                f"Metadata key {key} not found in sdata.shapes[{shape_key}]"
            )

    return sdata.shapes[shape_key][metadata_keys]


def set_points_metadata(
    sdata: SpatialData,
    points_key: str,
    metadata: Union[List, pd.Series, pd.DataFrame, np.ndarray],
    columns: Union[List[str], str],
) -> None:
    """Write metadata in SpatialData points element as column(s). Aligns metadata index to shape index if present.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    points_key : str
        Name of element in sdata.points
    metadata : pd.Series, pd.DataFrame, np.ndarray
        Metadata to set for points. Assumes input is already aligned to points index.
    column_names : str or list of str, optional
        Name of column(s) to set. If None, use metadata column name(s), by default None
    """
    if points_key not in sdata.points.keys():
        raise ValueError(f"{points_key} not found in sdata.points")

    columns = [columns] if isinstance(columns, str) else columns

    # metadata = pd.DataFrame(np.array(metadata), columns=columns)
    metadata = np.array(metadata)

    transform = sdata.points[points_key].attrs
    points = sdata.points[points_key].compute()
    points.loc[:, columns] = metadata
    points = PointsModel.parse(
        dd.from_pandas(points, npartitions=1), coordinates={"x": "x", "y": "y"}
    )
    points.attrs = transform
    sdata.points[points_key] = points

    # sdata.points[points_key] = sdata.points[points_key].reset_index(drop=True)
    # for name, series in metadata.items():
    #     series = series.fillna("") if series.dtype == object else series
    #     series = dd.from_pandas(
    #         series, npartitions=sdata.points[points_key].npartitions
    #     ).reset_index(drop=True)
    #     sdata.points[points_key] = sdata.points[points_key].assign(**{name: series})


def set_shape_metadata(
    sdata: SpatialData,
    shape_key: str,
    metadata: Union[List, pd.Series, pd.DataFrame, np.ndarray],
    column_names: Union[List[str], str] = None,
) -> None:
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

    shape_index = sdata.shapes[shape_key].index

    if isinstance(metadata, list):
        metadata = pd.Series(metadata, index=shape_index)

    if isinstance(metadata, pd.Series) or isinstance(metadata, np.ndarray):
        metadata = pd.DataFrame(metadata)

    if column_names is not None:
        metadata.columns = (
            [column_names] if isinstance(column_names, str) else column_names
        )

    # Fill missing values in string columns with empty string
    str_columns = metadata.select_dtypes(include="object", exclude="number").columns
    metadata[str_columns] = metadata[str_columns].fillna("")

    # Fill missing values in categorical columns with empty string
    cat_columns = metadata.select_dtypes(include="category").columns
    for col in cat_columns:
        if "" not in metadata[col].cat.categories:
            metadata[col] = metadata[col].cat.add_categories([""]).fillna("")

    sdata.shapes[shape_key].loc[:, metadata.columns] = metadata.reindex(
        shape_index
    ).fillna("")


def _sync_points(sdata, points_key):
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
    points = sdata.points[points_key].compute()
    instance_key = get_instance_key(sdata)

    # Only keep points within instance_key shape
    cells = set(sdata.shapes[instance_key].index)
    transform = sdata.points[points_key].attrs
    points_valid = points[
        points[instance_key].isin(cells)
    ]  # TODO why doesnt this grab the right cells
    # Set points back to SpatialData object
    points_valid = PointsModel.parse(
        dd.from_pandas(points_valid, npartitions=1),
        coordinates={"x": "x", "y": "y"},
    )
    points_valid.attrs = transform
    sdata.points[points_key] = points_valid


def _sync_shapes(sdata, shape_key, instance_key):
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
    shapes = sdata.shapes[shape_key]
    instance_shapes = sdata.shapes[instance_key]
    if shape_key == instance_key:
        return

    # Only keep shapes within instance_key shape
    cells = set(instance_shapes.index)
    shapes = shapes[shapes[instance_key].isin(cells)]

    # Set shapes back to SpatialData object
    transform = sdata.shapes[shape_key].attrs
    shapes_valid = ShapesModel.parse(shapes)
    shapes_valid.attrs = transform
    sdata.shapes[shape_key] = shapes_valid


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
        raise KeyError(
            "Instance key attribute not found in spatialdata object. Run bento.io.prep() to setup SpatialData object for bento-tools."
        )


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
