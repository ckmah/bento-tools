from typing import Dict, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import dask.dataframe as dd
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel

from .._utils import sync_points

def sindex_points(
    sdata: SpatialData,  points_key: str = "transcripts", shape_names: List[str] = ["cell_boundaries"]
):
    """Index points to shapes and add as columns to `data.points[points_key]`.

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
    transform = sdata.points[points_key].attrs
    points_gpd = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.x, points.y), copy=True)

    # Index points to shapes
    for shape in shape_gpds:
        shape_gpd = shape_gpds[shape]
        sjoined_points = gpd.sjoin(points_gpd, shape_gpd, how="left", predicate="intersects")
        sjoined_points = sjoined_points[~sjoined_points.index.duplicated(keep='last')]
        sjoined_points.loc[sjoined_points["index_right"].isna(), "index_right"] = ""
        points[shape.split('_')[0]] = sjoined_points["index_right"].astype('category')

    sdata.points[points_key] = PointsModel.parse(points, coordinates={'x': 'x', 'y': 'y', 'z': 'z'})
    sdata.points[points_key].attrs = transform

    return sdata

def sjoin_shapes(sdata: SpatialData, shape_names: List[str]):
    """Adds polygon columns sdata.shapes['cell_boundaries'][shape_name] for point feature analysis

    Parameters
    ----------
    sdata : SpatialData
        Spatially formatted SpatialData
    shape_names : str or list of str
        Names of the shapes to add.

    Returns
    -------
    sdata : SpatialData
        .shapes['cell_boundaries'][shape_name]
    """

    # Cast to list if not already
    if isinstance(shape_names, str):
        shape_names = [shape_names]

    # Check if shapes are already indexed to cell_boundaries
    shapes_found = set(shape_names).intersection(set(sdata.shapes["cell_boundaries"].columns.tolist()))
    if shapes_found == set(shape_names):
        return
    
    # Remove keys for shapes that are already indexed
    shape_names = list(set(shape_names).difference(set(sdata.shapes["cell_boundaries"].columns.tolist())))
    sjoined_shapes = sdata.shapes["cell_boundaries"]
    transform = sdata.shapes["cell_boundaries"].attrs
    for shape in shape_names:
        shape_gpd = gpd.GeoDataFrame(geometry=sdata.shapes[shape]['geometry'])
        sjoined_shapes = sjoined_shapes.sjoin(shape_gpd, how='left', predicate='contains')
        sjoined_shapes.rename(columns={"index_right": shape}, inplace=True)
        for index in list(sjoined_shapes.index):
            shape_id = sjoined_shapes.at[index, shape]
            try:
                sjoined_shapes.at[index, shape] = sdata.shapes[shape].loc[shape_id]["geometry"]
            except KeyError:
                pass
            
    # Add to sdata.shapes
    sdata.shapes["cell_boundaries"] = ShapesModel.parse(sjoined_shapes)
    sdata.shapes["cell_boundaries"].attrs = transform

    return sdata

def get_shape(sdata: SpatialData, shape_name: str, sync: bool = False) -> gpd.GeoSeries:
    """Get a GeoSeries of Polygon objects from an SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    shape_name : str
        Name of shape column in sdata.shapes
    sync : bool
        Whether to sync the shape to cell_boundaries. Default False.
        
    Returns
    -------
    GeoSeries
        GeoSeries of Polygon objects
    """
    if sync and shape_name != "cell_boundaries":
        if shape_name not in sdata.shapes["cell_boundaries"].columns:
            raise ValueError(f"Shape {shape_name} not synced to cell_boundaries. Run sjoin_shapes first.")
        return sdata.shapes["cell_boundaries"][shape_name]
        
    if shape_name not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_name} not found in sdata.shapes")
        
    return sdata.shapes[shape_name].geometry

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
        raise ValueError(f"astype must be one of ['dask', 'pandas', 'geopandas'], not {astype}")

    points = sync_points(sdata).points[points_key]
    
    if astype == "pandas":
        return points.compute()
    elif astype == "dask":
        return points
    elif astype == "geopandas":
        points = points.compute()
        return gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.x, points.y), copy=True)
    
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