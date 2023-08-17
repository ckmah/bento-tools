from typing import Dict, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from spatialdata._core.spatialdata import SpatialData

from .._utils import sync_points

def sindex_points(
    sdata: SpatialData, points_key: str, shape_names: List[str]
) -> SpatialData:
    """Index points to shapes and add as columns to `data.points[points_key]`.

    Parameters
    ----------
    data : SpatialData
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
        .points[points_key]: Updated points DataFrame with boolean column for each shape
    """

    if isinstance(shape_names, str):
        shape_names = [shape_names]

    meta_dict = {
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
        "feature_name": "category",
        "cell_id": np.int32,
        "transcript_id": np.uint64,
        "z_location": np.float32,
        "overlaps_nucleus": np.uint8,
        "qv": np.float32,
    }
    shape_gpds = {}
    for shape in shape_names:
        meta_dict[shape.split('_')[0]] = "str"
        shape_gpds[shape] = sdata.shapes[shape]

    def sindex(df, shape_gpds):
        points_gpd = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), copy=True)
        for shape in shape_gpds:
            shape_gpd = shape_gpds[shape]
            sjoined_points = gpd.sjoin(points_gpd, shape_gpd, how="left", predicate="intersects")
            sjoined_points = sjoined_points[~sjoined_points.index.duplicated(keep='last')]
            sjoined_points.loc[sjoined_points["index_right"].isna(), "index_right"] = "None"
            df[shape.split('_')[0]] = sjoined_points["index_right"].astype(str)
        return df
    
    sdata.points[points_key] = sdata.points[points_key].map_partitions(sindex, shape_gpds, meta=meta_dict)
    
    return sdata

def sjoin_shapes(sdata: SpatialData, shape_names: List[str],):
    """Adds polygon columns sdata.shapes['cell_boundaries'][shape_name] for point feature analysis

        Parameters
        ----------
        data : SpatialData
            Spatially formatted SpatialData
        shape_names : str or list of str
            Names of the shapes to add.

        Returns
        -------
    sdata : spatialdata.SpatialData
            .shapes['cell_boundaries'][shape_name]
    """

    # Cast to list if not already
    if isinstance(shape_names, str):
        shape_names = [shape_names]

    shapes_found = set(shape_names).intersection(set(sdata.shapes["cell_boundaries"].columns.tolist()))
    if shapes_found == set(shape_names):
        return sdata
    
    shape_names = list(set(shape_names).difference(set(sdata.shapes["cell_boundaries"].columns.tolist())))
    sjoined_shapes = sdata.shapes["cell_boundaries"].copy()
    for shape in shape_names:
        sjoined_shapes = sjoined_shapes.sjoin(sdata.shapes[shape], how='left', predicate='contains')
        sjoined_shapes.rename(columns={"index_right": shape}, inplace=True)
        for index in list(sjoined_shapes.index):
            shape_id = sjoined_shapes.at[index, shape]
            try:
                sjoined_shapes.at[index, shape] = sdata.shapes[shape].loc[shape_id][shape]
            except:
                pass
            
    sdata.shapes["cell_boundaries"] = sjoined_shapes
    return sdata

def get_shape(sdata: SpatialData, shape_name: str) -> gpd.GeoSeries:
    """Get a GeoSeries of Polygon objects from an SpatialData object.

    Parameters
    ----------
    adata : SpatialData
        Spatial formatted SpatialData object
    shape_name : str
        Name of shape column in sdata.shapes

    Returns
    -------
    GeoSeries
        GeoSeries of Polygon objects
    """
    if shape_name not in sdata.shapes.keys():
        raise ValueError(f"Shape {shape_name} not found in sdata.shapes")
        
    sdata.shapes[shape_name].geometry.name = shape_name
    return sdata.shapes[shape_name].geometry

def get_points(
    sdata: SpatialData, key: str = "transcripts", asgeo: bool = False
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Get points DataFrame synced to AnnData object.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object
    key : str, optional
        Key for `data.points` to use, by default "transcripts"
    asgeo : bool, optional
        Cast as GeoDataFrame using columns x and y for geometry, by default False

    Returns
    -------
    DataFrame or GeoDataFrame
        Returns `data.points[key]` as a `[Geo]DataFrame`
    """
    points = sync_points(sdata).points[key]
    
    if asgeo:
        # Cast to Dask GeoDataFrame
        def as_geo(df):
            return gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.x, df.y), copy=True
            )
        points = points.map_partitions(as_geo)
        
    return points
    