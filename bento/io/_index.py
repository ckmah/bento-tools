from typing import List

import pandas as pd
import geopandas as gpd
from spatialdata._core.spatialdata import SpatialData

from .._utils import (
    get_points,
    set_points_metadata,
    set_shape_metadata,
)


def _sjoin_points(
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
    points.index.name = "pt_index"

    # Index points to shapes
    indexed_points = {}
    for shape_key, shape in query_shapes.items():
        shape = query_shapes[shape_key]
        shape.index.name = None  # Forces sjoin to name index "index_right"
        shape.index = shape.index.astype(str)

        indexed_points[shape_key] = (
            points.sjoin(shape, how="left", predicate="intersects")
            .reset_index()
            .drop_duplicates(subset="pt_index")["index_right"]
            .fillna("")
            .values.flatten()
        )

    index_points = pd.DataFrame(indexed_points)
    set_points_metadata(
        sdata, points_key, index_points, columns=list(indexed_points.keys())
    )

    return sdata


def _sjoin_shapes(sdata: SpatialData, instance_key: str, shape_keys: List[str]):
    """Adds polygon indexes to sdata.shapes[instance_key][shape_key] for point feature analysis.

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

    parent_shape = gpd.GeoDataFrame(sdata.shapes[instance_key])

    # sjoin shapes to instance_key shape
    for shape_key in shape_keys:
        child_shape = sdata.shapes[shape_key]
        # Hack for polygons that are 99% contained in parent shape or have shared boundaries
        child_shape = gpd.GeoDataFrame(geometry=child_shape.buffer(-10e-6))

        # Map child shape index to parent shape and process the result
        parent_shape = (
            parent_shape.sjoin(child_shape, how="left", predicate="covers")
            .reset_index()
            .drop_duplicates(subset="index", keep="last")
            .set_index("index")
            .assign(
                index_right=lambda df: df.loc[
                    ~df["index_right"].duplicated(keep="first"), "index_right"
                ]
                .fillna("")
                .astype("category")
            )
            .rename(columns={"index_right": shape_key})
        )
        parent_shape[shape_key] = parent_shape[shape_key].fillna("")

        # Save shape index as column in instance_key shape
        set_shape_metadata(
            sdata, shape_key=instance_key, metadata=parent_shape[shape_key]
        )

        # Add instance_key shape index to shape
        instance_index = (
            parent_shape.drop_duplicates(subset=shape_key)
            .reset_index()
            .set_index(shape_key)["index"]
            .rename(instance_key)
            .loc[lambda s: s.index != ""]
        )

        set_shape_metadata(sdata, shape_key=shape_key, metadata=instance_index)

    return sdata
