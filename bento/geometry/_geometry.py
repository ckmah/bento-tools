import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm

from .._utils import sync

# Write a function to quantify the number of points in a given shape by summing boolean column in points. Add as a layer dimensions cell by gene in adata.layers.
def count_points(data, shape_names, copy=False):
    """Count points in shapes and add as layers to `data`. Expects points to already be indexed to shapes.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    shape_names : str, list
        List of shape names to index points to
    copy : bool, optional
        Whether to return a copy the AnnData object. Default False.
    Returns
    -------
    AnnData
        .layers: Updated layers with count of points in each shape
    """
    adata = data.copy() if copy else data

    if isinstance(shape_names, str):
        shape_names = [shape_names]

    points = get_points(data, asgeo=True)

    if shape_names[0].endswith("_shape"):
        shape_prefixes = [
            "_".join(shp_name.split("_shape")[:-1]) for shp_name in shape_names
        ]
    else:
        shape_prefixes = shape_names

    shape_counts = points.groupby(["cell", "gene"], observed=True)[shape_prefixes].sum()

    for shape in shape_counts.columns:
        pos_counts = shape_counts[shape]
        pos_counts = pos_counts[pos_counts > 0]
        values = pos_counts

        row = adata.obs_names.get_indexer(pos_counts.index.get_level_values("cell"))
        col = adata.var_names.get_indexer(pos_counts.index.get_level_values("gene"))
        adata.layers[f"{shape}"] = coo_matrix((values, (row, col)))

    return adata if copy else None


def sindex_points(data, points_key, shape_names, copy=False):
    """Index points to shapes and add as columns to `data.uns[points_key]`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    points_key : str
        Key for points DataFrame in `data.uns`
    shape_names : str, list
        List of shape names to index points to
    copy : bool, optional
        Whether to return a copy the AnnData object. Default False.
    Returns
    -------
    AnnData
        .uns[points_key]: Updated points DataFrame with boolean column for each shape
    """
    adata = data.copy() if copy else data

    if isinstance(shape_names, str):
        shape_names = [shape_names]

    points = get_points(data, points_key, asgeo=True).sort_values("cell")
    points = points.drop(
        columns=shape_names, errors="ignore"
    )  # Drop columns to overwrite
    points_grouped = points.groupby("cell", observed=True)
    cells = list(points_grouped.groups.keys())
    point_sindex = []

    # Iterate over cells and index points to shapes
    for cell in tqdm(cells, leave=False):
        pt_group = points_grouped.get_group(cell)

        # Get shapes to index in current cell
        cur_shapes = gpd.GeoDataFrame(geometry=data.obs.loc[cell, shape_names].T)
        cur_sindex = (
            pt_group.reset_index()
            .sjoin(cur_shapes, how="left", op="intersects")
            .drop_duplicates(subset="index", keep="first")
            .sort_index()
            .reset_index()["index_right"]
            .astype(str)
        )
        point_sindex.append(cur_sindex)

    # TODO: concat is hella slow
    point_sindex = (
        pd.concat(point_sindex, ignore_index=True).str.get_dummies() == 1
    ).fillna(False)
    point_sindex.columns = [col.replace("_shape", "") for col in point_sindex.columns]

    # Add new columns to points
    points[point_sindex.columns] = point_sindex.values
    adata.uns[points_key] = points

    return adata if copy else None


def crop(data, xlims=None, ylims=None):
    """Returns a view of data within specified coordinates.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    xlims : list, optional
        Upper and lower x limits, by default None
    ylims : list, optional
        Upper and lower y limits, by default None
    """
    if len(xlims) < 1 and len(xlims) > 2:
        return ValueError("Invalid xlims")

    if len(ylims) < 1 and len(ylims) > 2:
        return ValueError("Invalid ylims")

    xmin, xmax = xlims[0], xlims[1]
    ymin, ymax = ylims[0], ylims[1]
    box = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    in_crop = get_shape(data, "cell_shape").within(box)

    adata = data[in_crop, :]
    sync(adata)

    return adata


def get_shape(adata, shape_name):
    """Get a GeoSeries of Polygon objects from an AnnData object."""
    if shape_name not in adata.obs.columns:
        raise ValueError(f"Shape {shape_name} not found in adata.obs.")

    if adata.obs[shape_name].astype(str).str.startswith("POLYGON").any():
        return gpd.GeoSeries(
            adata.obs[shape_name]
            .astype(str)
            .apply(lambda val: wkt.loads(val) if val != "None" else None)
        )

    else:
        return gpd.GeoSeries(adata.obs[shape_name])


def get_points(data, key="points", asgeo=False):
    """Get points DataFrame.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    key : str, optional
        Key for `data.uns` to use, by default "points"
    asgeo : bool, optional
        Cast as GeoDataFrame using columns x and y for geometry, by default False

    Returns
    -------
    DataFrame or GeoDataFrame
        Returns `data.uns[key]` as a `[Geo]DataFrame`
    """
    points = sync(data, copy=True).uns[key]

    # Cast to GeoDataFrame
    if asgeo:
        points = gpd.GeoDataFrame(
            points, geometry=gpd.points_from_xy(points.x, points.y)
        )

    return points
