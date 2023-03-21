import re
from typing import Dict, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
from anndata import AnnData
from scipy.sparse import coo_matrix
from shapely import wkt
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from .._utils import sync


def count_points(
    data: AnnData, shape_names: List[str], copy: bool = False
) -> Optional[AnnData]:
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


def sindex_points(
    data: AnnData, points_key: str, shape_names: List[str], copy: bool = False
) -> Optional[AnnData]:
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


def crop(
    data: AnnData,
    xlims: Tuple[int],
    ylims: Tuple[int],
    copy: bool = True,
) -> Optional[AnnData]:
    """Returns a view of data within specified coordinates.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    xlims : list, optional
        Upper and lower x limits, by default None
    ylims : list, optional
        Upper and lower y limits, by default None
    copy : bool, optional
        Whether to return a copy the AnnData object. Default True.

    Returns
    -------
    AnnData
        AnnData object with data cropped to specified coordinates
    """
    adata = data.copy() if copy else data

    if len(xlims) < 1 and len(xlims) > 2:
        return ValueError("Invalid xlims")

    if len(ylims) < 1 and len(ylims) > 2:
        return ValueError("Invalid ylims")

    xmin, xmax = xlims[0], xlims[1]
    ymin, ymax = ylims[0], ylims[1]
    box = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    in_crop = get_shape(data, "cell_shape").within(box)

    adata = data[in_crop, :]
    sync(adata, copy=False)

    return adata if copy else None


def get_shape(adata: AnnData, shape_name: str) -> gpd.GeoSeries:
    """Get a GeoSeries of Polygon objects from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Spatial formatted AnnData object
    shape_name : str
        Name of shape column in adata.obs

    Returns
    -------
    GeoSeries
        GeoSeries of Polygon objects
    """
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


def rename_shapes(
    data: AnnData,
    mapping: Dict[str, str],
    points_key: Optional[Union[List[str], None]] = None,
    points_encoding: Union[List[Literal["label", "onehot"]], None] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """Rename shape columns in adata.obs and points columns in adata.uns.

    Parameters
    ----------
    adata : AnnData
        Spatial formatted AnnData object
    mapping : Dict[str, str]
        Mapping of old shape names to new shape names
    points_key : list of str, optional
        List of keys for points DataFrame in `adata.uns`, by default None
    points_encoding : {"label", "onehot"}, optional
        Encoding type for each specified points
    copy : bool, optional
        Whether to return a copy of the AnnData object. Default False.

    Returns
    -------
    AnnData
        .obs: Updated shape column names
        .uns[points_key]: Updated points shape(s) columns according to encoding type
    """
    adata = data.copy() if copy else data
    adata.obs.rename(columns=mapping, inplace=True)

    # Map point columns
    if points_key:
        # Get mapping for points column names
        prefix_map = {
            _get_shape_prefix(shape_name): _get_shape_prefix(new_name)
            for shape_name, new_name in mapping.items()
        }
        # Get mapping for label encoding
        label_map = {
            re.sub(r"\D", "", shape_name): re.sub(r"\D", "", new_name)
            for shape_name, new_name in prefix_map.items()
        }

        for p_key, p_encoding in zip(points_key, points_encoding):
            if p_encoding == "label":
                # Point column name with label encoding
                col = re.sub(r"\d", "", list(prefix_map.keys())[0])
                adata.uns[p_key][col] = adata.uns[p_key][col].astype(str).map(label_map)

            elif p_encoding == "onehot":
                # Remap column names
                adata.uns[p_key].rename(columns=prefix_map, inplace=True)

    return adata if copy else None


def _get_shape_prefix(shape_name):
    """Get prefix of shape name."""
    return "_".join(shape_name.split("_")[:-1])


def get_points(
    data: AnnData, key: str = "points", asgeo: bool = False
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Get points DataFrame synced to AnnData object.

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


def get_points_metadata(
    data: AnnData,
    metadata_key: str,
    points_key: str = "points",
):
    """Get points metadata synced to AnnData object.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    metadata_key : str
        Key for `data.uns[key]` to use
    key : str, optional
        Key for `data.uns` to use, by default "points"

    Returns
    -------
    Series
        Returns `data.uns[key][metadata_key]` as a `Series`
    """
    metadata = sync(data, copy=True).uns[metadata_key]
    return metadata
