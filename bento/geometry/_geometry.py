import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon


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

    points = get_points(data, points_key, asgeo=True)

    point_sindex = dict()
    for col in shape_names:
        shp_gdf = gpd.GeoDataFrame(geometry=adata.obs[col])
        shp_name = "_".join(str(col).split("_")[:-1])

        # Remove column if it already exists
        points = points.drop(columns=shp_name, errors="ignore")

        # remove multiple polygons assigned to same point
        sindex = gpd.sjoin(points.reset_index(), shp_gdf, how="left", op="intersects")
        sindex = (
            sindex.drop_duplicates(subset="index", keep="first")
            .set_index("index")["index_right"]
            .notna()
        )

        point_sindex[shp_name] = sindex

    point_sindex = pd.DataFrame.from_dict(point_sindex)

    # Add new columns to points
    adata.uns[points_key] = points.join(point_sindex)

    return adata if copy else None


def crop(data, xlims=None, ylims=None):
    """Returns a view of data within specified coordinates."""
    if len(xlims) < 1 and len(xlims) > 2:
        return ValueError("Invalid xlims")

    if len(ylims) < 1 and len(ylims) > 2:
        return ValueError("Invalid ylims")

    xmin, xmax = xlims[0], xlims[1]
    ymin, ymax = ylims[0], ylims[1]
    box = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    in_crop = get_shape(data, "cell_shape").within(box)

    adata = data[in_crop, :]
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
    points = data.uns[key]

    # Subset for cells
    cells = data.obs_names.tolist()
    in_cells = points["cell"].isin(cells)

    # Subset for genes
    in_genes = [True] * points.shape[0]
    if key == "points":
        genes = data.var_names.tolist()
        in_genes = points["gene"].isin(genes)

    # Subset for genes
    points = points.loc[in_cells & in_genes]

    # Remove unused categories for categorical columns
    for col in points.columns:
        if points[col].dtype == "category":
            points[col].cat.remove_unused_categories(inplace=True)

    # Cast to GeoDataFrame
    if asgeo:
        points = gpd.GeoDataFrame(
            points, geometry=gpd.points_from_xy(points.x, points.y)
        )

    return points


def set_points(data, copy=False):
    """Set points for the given `AnnData` object, data. Call this setter
    to keep the points DataFrame in sync.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    copy : bool
            Return a copy of `data` instead of writing to data, by default False.
    Returns
    -------
    _type_
        _description_
    """
    adata = data.copy() if copy else data
    points = get_points(adata)
    adata.uns["points"] = points
    return adata if copy else None
