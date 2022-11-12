import geopandas as gpd

from shapely import wkt
from shapely.geometry import Polygon


def crop(data, xlims=None, ylims=None):
    """Returns a view of data within specified coordinates.
    """
    if len(xlims) < 1 and len(xlims) > 2:
        return ValueError("Invalid xlims")
    
    if len(ylims) < 1 and len(ylims) > 2:
        return ValueError("Invalid ylims")
    
    xmin, xmax = xlims[0], xlims[1]
    ymin, ymax = ylims[0], ylims[1]
    box = Polygon([[xmin, ymin],[xmin, ymax],[xmax, ymax],[xmax, ymin]])
    in_crop = get_shape(data, "cell_shape").within(box)
    
    adata = data[in_crop,:]
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


def get_points(data, asgeo=False):
    """Get points DataFrame.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    asgeo : bool, optional
        Cast as GeoDataFrame using columns x and y for geometry, by default False

    Returns
    -------
    DataFrame or GeoDataFrame
        Returns `data.uns['points']` as a `[Geo]DataFrame`
    """
    points = data.uns["points"]

    cells = data.obs_names.tolist()
    genes = data.var_names.tolist()

    # Subset for cells
    in_cells = points["cell"].isin(cells)
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
