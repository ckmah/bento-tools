import geopandas as gpd

from .._utils import track


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


@track
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


def get_layers(data, layers, min_count=None):
    """Get values of layers reformatted as a long-form dataframe.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    layers : list of str
        all values must to be keys in data.layers
    min_count : int, default None
        minimum number of molecules (count) required to include in output
    Returns
    -------
    DataFrame
        rows are samples indexed as (cell, gene) and columns are features
    """
    sample_index = (
        data.to_df()
        .reset_index()
        .melt(id_vars="cell")
        .dropna()
        .set_index(["cell", "gene"])
    )

    if min_count:
        sample_index = sample_index[sample_index["value"] >= min_count].drop(
            "value", axis=1
        )

    for layer in layers:
        values = (
            data.to_df(layer)
            .reset_index()
            .melt(id_vars="cell")
            .set_index(["cell", "gene"])
        )
        values.columns = [layer]
        sample_index = sample_index.join(values)

    return sample_index[layers]
