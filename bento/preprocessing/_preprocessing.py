import dask.dataframe as dd
import dask_geopandas
import geopandas as gpd
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

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


@track
def remove_extracellular(data, copy=False):
    """
    Remove extracellular points.
    """
    adata = data.copy() if copy else data
    points = get_points(adata)

    points = points.set_index("cell").join(data.obs["cell_shape"]).reset_index()
    points["cell"] = points["cell"].astype("category").cat.as_ordered()

    def _filter_points(pts):
        pts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.x, pts.y))

        cell_shape = pts["cell_shape"].values[0]
        in_cell = pts.within(cell_shape)
        pts = pts[in_cell]

        return pts

    ngroups = points.groupby("cell").ngroups

    if ngroups > 100:
        npartitions = min(1000, ngroups)

        with ProgressBar():

            points = (
                dask_geopandas.from_geopandas(
                    points,
                    npartitions=npartitions,
                )
                .groupby("cell", observed=True)
                .apply(_filter_points, meta=dict(zip(points.columns, points.dtypes)))
                .compute()
            )
    else:
        tqdm.pandas()
        points.groupby("cell").progress_apply(_filter_points)

    points = (
        points.drop("cell_shape", axis=1)
        .drop("geometry", axis=1)
        .reset_index(drop=True)
    )

    cat_vars = ["cell", "gene", "nucleus"]
    points[cat_vars] = points[cat_vars].astype("category")

    adata.uns["points"] = points

    return adata if copy else None


@track
def subsample(data, frac=0.2, copy=True):
    """
    Subsample transcripts, preserving the number of cells and genes detected per cell.
    """
    adata = data.copy() if copy else data
    points = get_points(data)

    with ProgressBar():
        sampled_pts = (
            dd.from_pandas(points, chunksize=1000000)
            .groupby(["cell", "gene"])
            .apply(
                lambda df: df.sample(frac=frac),
                meta=dict(zip(points.columns, points.dtypes)),
            )
            .reset_index(drop=True)
            .compute()
        )

    X = sampled_pts[["cell", "gene"]].pivot_table(
        index="cell", columns="gene", aggfunc=len, fill_value=0
    )

    adata.uns["points"] = sampled_pts

    adata = adata[X.index, X.columns]
    adata.X = X

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
            data.to_df(layer).reset_index().melt(id_vars="cell").set_index(["cell", "gene"])
        )
        values.columns = [layer]
        sample_index = sample_index.join(values)

    return sample_index[layers]
