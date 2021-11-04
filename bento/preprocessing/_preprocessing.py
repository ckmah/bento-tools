from .._utils import track

import pandas as pd
import numpy as np
import geopandas as gpd
import dask_geopandas
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder

def get_points(data, cells=None, genes=None):

    points = data.uns["points"]

    if cells is not None:
        cells = [cells] if type(cells) is str else cells
    else:
        cells = data.obs_names.tolist()

    # Subset for cells
    points = points.loc[points["cell"].isin(cells)]

    if genes is not None:
        genes = [genes] if type(genes) is str else genes
    else:
        genes = data.var_names.tolist()

    # Subset for genes
    points = points.loc[points["gene"].isin(genes)]

    # Cast to GeoDataFrame
    points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.x, points.y))

    return points


@track
def set_points(data, cells=None, genes=None, copy=False):
    adata = data.copy() if copy else data
    points = get_points(adata, cells, genes)
    adata.uns["points"] = points
    return adata if copy else None


@track
def filter_points(data, min=None, max=None, copy=False):
    """Select samples with at least min_count and at most max_count points.

    Parameters
    ----------
    data : AnnData
        bento loaded AnnData
    min : int, optional
        minimum points needed to keep sample, by default None
    max : int, optional
        maximum points needed to keep sample, by default None
    copy : bool, optional
        True modifies data in place while False returns a copy of the modified data, by default False
    """
    adata = data.copy() if copy else data
    points = get_points(data)

    expr_flat = adata.to_df().reset_index().melt(id_vars="cell")

    if min:
        expr_flat = expr_flat.query(f"value >= {min}")

    if max:
        expr_flat = expr_flat.query(f"value <= {max}")

    expr_flat = set(tuple(x) for x in expr_flat[["cell", "gene"]].values)

    sample_ids = [tuple(x) for x in points[["cell", "gene"]].values]
    keep = [True if x in expr_flat else False for x in sample_ids]

    points = points.loc[keep]

    # points = points.groupby(['cell', 'gene']).apply(lambda df: df if df.shape[0] >= 5 else None).reset_index(drop=True)
    adata.uns["points"] = points
    return adata if copy else None


@track
def remove_extracellular(data, copy=False):
    """
    Remove extracellular points.
    """
    adata = data.copy() if copy else data
    points = get_points(adata)

    points = points.set_index('cell').join(data.obs["cell_shape"]).reset_index()
    points['cell'] = points['cell'].astype('category').cat.as_ordered()
    
    def _filter_points(pts):
        pts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.x, pts.y))

        cell_shape = pts["cell_shape"].values[0]
        in_cell = pts.within(cell_shape)
        pts = pts[in_cell]

        return pts

    ngroups = points.groupby('cell').ngroups
    
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
    
    points = points.drop("cell_shape", axis=1).drop("geometry", axis=1).reset_index(drop=True)

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
