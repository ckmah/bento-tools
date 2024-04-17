from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import TableModel
from dask import dataframe as dd

from .geometry._geometry import get_points

def filter_by_gene(
    sdata: SpatialData,
    threshold: int = 10, 
    points_key: str = "transcripts", 
    feature_key: str = "feature_name"
):
    """
    Filters out genes with low expression from the spatial data object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    threshold : int
        Minimum number of counts for a gene to be considered expressed.
        Keep genes where at least {threshold} molecules are detected in at least one cell.
    points_key : str
        key for points element that holds transcript coordinates
    feature_key : str
        Key for gene instances

    Returns
    -------
    sdata : SpatialData
        .points[points_key] is updated to remove genes with low expression.
        .table is updated to remove genes with low expression.
    """
    gene_filter = (sdata.table.X >= threshold).sum(axis=0) > 0
    filtered_table = sdata.table[:, gene_filter]

    filtered_genes = list(sdata.table.var_names.difference(filtered_table.var_names))
    points = get_points(sdata, points_key=points_key, astype="pandas", sync=False)
    points = points[~points[feature_key].isin(filtered_genes)]
    points[feature_key] = points[feature_key].cat.remove_unused_categories()

    transform = sdata.points["transcripts"].attrs
    sdata.points[points_key] = dd.from_pandas(points, npartitions=1)
    sdata.points[points_key].attrs = transform
    
    del sdata.table
    sdata.table = TableModel.parse(filtered_table)

    return sdata