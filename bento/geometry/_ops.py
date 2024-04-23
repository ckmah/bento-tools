# Geometric operations for SpatialData ShapeElements wrapping GeoPandas GeoDataFrames.

from dask import dataframe as dd
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import get_transformation

from .._utils import get_feature_key, get_instance_key
from ..io import prep
from ._geometry import get_points


def overlay(
    sdata: SpatialData,
    s1: str,
    s2: str,
    name : str,
    how: str = "intersection",
    make_valid: bool = True,
):
    """Overlay two shape elements in a SpatialData object and store the result as a new shape element.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    s1 : str
        Name of the first shape element
    s2 : str
        Name of the second shape element
    how : str
        Type of overlay operation to perform. Options are "intersection", "union", "difference", "symmetric_difference", by default "intersection"
    make_valid : bool
        If True, correct invalid geometries with GeoPandas, by default True
    instance_key : str
        Name of the shape element to use as the instance for indexing, by default "cell_boundaries". If None, no indexing is performed.

    Returns
    -------
    SpatialData
        A new SpatialData object with the resulting shapes from the overlay operation.
    """
    shape1 = sdata[s1]
    shape2 = sdata[s2]

    new_shape = shape1.overlay(shape2, how=how, make_valid=make_valid)
    new_shape.attrs = {}

    transforms = get_transformation(shape1, get_all=True)
    sdata.shapes[name] = ShapesModel.parse(new_shape, transformations=transforms)

    sdata = prep(
        sdata,
        shape_keys=[name],
        instance_key=get_instance_key(sdata),
        feature_key=get_feature_key(sdata),
    )




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