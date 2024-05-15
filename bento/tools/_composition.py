import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import paired_distances
from spatialdata._core.spatialdata import SpatialData

from .._utils import get_feature_key, get_instance_key, get_points


def _get_compositions(
    points: pd.DataFrame, shape_names: list, instance_key: str, feature_key: str
) -> pd.DataFrame:
    """Compute the mean composition of each gene across shapes.

    Parameters
    ----------
    points : pandas.DataFrame
        Points indexed to shape_names denoted by boolean columns.
    shape_names : list of str
        Names of shapes to calculate compositions for.
    instance_key : str
        Key for

    Returns
    -------
    comp_data : DataFrame
        For every gene return the composition of each shape, mean log(counts) and cell fraction.

    """

    n_cells = points[instance_key].nunique()
    points_grouped = points.groupby([instance_key, feature_key], observed=True)
    counts = points_grouped.apply(lambda x: (x[shape_names] != "").sum())
    total_counts = points_grouped.size()
    comps = counts.divide(total_counts, axis=0).fillna(0)  # Normalize rows

    genes = points[feature_key].unique()
    gene_comps = (
        comps.groupby(feature_key, observed=True).mean().reindex(genes, fill_value=0)
    )

    gene_logcount = (
        points.groupby(feature_key, observed=True).size().reindex(genes, fill_value=0)
    )
    gene_logcount = np.log2(gene_logcount + 1)
    cell_fraction = (
        100
        * points.groupby(feature_key, observed=True)[instance_key].nunique()
        / n_cells
    )

    stats = pd.DataFrame(
        [gene_logcount, cell_fraction], index=["logcounts", "cell_fraction"]
    ).T

    comp_stats = pd.concat([gene_comps, stats], axis=1)

    return comp_stats


def comp(sdata: SpatialData, points_key: str, shape_names: list):
    """Calculate the average gene composition for shapes across all cells.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        Spatial formatted SpatialData object.
    shape_names : list of str
        Names of shapes to calculate compositions for.

    Returns
    -------
    sdata : spatialdata.SpatialData
        Updates `sdata.table.uns` with average gene compositions for each shape.
    """
    points = get_points(sdata,points_key=points_key, astype="pandas")

    instance_key = get_instance_key(sdata)
    feature_key = get_feature_key(sdata)

    # Get average gene compositions for each batch
    comp_stats = _get_compositions(
        points, shape_names, instance_key=instance_key, feature_key=feature_key
    )

    sdata.table.uns["comp_stats"] = comp_stats


def comp_diff(sdata: SpatialData, points_key: str, shape_names: list, groupby: str, ref_group: str):
    """Calculate the average difference in gene composition for shapes across batches of cells. Uses the Wasserstein distance.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        Spatial formatted SpatialData object.
    shape_names : list of str
        Names of shapes to calculate compositions for.
    groupby : str
        Key in `sdata.points['transcripts]` to group cells by.
    ref_group : str
        Reference group to compare other groups to.

    """
    points = get_points(sdata, points_key=points_key, astype="pandas")

    instance_key = get_instance_key(sdata)
    feature_key = get_feature_key(sdata)

    # Get average gene compositions for each batch
    comp_stats = dict()
    for group, pt_group in points.groupby(groupby):
        comp_stats[group] = _get_compositions(pt_group, shape_names, instance_key=instance_key, feature_key=feature_key)

    ref_comp = comp_stats[ref_group]

    dims = [s.replace("_boundaries", "") for s in shape_names]
    for group in comp_stats.keys():
        if group == ref_group:
            continue

        diff_key = f"{group}_diff"
        comp_stats[group][diff_key] = pd.Series(
            paired_distances(
                comp_stats[group][dims].reindex(ref_comp.index, fill_value=1e-10),
                ref_comp[dims],
                metric=wasserstein_distance,
            ),
            index=ref_comp.index,
        )

    sdata.table.uns[f"{groupby}_comp_stats"] = comp_stats
