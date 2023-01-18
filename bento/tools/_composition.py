from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import paired_distances

import pandas as pd

from ..geometry import get_points
from .._utils import track


def _get_compositions(points, shape_names, return_counts=False):
    """Compute the composition of each gene across shapes within each cell.

    Parameters
    ----------
    points : pandas.DataFrame
        Points indexed to shape_names denoted by boolean columns.
    shape_names : list of str
        Names of shapes to calculate compositions for.
    return_counts : bool, optional (default: False)
        Whether to return the counts of each shape in each cell.

    Returns
    -------
    comps : pandas.DataFrame
        Gene composition of each shape in each cell.
    counts : pandas.DataFrame, optional
        Gene counts of each shape in each cell.
    """

    dims = ["_".join(s.split("_")[:-1]) for s in shape_names]

    points_grouped = points.groupby(["cell", "gene"], observed=True)
    counts = points_grouped[dims].sum()
    total_counts = points_grouped.size()
    comps = counts.divide(total_counts, axis=0).fillna(0)  # Normalize rows

    if return_counts:
        return comps, counts
    else:
        return comps


@track
def comp_diff(data, shape_names, groupby, ref_group, copy=False):
    """Calculate the average difference in gene composition for shapes across batches of cells. Uses the Wasserstein distance.

    Parameters
    ----------
    data : anndata.AnnData
        Spatial formatted AnnData object.
    shape_names : list of str


    """

    adata = data.copy() if copy else data

    points = get_points(data)

    # Get average gene compositions for each batch
    comps = dict()
    for group, pt_group in points.groupby(groupby):
        comps[group] = (
            _get_compositions(pt_group, shape_names)
            .groupby("gene")
            .mean()
            .reindex(data.var_names, fill_value=0)
        )

    ref_comp = comps[ref_group]

    diff = dict()
    for batch in comps.keys():
        if batch == ref_group:
            continue

        diff[batch] = pd.Series(
            paired_distances(comps[batch], ref_comp, metric=wasserstein_distance),
            index=ref_comp.index,
        )

    diff = pd.DataFrame.from_dict(diff)

    adata.uns[f"{groupby}_comp_diff"] = diff
