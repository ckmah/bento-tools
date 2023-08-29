from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import paired_distances

import pandas as pd
import numpy as np

from ..geometry import get_points
#from .._utils import track

from spatialdata._core.spatialdata import SpatialData


def _get_compositions(points: pd.DataFrame, shape_names: list) -> pd.DataFrame:
    """Compute the mean composition of each gene across shapes.

    Parameters
    ----------
    points : pandas.DataFrame
        Points indexed to shape_names denoted by boolean columns.
    shape_names : list of str
        Names of shapes to calculate compositions for.

    Returns
    -------
    comp_data : DataFrame
        For every gene return the composition of each shape, mean log(counts) and cell fraction.

    """

    dims = ["_".join(s.split("_")[:-1]) for s in shape_names]

    n_cells = points["cell"].nunique()
    points_grouped = points.groupby(["cell", "gene"], observed=True)
    counts = points_grouped[dims].sum()
    total_counts = points_grouped.size()
    comps = counts.divide(total_counts, axis=0).fillna(0)  # Normalize rows

    genes = points["gene"].unique()
    gene_comps = (
        comps.groupby("gene", observed=True).mean().reindex(genes, fill_value=0)
    )

    gene_logcount = (
        points.groupby("gene", observed=True).size().reindex(genes, fill_value=0)
    )
    gene_logcount = np.log2(gene_logcount + 1)
    cell_fraction = (
        100 * points.groupby("gene", observed=True)["cell"].nunique() / n_cells
    )

    stats = pd.DataFrame(
        [gene_logcount, cell_fraction], index=["logcounts", "cell_fraction"]
    ).T

    comp_stats = pd.concat([gene_comps, stats], axis=1)

    return comp_stats


#@track
def comp_diff(
    sdata: SpatialData, shape_names: list, groupby: str, ref_group: str, copy: bool = False
):
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
    copy : bool
        Return a copy of `sdata` instead of writing to data, by default False.

    """
    points = get_points(sdata, astype="Pandas")

    # Get average gene compositions for each batch
    comp_stats = dict()
    for group, pt_group in points.groupby(groupby):
        comp_stats[group] = _get_compositions(pt_group, shape_names)

    ref_comp = comp_stats[ref_group]

    dims = [s.replace("_boundaries", "") for s in shape_names]
    for group in comp_stats.keys():
        if group == ref_group:
            continue

        diff_key = f"{group}_diff"
        return comp_stats
        comp_stats[group][diff_key] = pd.Series(
            paired_distances(
                comp_stats[group][dims].reindex(ref_comp.index, fill_value=1e-10),
                ref_comp[dims],
                metric=wasserstein_distance,
            ),
            index=ref_comp.index,
        )

    sdata.table.uns[f"{groupby}_comp_stats"] = comp_stats
