from typing import List

import emoji
import numpy as np
import pandas as pd
import seaborn as sns
import sparse
from spatialdata._core.spatialdata import SpatialData
from kneed import KneeLocator
from tqdm.auto import tqdm

from .._utils import get_points
from ._neighborhoods import _count_neighbors
from ._decomposition import decompose


def colocation(
    sdata: SpatialData,
    ranks: List[int],
    instance_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    iterations: int = 3,
    plot_error: bool = True,
):
    """Decompose a tensor of pairwise colocalization quotients into signatures.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    ranks : list
        List of ranks to decompose the tensor.
    instance_key : str
        Key that specifies cell_boundaries instance in sdata.
    feature_key : str
        Key that specifies genes in sdata.
    iterations : int
        Number of iterations to run the decomposition.
    plot_error : bool
        Whether to plot the error of the decomposition.

    Returns
    -------
    sdata : SpatialData
        .table.uns['factors']: Decomposed tensor factors.
        .table.uns['factors_error']: Decomposition error.
    """

    print("Preparing tensor...")
    _colocation_tensor(sdata, instance_key, feature_key)

    tensor = sdata.table.uns["tensor"]

    print(emoji.emojize(":running: Decomposing tensor..."))
    factors, errors = decompose(tensor, ranks, iterations=iterations)

    if plot_error and errors.shape[0] > 1:
        kl = KneeLocator(
            errors["rank"], errors["rmse"], direction="decreasing", curve="convex"
        )
        kl.plot_knee()
        sns.lineplot(data=errors, x="rank", y="rmse", ci=95, marker="o")

    sdata.table.uns["factors"] = factors
    sdata.table.uns["factors_error"] = errors

    print(emoji.emojize(":heavy_check_mark: Done."))


def _colocation_tensor(sdata: SpatialData, instance_key: str, feature_key: str):
    """
    Convert a dictionary of colocation quotient values in long format to a dense tensor.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    instance_key : str
        Key that specifies cell_boundaries instance in sdata.
    feature_key : str
        Key that specifies genes in sdata.
    """

    clqs = sdata.table.uns["clq"]

    clq_long = []
    for shape, clq in clqs.items():
        clq["compartment"] = shape
        clq_long.append(clq)

    clq_long = pd.concat(clq_long, axis=0)
    clq_long["pair"] = (
        clq_long[feature_key].astype(str) + "_" + clq_long["neighbor"].astype(str)
    )

    label_names = ["compartment", instance_key, "pair"]
    labels = dict()
    label_orders = []
    for name in label_names:
        label, order = np.unique(clq_long[name], return_inverse=True)
        labels[name] = label
        label_orders.append(order)

    label_orders = np.array(label_orders)

    s = sparse.COO(label_orders, data=clq_long["log_clq"].values)
    tensor = s.todense()

    sdata.table.uns["tensor"] = tensor
    sdata.table.uns["tensor_labels"] = labels
    sdata.table.uns["tensor_names"] = label_names


def coloc_quotient(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    shapes: List[str] = ["cell_boundaries"],
    radius: int = 20,
    min_points: int = 10,
    min_cells: int = 0,
):
    """Calculate pairwise gene colocalization quotient in each cell.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    points_key: str
        Key that specifies transcript points in sdata.
    instance_key : str
        Key that specifies cell_boundaries instance in sdata.
    feature_key : str
        Key that specifies genes in sdata.
    shapes : list
        Specify which shapes to compute colocalization separately.
    radius : int
        Unit distance to count neighbors, default 20
    min_points : int
        Minimum number of points for sample to be considered for colocalization, default 10
    min_cells : int
        Minimum number of cells for gene to be considered for colocalization, default 0

    Returns
    -------
    sdata : SpatialData
        .table.uns['clq']: Pairwise gene colocalization similarity within each cell formatted as a long dataframe.
    """

    all_clq = dict()
    for shape in shapes:
        points = get_points(sdata, points_key=points_key, astype="pandas", sync=True)
        points = (
            points.query(f"{instance_key} != ''")
            .sort_values(instance_key)[[instance_key, feature_key, "x", "y"]]
            .reset_index(drop=True)
        )

        # Keep genes expressed in at least min_cells cells
        gene_counts = points.groupby(feature_key).size()
        valid_genes = gene_counts[gene_counts >= min_cells].index
        points = points[points[feature_key].isin(valid_genes)]

        # Partition so {chunksize} cells per partition
        cells, group_loc = np.unique(
            points[instance_key].astype(str),
            return_index=True,
        )

        end_loc = np.append(group_loc[1:], points.shape[0])

        cell_clqs = []
        for cell, start, end in tqdm(
            zip(cells, group_loc, end_loc), desc=shape, total=len(cells)
        ):
            cell_points = points.iloc[start:end]
            cell_clq = _cell_clq(cell_points, radius, min_points, feature_key)
            cell_clq[instance_key] = cell

            cell_clqs.append(cell_clq)

        cell_clqs = pd.concat(cell_clqs)
        cell_clqs[[instance_key, feature_key, "neighbor"]] = (
            cell_clqs[[instance_key, feature_key, "neighbor"]]
            .astype(str)
            .astype("category")
        )
        cell_clqs["log_clq"] = cell_clqs["clq"].replace(0, np.nan).apply(np.log2)

        # Save to uns['clq'] as adjacency list
        all_clq[shape] = cell_clqs

    sdata.table.uns["clq"] = all_clq


def _cell_clq(cell_points, radius, min_points, feature_key):
    # Count number of points for each gene
    gene_counts = cell_points[feature_key].value_counts()

    # Keep genes with at least min_count
    gene_counts = gene_counts[gene_counts >= min_points]

    if len(gene_counts) < 2:
        return pd.DataFrame()

    # Get points
    valid_points = cell_points[cell_points[feature_key].isin(gene_counts.index)]

    # Cleanup gene categories
    # valid_points["gene"] = valid_points["gene"].cat.remove_unused_categories()

    # Count number of source points that have neighbor gene
    point_neighbors = _count_neighbors(
        valid_points,
        len(valid_points[feature_key].cat.categories),
        radius=radius,
        agg="binary",
    ).toarray()
    neighbor_counts = (
        pd.DataFrame(point_neighbors, columns=valid_points[feature_key].cat.categories)
        .groupby(valid_points[feature_key].values)
        .sum()
        .reset_index()
        .melt(id_vars="index")
        .query("value > 0")
    )
    neighbor_counts.columns = [feature_key, "neighbor", "count"]
    clq_df = _clq_statistic(neighbor_counts, gene_counts, feature_key)

    return clq_df


def _clq_statistic(neighbor_counts, counts, feature_key):
    """
    Compute the colocation quotient for each gene pair.

    Parameters
    ----------
    neighbor_counts : pd.DataFrame
        Dataframe with columns "gene", "neighbor", and "count".
    counts : pd.Series
        Series of raw gene counts.
    """
    clq_df = neighbor_counts.copy()
    clq_df["clq"] = (clq_df["count"] / counts.loc[clq_df[feature_key]].values) / (
        counts.loc[clq_df["neighbor"]].values / counts.sum()
    )
    return clq_df.drop("count", axis=1)
