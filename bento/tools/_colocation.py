from typing import List

import dask
import emoji
import numpy as np
import pandas as pd
import seaborn as sns
import sparse
from kneed import KneeLocator
from spatialdata._core.spatialdata import SpatialData
from tqdm.dask import TqdmCallback

from .._utils import get_points
from ._decomposition import decompose
from ._neighborhoods import _count_neighbors
import dask.bag as db


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
        SpatialData object.
    ranks : List[int]
        List of ranks to decompose the tensor.
    instance_key : str, default "cell_boundaries"
        Key that specifies cell_boundaries instance in sdata.
    feature_key : str, default "feature_name"
        Key that specifies genes in sdata.
    iterations : int, default 3
        Number of iterations to run the decomposition.
    plot_error : bool, default True
        Whether to plot the error of the decomposition.

    Returns
    -------
    SpatialData
        Updated SpatialData object with:
        - .tables["table"].uns['factors']: Decomposed tensor factors.
        - .tables["table"].uns['factors_error']: Decomposition error.
    """

    print("Preparing tensor...")
    _colocation_tensor(sdata, instance_key, feature_key)

    tensor = sdata.tables["table"].uns["tensor"]

    print(emoji.emojize(":running: Decomposing tensor..."))
    factors, errors = decompose(tensor, ranks, iterations=iterations)

    if plot_error and errors.shape[0] > 1:
        kl = KneeLocator(
            errors["rank"], errors["rmse"], direction="decreasing", curve="convex"
        )
        kl.plot_knee()
        sns.lineplot(data=errors, x="rank", y="rmse", ci=95, marker="o")

    sdata.tables["table"].uns["factors"] = factors
    sdata.tables["table"].uns["factors_error"] = errors

    print(emoji.emojize(":heavy_check_mark: Done."))


def _colocation_tensor(sdata: SpatialData, instance_key: str, feature_key: str):
    """
    Convert a dictionary of colocation quotient values in long format to a dense tensor.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object.
    instance_key : str
        Key that specifies cell_boundaries instance in sdata.
    feature_key : str
        Key that specifies genes in sdata.
    """

    clqs = sdata.tables["table"].uns["clq"]

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

    sdata.tables["table"].uns["tensor"] = tensor
    sdata.tables["table"].uns["tensor_labels"] = labels
    sdata.tables["table"].uns["tensor_names"] = label_names


def coloc_quotient(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    shapes: List[str] = ["cell_boundaries"],
    radius: int = 20,
    min_points: int = 10,
    min_cells: int = 0,
    num_workers: int = 1,
):
    """Calculate pairwise gene colocalization quotient in each cell.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object.
    points_key: str, default "transcripts"
        Key that specifies transcript points in sdata.
    instance_key : str, default "cell_boundaries"
        Key that specifies cell_boundaries instance in sdata.
    feature_key : str, default "feature_name"
        Key that specifies genes in sdata.
    shapes : List[str], default ["cell_boundaries"]
        Specify which shapes to compute colocalization separately.
    radius : int, default 20
        Unit distance to count neighbors.
    min_points : int, default 10
        Minimum number of points for sample to be considered for colocalization.
    min_cells : int, default 0
        Minimum number of cells for gene to be considered for colocalization.
    num_workers : int, default 1
        Number of workers to use for parallel processing.

    Returns
    -------
    SpatialData
        Updated SpatialData object with:
        - .tables["table"].uns['clq']: Pairwise gene colocalization similarity within each cell formatted as a long dataframe.
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

        # Filter points by valid genes
        points = points[points[feature_key].isin(valid_genes)]

        # Group points by cell
        points_grouped = points.groupby(instance_key)
        cells = list(points_grouped.groups.keys())
        cells.sort()

        args = [
            (
                points_grouped.get_group(cell),
                radius,
                min_points,
                feature_key,
                instance_key,
            )
            for cell in cells
        ]

        bags = db.from_sequence(args).map(lambda x: _cell_clq(*x))

        # Use dask.compute to execute the operations in parallel
        with TqdmCallback(desc="Batches"), dask.config.set(num_workers=num_workers):
            cell_clqs = bags.compute()

        cell_clqs = pd.concat(cell_clqs)
        cell_clqs[[instance_key, feature_key, "neighbor"]] = (
            cell_clqs[[instance_key, feature_key, "neighbor"]]
            .astype(str)
            .astype("category")
        )

        # Compute log2 of clq and confidence intervals
        cell_clqs["log_clq"] = cell_clqs["clq"].replace(0, np.nan).apply(np.log2)
        cell_clqs["log_ci_lower"] = (
            cell_clqs["ci_lower"].replace(0, np.nan).apply(np.log2)
        )
        cell_clqs["log_ci_upper"] = (
            cell_clqs["ci_upper"].replace(0, np.nan).apply(np.log2)
        )
        # Save to uns['clq'] as adjacency list
        all_clq[shape] = cell_clqs

    sdata.tables["table"].uns["clq"] = all_clq


def _cell_clq(cell_points, radius, min_points, feature_key, instance_key):
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

    point_neighbors = pd.DataFrame(
        point_neighbors, columns=valid_points[feature_key].cat.categories
    )

    # Get gene-level neighbor counts for each gene
    neighbor_counts = (
        point_neighbors.groupby(valid_points[feature_key].values)
        .sum()
        .reset_index()
        .melt(id_vars="index")
        .query("value > 0")
    )
    neighbor_counts.columns = [feature_key, "neighbor", "count"]
    neighbor_counts[instance_key] = cell_points[instance_key].iloc[0]
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
    a = clq_df["count"]
    b = counts.loc[clq_df[feature_key]].values
    c = counts.loc[clq_df["neighbor"]].values
    d = counts.sum()

    clq_df["clq"] = (a / b) / (c / d)

    # Calculate two-tailed 95% confidence interval
    ci_lower = clq_df["clq"] - 1.96 * np.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
    ci_upper = clq_df["clq"] + 1.96 * np.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
    clq_df["ci_lower"] = ci_lower
    clq_df["ci_upper"] = ci_upper
    return clq_df.drop("count", axis=1)
