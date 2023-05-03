from typing import List

import emoji
import numpy as np
import pandas as pd
import seaborn as sns
import sparse
from anndata import AnnData
from kneed import KneeLocator
from tqdm.auto import tqdm

from .._utils import track
from ..geometry import get_points
from ._neighborhoods import _count_neighbors
from ._decomposition import decompose


@track
def colocation(
    data: AnnData,
    ranks: List[int],
    iterations: int = 3,
    plot_error: bool = True,
    copy: bool = False,
):
    """Decompose a tensor of pairwise colocalization quotients into signatures.

    Parameters
    ----------
    adata : AnnData
        Spatial formatted AnnData object.
    ranks : list
        List of ranks to decompose the tensor.
    iterations : int
        Number of iterations to run the decomposition.
    plot_error : bool
        Whether to plot the error of the decomposition.
    copy : bool
        Whether to return a copy of the AnnData object. Default False.
    Returns
    -------
    adata : AnnData
        .uns['factors']: Decomposed tensor factors.
        .uns['factors_error']: Decomposition error.
    """
    adata = data.copy() if copy else data

    print("Preparing tensor...")
    _colocation_tensor(adata, copy=copy)

    tensor = adata.uns["tensor"]

    print(emoji.emojize(":running: Decomposing tensor..."))
    factors, errors = decompose(tensor, ranks, iterations=iterations)

    if plot_error and errors.shape[0] > 1:
        kl = KneeLocator(
            errors["rank"], errors["rmse"], direction="decreasing", curve="convex"
        )
        kl.plot_knee()
        sns.lineplot(data=errors, x="rank", y="rmse", ci=95, marker="o")

    adata.uns["factors"] = factors
    adata.uns["factors_error"] = errors

    print(emoji.emojize(":heavy_check_mark: Done."))
    return adata if copy else None


def _colocation_tensor(data: AnnData, copy: bool = False):
    """
    Convert a dictionary of colocation quotient values in long format to a dense tensor.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    copy : bool
        Whether to return a copy of the AnnData object. Default False.
    """
    adata = data.copy() if copy else data

    clqs = adata.uns["clq"]

    clq_long = []
    for shape, clq in clqs.items():
        clq["compartment"] = shape
        clq_long.append(clq)

    clq_long = pd.concat(clq_long, axis=0)
    clq_long["pair"] = (
        clq_long["gene"].astype(str) + "_" + clq_long["neighbor"].astype(str)
    )

    label_names = ["compartment", "cell", "pair"]
    labels = dict()
    label_orders = []
    for name in label_names:
        label, order = np.unique(clq_long[name], return_inverse=True)
        labels[name] = label
        label_orders.append(order)

    label_orders = np.array(label_orders)

    s = sparse.COO(label_orders, data=clq_long["log_clq"].values)
    tensor = s.todense()
    print(tensor.shape)

    adata.uns["tensor"] = tensor
    adata.uns["tensor_labels"] = labels
    adata.uns["tensor_names"] = label_names

    return adata


@track
def coloc_quotient(
    data: AnnData,
    shapes: List[str] = ["cell_shape"],
    radius: int = 20,
    min_points: int = 10,
    min_cells: int = 0,
    copy: bool = False,
):
    """Calculate pairwise gene colocalization quotient in each cell.

    Parameters
    ----------
    adata : AnnData
        Spatial formatted AnnData object.
    shapes : list
        Specify which shapes to compute colocalization separately.
    radius : int
        Unit distance to count neighbors, default 20
    min_points : int
        Minimum number of points for sample to be considered for colocalization, default 10
    min_cells : int
        Minimum number of cells for gene to be considered for colocalization, default 0
    copy : bool
        Whether to return a copy of the AnnData object. Default False.
    Returns
    -------
    adata : AnnData
        .uns['clq']: Pairwise gene colocalization similarity within each cell formatted as a long dataframe.
    """

    adata = data.copy() if copy else data

    all_clq = dict()
    for shape in shapes:
        shape_col = "_".join(str(shape).split("_")[:-1])
        points = get_points(adata, asgeo=False)
        points[shape_col] = points[shape_col].astype(str)
        points = (
            points.query(f"{shape_col} != '-1'")
            .sort_values("cell")[["cell", "gene", "x", "y"]]
            .reset_index(drop=True)
        )

        # Keep genes expressed in at least min_cells cells
        gene_counts = points.groupby("gene").size()
        valid_genes = gene_counts[gene_counts >= min_cells].index
        points = points[points["gene"].isin(valid_genes)]

        # Partition so {chunksize} cells per partition
        cells, group_loc = np.unique(
            points["cell"].astype(str),
            return_index=True,
        )

        end_loc = np.append(group_loc[1:], points.shape[0])

        cell_clqs = []
        for cell, start, end in tqdm(
            zip(cells, group_loc, end_loc), desc=shape, total=len(cells)
        ):
            cell_points = points.iloc[start:end]
            cell_clq = _cell_clq(cell_points, adata.n_vars, radius, min_points)
            cell_clq["cell"] = cell

            cell_clqs.append(cell_clq)

        cell_clqs = pd.concat(cell_clqs)
        cell_clqs[["cell", "gene", "neighbor"]] = (
            cell_clqs[["cell", "gene", "neighbor"]].astype(str).astype("category")
        )
        cell_clqs["log_clq"] = cell_clqs["clq"].replace(0, np.nan).apply(np.log2)

        # Save to uns['clq'] as adjacency list
        all_clq[shape] = cell_clqs

    adata.uns["clq"] = all_clq

    return adata if copy else None


def _cell_clq(cell_points, n_genes, radius, min_points):

    # Count number of points for each gene
    gene_counts = cell_points["gene"].value_counts()

    # Keep genes with at least min_count
    gene_counts = gene_counts[gene_counts >= min_points]

    if len(gene_counts) < 2:
        return pd.DataFrame()

    # Get points
    valid_points = cell_points[cell_points["gene"].isin(gene_counts.index)]

    # Cleanup gene categories
    # valid_points["gene"] = valid_points["gene"].cat.remove_unused_categories()

    # Count number of source points that have neighbor gene
    point_neighbors = _count_neighbors(
        valid_points, n_genes, radius=radius, agg="binary"
    ).toarray()
    neighbor_counts = (
        pd.DataFrame(point_neighbors, columns=valid_points["gene"].cat.categories)
        .groupby(valid_points["gene"].values)
        .sum()
        .reset_index()
        .melt(id_vars="index")
        .query("value > 0")
    )
    neighbor_counts.columns = ["gene", "neighbor", "count"]
    clq_df = _clq_statistic(neighbor_counts, gene_counts)

    return clq_df


def _clq_statistic(neighbor_counts, counts):
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
    clq_df["clq"] = (clq_df["count"] / counts.loc[clq_df["gene"]].values) / (
        counts.loc[clq_df["neighbor"]].values / counts.sum()
    )
    return clq_df.drop("count", axis=1)
