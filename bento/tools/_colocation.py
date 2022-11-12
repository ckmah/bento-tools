import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns

from ..preprocessing import get_points
from ._neighborhoods import _count_neighbors
from ._signatures import decompose
from .._utils import track


def colocation(
    data,
    ranks,
    shapes,
    iterations=3,
    self_pairs=False,
    radius=20,
    min_count=20,
    plot_error=True,
    copy=False,
):
    adata = data.copy() if copy else data

    coloc_quotient(data, shapes, radius=radius, min_count=min_count)

    compartment_names = ["_".join(str(shape).split("_")[:-1]) for shape in shapes]

    tensor, labels, label_names = _colocation_tensor(
        list(data.uns["clq"].values()),
        compartment_names=compartment_names,
        self_pairs=self_pairs,
    )

    factors, errors = decompose(tensor, ranks, iterations=iterations)

    if plot_error:
        sns.lineplot(data=errors, x="rank", y="rmse", ci=95, marker="o")

    return factors, labels, label_names


def _to_dense(clq, min_fraction_cells=0.1, self_pairs=True, log=True):
    clq_df = clq.copy()

    # Remove self pairs as specified
    if not self_pairs:
        clq_df[["gene", "neighbor"]] = clq_df[["gene", "neighbor"]].astype(str)
        clq_df = clq_df.query("gene != neighbor")

    # scale by log2(clq+1)
    if log:
        clq_df["clq"] = np.log2(clq_df["clq"] + 1)

    clq_df["pair"] = clq_df["gene"].astype(str) + "_" + clq_df["neighbor"].astype(str)

    # Keep pairs expressed >= min_fraction_Cells
    pair_counts = clq_df["pair"].value_counts()
    min_ncells = int(clq_df["cell"].nunique() * min_fraction_cells)
    valid_pairs = pair_counts.index[pair_counts >= min_ncells].tolist()
    clq_df = clq_df[clq_df["pair"].isin(valid_pairs)]

    # Dense
    clq_dense = clq_df.pivot(index="cell", columns="pair", values="clq")

    return clq_dense


def _colocation_tensor(clqs, compartment_names, self_pairs):
    mtxs = []
    cell_names = []
    pair_names = []
    for clq in clqs:
        mtx = _to_dense(clq, self_pairs=self_pairs)
        mtxs.append(mtx)
        cell_names.extend(mtx.index.tolist())
        pair_names.extend(mtx.columns.tolist())

    pair_names = list(set(pair_names))
    cell_names = list(set(cell_names))

    slices = []
    for mtx in mtxs:
        slices.append(mtx.reindex(index=cell_names, columns=pair_names).to_numpy())

    tensor = np.stack(slices)
    print(tensor.shape)

    labels = [compartment_names, cell_names, pair_names]
    label_names = ["compartments", "cells", "pairs"]

    return tensor, labels, label_names


@track
def coloc_quotient(
    data,
    shapes=["cell_shape"],
    radius=20,
    min_count=20,
    copy=False,
):
    """Calculate pairwise gene colocalization quotient in each cell.

    Parameters
    ----------
    adata : AnnData
        AnnData formatted spatial data.
    shapes : list
        Specify which shapes to compute colocalization separately.
    radius : int
        Unit distance to count neighbors, default 20
    min_count : int
        Minimum number of points for a given gene in a cell to be considered, default 20
    chunksize : int
        Number of cells per processing chunk. Default 64.
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
        points[shape_col] = points[shape_col].astype(int)
        points = (
            points.query(f"{shape_col} != -1")
            .sort_values("cell")[["cell", "gene", "x", "y"]]
            .reset_index(drop=True)
        )

        # Partition so {chunksize} cells per partition
        cells, group_loc = np.unique(
            points["cell"].astype(str),
            return_index=True,
        )

        end_loc = np.append(group_loc[1:], points.shape[0])

        cell_clqs = []
        for cell, start, end in tqdm(zip(cells, group_loc, end_loc), total=len(cells)):
            cell_points = points.iloc[start:end]
            cell_clq = _cell_clq(cell_points, adata.n_vars, radius, min_count)
            cell_clq["cell"] = cell
            cell_clqs.append(cell_clq)

        cell_clqs = pd.concat(cell_clqs)
        cell_clqs[["cell", "gene", "neighbor"]] = (
            cell_clqs[["cell", "gene", "neighbor"]].astype(str).astype("category")
        )

        # Save to uns['clq'] as adjacency list
        all_clq[shape] = cell_clqs

    adata.uns["clq"] = all_clq

    return adata if copy else None


def _cell_clq(cell_points, n_genes, radius, min_count):

    # Count number of points for each gene
    counts = cell_points["gene"].value_counts()

    # Only keep genes >= min_count
    counts = counts[counts >= min_count]
    valid_genes = counts.sort_index().index.tolist()
    counts = counts[valid_genes]

    if len(valid_genes) < 2:
        return pd.DataFrame()

    # Get points
    valid_points = cell_points[cell_points["gene"].isin(valid_genes)]

    # Cleanup gene categories
    valid_points["gene"] = valid_points["gene"].cat.remove_unused_categories()

    neighbor_counts = _count_neighbors(valid_points, n_genes, radius=radius, agg=True)

    clq_df = _clq_statistic(neighbor_counts, counts)

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


def global_clq(neighbor_counts, counts):
    gclq_df = neighbor_counts.copy()
    gclq_df = gclq_df[gclq_df["gene"] == gclq_df["neighbor"]]
    global_counts = counts.loc[gclq_df["gene"]].values
    total_count = global_counts.sum()
    gclq_df["gclq"] = (
        gclq_df["gclq"].sum()
        / (global_counts * ((global_counts - 1) / (total_count - 1))).sum()
    )

    return global_clq


def local_clq(adata, gene_a, gene_b):
    """
    Compute local colocation quotients for every point between two genes across all cells. Note that this is not a symmetric function.
    Parameters
    ----------
    gene_a : str
        Gene name
    gene_b : str
        Gene name
    Returns
    -------
    clq : float
        Local colocation quotient for each point in gene_b
    """
    # Get points for cell
    points = get_points(adata, asgeo=False)

    # Get counts for each gene
    counts = points["gene"].value_counts()

    return
    # nai->b / nb (N - 1)
