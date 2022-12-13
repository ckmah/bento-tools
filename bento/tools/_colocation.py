import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
from scipy.stats import zscore
from kneed import KneeLocator
import sparse
import emoji


from ..geometry import get_points
from ._neighborhoods import _count_neighbors
from ._signatures import decompose
from .._utils import track


@track
def colocation(
    data,
    ranks,
    iterations=3,
    log=True,
    plot_error=True,
    copy=False,
):
    adata = data.copy() if copy else data

    print("Preparing tensor...")
    _colocation_tensor(adata, copy=copy)

    tensor = adata.uns["tensor"]
    if log:
        tensor = np.log2(tensor + 1)

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


def _colocation_tensor(data, copy=False):
    """
    Convert a dictionary of colocation quotient values in long format to a dense tensor.
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
    labels = []
    label_orders = []
    for name in label_names:
        label, order = np.unique(clq_long[name], return_inverse=True)
        labels.append(label)
        label_orders.append(order)

    label_orders = np.array(label_orders)

    s = sparse.COO(label_orders, data=clq_long["clq"].values)
    tensor = s.todense()
    print(tensor.shape)

    adata.uns["tensor"] = tensor
    adata.uns["tensor_labels"] = labels
    adata.uns["tensor_names"] = label_names

    return adata


@track
def coloc_quotient(
    data,
    shapes=["cell_shape"],
    radius=20,
    min_points=10,
    min_cells=10,
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
    copy : bool
        Whether to copy the AnnData object. Default False.
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
            zip(cells, group_loc, end_loc), desc="shape", total=len(cells)
        ):
            cell_points = points.iloc[start:end]
            cell_clq = _cell_clq(cell_points, adata.n_vars, radius, min_points)
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


def _cell_clq(cell_points, n_genes, radius, min_points):

    # Count number of points for each gene
    counts = cell_points["gene"].value_counts()

    # Keep genes with at least min_count
    counts = counts[counts >= min_points]

    if len(counts) < 2:
        return pd.DataFrame()

    # Get points
    valid_points = cell_points[cell_points["gene"].isin(counts.index)]

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
