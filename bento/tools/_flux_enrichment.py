from typing import Optional

import decoupler as dc

import numpy as np
import pandas as pd
from scipy import sparse
from spatialdata._core.spatialdata import SpatialData

from .._utils import get_points, set_points_metadata


def fe_fazal2019(sdata: SpatialData, **kwargs):
    """Compute enrichment scores from subcellular compartment gene sets from Fazal et al. 2019 (APEX-seq).
    See `bento.tl.fe` docs for parameter details.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object.

    Returns
    -------
    DataFrame
        Enrichment scores for each gene set.
    """

    gene_sets = load_gene_sets("fazal2019")
    fe(sdata, net=gene_sets, **kwargs)


def fe_xia2019(sdata: SpatialData, **kwargs):
    """Compute enrichment scores from subcellular compartment gene sets from Xia et al. 2019 (MERFISH 10k U2-OS).
    See `bento.tl.fe` docs for parameters details.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object.

    Returns
    -------
    DataFrame
        Enrichment scores for each gene set.
    """

    gene_sets = load_gene_sets("xia2019")
    fe(sdata, gene_sets, **kwargs)


def fe(
    sdata: SpatialData,
    net: pd.DataFrame,
    instance_key: Optional[str] = "cell_boundaries",
    source: Optional[str] = "source",
    target: Optional[str] = "target",
    weight: Optional[str] = "weight",
    batch_size: int = 10000,
    min_n: int = 0,
):
    """
    Perform functional enrichment of RNAflux embeddings. Uses decoupler wsum function.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    net : DataFrame
        DataFrame with columns "source", "target", and "weight". See decoupler API for more details.
    source : str, optional
        Column name for source nodes in `net`. Default "source".
    target : str, optional
        Column name for target nodes in `net`. Default "target".
    weight : str, optional
        Column name for weights in `net`. Default "weight".
    batch_size : int
        Number of points to process in each batch. Default 10000.
    min_n : int
        Minimum number of targets per source. If less, sources are removed.

    Returns
    -------
    sdata : SpatialData
        .points["cell_boundaries_raster"]["flux_fe"] : DataFrame
            Enrichment scores for each gene set.
    """
    # Make sure embedding is run first
    if "flux_genes" in sdata.table.uns:
        flux_genes = set(sdata.table.uns["flux_genes"])
        cell_raster_columns = set(sdata.points[f"{instance_key}_raster"].columns)
        if len(flux_genes.intersection(cell_raster_columns)) != len(flux_genes):
            print("Recompute bento.tl.flux first.")
            return
    else:
        print("Run bento.tl.flux first.")
        return

    features = sdata.table.uns["flux_genes"]
    cell_raster = get_points(
        sdata, points_key=f"{instance_key}_raster", astype="pandas", sync=False
    )[features]
    cell_raster_matrix = np.mat(cell_raster)
    mat = sparse.csr_matrix(cell_raster_matrix)  # sparse matrix in csr format

    samples = cell_raster.index.astype(str)

    enrichment = dc.run_wsum(
        mat=[mat, samples, features],
        net=net,
        source=source,
        target=target,
        weight=weight,
        batch_size=batch_size,
        min_n=min_n,
        verbose=True,
    )

    scores = enrichment[1].add_prefix("flux_").reindex(samples)
    set_points_metadata(
        sdata,
        points_key=f"{instance_key}_raster",
        metadata=scores,
        columns=scores.columns,
    )

    _fe_stats(sdata, net, source=source, target=target)


def _fe_stats(
    sdata: SpatialData,
    net: pd.DataFrame,
    source: str = "source",
    target: str = "target",
):
    # rows = cells, columns = pathways, values = count of genes in pathway
    expr_binary = sdata.table.to_df() >= 5
    # {cell : present gene list}
    expr_genes = expr_binary.apply(lambda row: sdata.table.var_names[row], axis=1)

    # Count number of genes present in each pathway
    net_ngenes = net.groupby(source).size().to_frame().T.rename(index={0: "n_genes"})

    sources = []
    # common_genes = {}  # list of [cells: gene set overlaps]
    common_ngenes = []  # list of [cells: overlap sizes]
    for source, group in net.groupby(source):
        sources.append(source)
        common = expr_genes.apply(lambda genes: set(genes).intersection(group[target]))
        common_ngenes.append(common.apply(len))

    fe_stats = pd.concat(common_ngenes, axis=1)
    fe_stats.columns = sources

    sdata.table.uns["fe_stats"] = fe_stats
    sdata.table.uns["fe_ngenes"] = net_ngenes


gene_sets = dict(
    fazal2019="fazal2019.csv",
    xia2019="xia2019.csv",
)


def load_gene_sets(name):
    """Load a gene set; list available ones with `bento.tl.gene_sets`.

    Parameters
    ----------
    name : str
        Name of gene set to load.

    Returns
    -------
    DataFrame
        Gene set.
    """
    from importlib.resources import files, as_file

    fname = gene_sets[name]
    ref = files(__package__) / f"gene_sets/{fname}"
    with as_file(ref) as path:
        gs = pd.read_csv(path)

        return gs
