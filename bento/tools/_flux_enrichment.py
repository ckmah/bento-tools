from typing import Optional

import decoupler as dc

import pandas as pd
import pkg_resources
from anndata import AnnData

from bento._utils import track, _register_points


def fe_fazal2019(data: AnnData, copy: bool = False, **kwargs) -> Optional[AnnData]:
    """Compute enrichment scores from subcellular compartment gene sets from Fazal et al. 2019 (APEX-seq).
    See `bento.tl.fe` docs for parameter details.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    copy : bool
        Return a copy instead of writing to `adata`. Default False.
    Returns
    -------
    DataFrame
        Enrichment scores for each gene set.
    """
    adata = data.copy() if copy else data

    gene_sets = load_gene_sets("fazal2019")
    fe(adata, net=gene_sets, **kwargs)

    return adata if copy else None


def fe_xia2019(data: AnnData, copy: bool = False, **kwargs) -> Optional[AnnData]:
    """Compute enrichment scores from subcellular compartment gene sets from Xia et al. 2019 (MERFISH 10k U2-OS).
    See `bento.tl.fe` docs for parameters details.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    copy : bool
        Return a copy instead of writing to `adata`. Default False.
    Returns
    -------
    DataFrame
        Enrichment scores for each gene set.
    """
    adata = data.copy() if copy else data

    gene_sets = load_gene_sets("xia2019")
    fe(adata, gene_sets, **kwargs)

    return adata if copy else None


@track
def fe(
    data: AnnData,
    net: pd.DataFrame,
    source: Optional[str] = "source",
    target: Optional[str] = "target",
    weight: Optional[str] = "weight",
    batch_size: int = 10000,
    min_n: int = 0,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Perform functional enrichment on point embeddings. Wrapper for decoupler wsum function.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
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
    copy : bool
        Return a copy instead of writing to `adata`. Default False.

    Returns
    -------
    adata : AnnData
        uns["flux_fe"] : DataFrame
            Enrichment scores for each gene set.
    """

    adata = data.copy() if copy else data

    # Make sure embedding is run first
    if "flux" not in data.uns:
        print("Run bento.tl.flux first.")
        return

    mat = adata.uns["flux"]  # sparse matrix in csr format
    zero_rows = mat.getnnz(1) == 0

    samples = adata.uns["cell_raster"].index.astype(str)
    features = adata.uns["flux_genes"]

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

    scores = enrichment[1].reindex(index=samples)

    for col in scores.columns:
        score_key = f"flux_{col}"
        adata.uns[score_key] = scores[col].values

        # Manually call register_points since it is dynamic
        _register_points(adata, "cell_raster", [score_key])

    _fe_stats(adata, net, source=source, target=target, copy=copy)

    return adata if copy else None


def _fe_stats(
    data: AnnData,
    net: pd.DataFrame,
    source: str = "source",
    target: str = "target",
    copy: bool = False,
):
    adata = data.copy() if copy else data

    # rows = cells, columns = pathways, values = count of genes in pathway
    expr_binary = adata.to_df() >= 5
    # {cell : present gene list}
    expr_genes = expr_binary.apply(lambda row: adata.var_names[row], axis=1)

    # Count number of genes present in each pathway
    net_ngenes = net.groupby(source).size().to_frame().T.rename(index={0: "n_genes"})

    sources = []
    # common_genes = {}  # list of [cells: gene set overlaps]
    common_ngenes = []  # list of [cells: overlap sizes]
    for source, group in net.groupby(source):
        sources.append(source)
        common = expr_genes.apply(lambda genes: set(genes).intersection(group[target]))
        # common_genes[source] = np.array(common)
        common_ngenes.append(common.apply(len))

    fe_stats = pd.concat(common_ngenes, axis=1)
    fe_stats.columns = sources

    adata.uns["fe_stats"] = fe_stats
    # adata.uns["fe_genes"] = common_genes
    adata.uns["fe_ngenes"] = net_ngenes

    return adata if copy else None


gene_sets = dict(
    fazal2019="fazal2019.csv",
    xia2019="xia2019.csv",
)


def load_gene_sets(name):
    """Load a gene set from bento.

    Parameters
    ----------
    name : str
        Name of gene set to load.

    Returns
    -------
    DataFrame
        Gene set.
    """
    global pkg_resources
    if pkg_resources is None:
        import pkg_resources

    fname = gene_sets[name]
    stream = pkg_resources.resource_stream(__name__, f"gene_sets/{fname}")
    gs = pd.read_csv(stream)

    return gs
