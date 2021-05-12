import warnings
from collections import defaultdict

import bento
import geopandas
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as sfm
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError
from tqdm.auto import tqdm

warnings.simplefilter("ignore", ConvergenceWarning)


def spots_diff(data, groups=None, continuous=None, copy=False, n_cores=1):
    """Test for differential localization across phenotype of interest.

    Parameters
    ----------
    data : AnnData
        Anndata formatted spatial transcriptomics data.
    groups : str, pd.Series
        Variable grouping cells for differential analysis. If str, needs to be a key in data.uns['sample_data']. If pandas Series, must be same length as number of cells in 'data'.
    copy : bool, optional
        Return view of AnnData if False, return copy if True. By default False.
    """
    adata = data.copy() if copy else data

    # Get index and patterns
    diff_data = (
        adata.uns["sample_index"]
        .reset_index(drop=True)
        .join(
            adata.uns["sample_data"]["patterns"]
            .loc[:, adata.uns["sample_data"]["patterns"].sum() > 0]
            .reset_index(drop=True)
        )
    )

    # Get group/continuous phenotype
    phenotype = None
    if groups and not continuous:
        phenotype = groups
    elif continuous and not groups:
        phenotype = continuous
    else:
        print(
            'Either "groups" or "continuous" parameters need to be specified, not both.'
        )

    diff_data = diff_data.reset_index(drop=True).join(
        adata.uns["sample_data"][phenotype]
    )

    # Test each gene independently
    if n_cores > 1:
        parallel = Parallel(n_jobs=n_cores, verbose=0)
        results = parallel(
            delayed(_test_gene)(gene, gene_df, phenotype, continuous)
            for gene, gene_df in tqdm(diff_data.groupby("gene"))
        )
        results = pd.concat(results)
    else:
        tqdm.pandas(desc=f"Testing {phenotype}")
        results = diff_data.groupby("gene").progress_apply(
            lambda gene_df: _test_gene(gene_df.name, gene_df, phenotype, continuous)
        )

    # Format pattern column
    results = results.reset_index(drop=True)

    # FDR correction
    results_adj = []
    for _, df in results.groupby("pattern"):
        df["padj"] = multipletests(df["pvalue"], method="hs")[1]
        results_adj.append(df)

    results_adj = pd.concat(results_adj)
    results_adj = results_adj.dropna()

    # -log10pvalue, padj
    results_adj["-log10p"] = -np.log10(results_adj["pvalue"].astype(np.float32))
    results_adj["-log10padj"] = -np.log10(results_adj["padj"].astype(np.float32))

    # Sort results
    results_adj = results_adj.sort_values("pvalue")

    # Save back to AnnData
    adata.uns["sample_data"][f"dl_{phenotype}"] = results_adj

    return adata if copy else None


def _test_gene(gene, data, phenotype, continuous):
    """Perform pairwise comparison between groupby and every class.

    Parameters
    ----------
    data : DataFrame
        Phenotype and localization pattern labels across cells for a single gene.
    groupby : str
        Variable grouping cells for differential analysis. Should be present in data.columns.

    Returns
    -------
    DataFrame
        Differential localization test results. [# of patterns, ]
    """
    results = []

    # Make dummy columns and add to table
    classes = [
        c
        for c in data.columns
        if c in ["cell2D", "cellext", "foci", "nuc2D", "polarized", "random"]
    ]
    if not continuous:
        group_dummies = pd.get_dummies(data[phenotype])
        group_names = group_dummies.columns.tolist()
        group_data = pd.concat([data, group_dummies], axis=1)

        for g in group_names:
            for c in classes:
                try:
                    res = sfm.logit(formula=f"{g} ~ {c}", data=group_data).fit(disp=0)
                    r = res.get_margeff().summary_frame()
                    r["gene"] = gene
                    r["phenotype"] = g
                    r["pattern"] = c
                    r.columns = [
                        "dy/dx",
                        "std_err",
                        "z",
                        "pvalue",
                        "ci_low",
                        "ci_high",
                        "gene",
                        "phenotype",
                        "pattern",
                    ]
                    r = r.reset_index(drop=True)
                    results.append(r)
                except (np.linalg.LinAlgError, PerfectSeparationError):
                    continue
        results = pd.concat(results)
    else:
        for c in classes:
            corr, p = stats.spearmanr(data[phenotype], data[c])
            results.append(
                pd.Series(
                    [corr, p, gene, continuous],
                    index=["r", "pvalue", "gene", "phenotype"],
                )
                .to_frame()
                .T
            )
        results = pd.concat(results)
        results["pattern"] = classes

    return results


def score_genes_cell_cycle(data, copy=False, **kwargs):
    """Wrapper to preprocess counts similar to scRNA-seq and score cell cycle genes via scanpy.

    Parameters
    ----------
    data : AnnData
        Spatial anndata object.
    copy : bool, optional
        Copy data or modify inplace, by default False.
    **kwargs
        Passed to scanpy.tl.score_genes().
    """
    adata = data.copy() if copy else data

    # Extract points
    expression = pd.DataFrame(
        adata.X, index=pd.MultiIndex.from_frame(adata.obs[["cell", "gene"]])
    )

    # Aggregate points to counts
    expression = (
        adata.obs[["cell", "gene"]]
        .groupby(["cell", "gene"])
        .apply(lambda x: x.shape[0])
        .to_frame()
    )
    expression = expression.reset_index()

    # Remove extracellular points
    expression = expression.loc[expression["cell"] != "-1"]

    # Format as dense cell x gene counts matrix
    expression = expression.pivot(index="cell", columns="gene").fillna(0)
    expression.columns = expression.columns.droplevel(0)
    expression.columns = expression.columns.str.upper()

    # Create scanpy anndata object to use scoring function
    sc_data = sc.AnnData(expression)

    # Perform standard scRNA-seq filtering
    # sc.pp.filter_cells(sc_data, min_genes=200)
    # sc.pp.filter_genes(sc_data, min_cells=3)

    # Standard scaling
    sc.pp.normalize_per_cell(sc_data, counts_per_cell_after=1e4)
    sc.pp.log1p(sc_data)
    sc.pp.scale(sc_data)

    # Get cell cycle genes
    cell_cycle_genes = pd.read_csv(
        "https://github.com/theislab/scanpy_usage/raw/master/180209_cell_cycle/data/regev_lab_cell_cycle_genes.txt",
        header=None,
    )
    cell_cycle_genes = cell_cycle_genes.values.flatten().tolist()

    # Define genes for each phase
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]

    # Only score genes present in data
    cell_cycle_genes = [x for x in cell_cycle_genes if x in sc_data.var_names]

    # Score cell cycle genes
    sc.tl.score_genes_cell_cycle(sc_data, s_genes=s_genes, g2m_genes=g2m_genes)

    # Get indexed cell cycle scores and phase labels
    sc_obs = sc_data.obs.reset_index()
    sc_obs.index = sc_obs.index.astype(str)
    sc_obs["cell"] = sc_obs["cell"].astype(str)
    sc_obs = adata.uns["sample_index"].merge(sc_obs, on="cell", how="left")
    sc_obs = sc_obs.rename({"phase": "cell_cycle"}, axis=1)

    # Save to spatial anndata object
    _init_sample_info(adata)
    adata.uns["sample_data"]["S_score"] = sc_obs[["S_score"]]
    adata.uns["sample_data"]["G2M_score"] = sc_obs[["G2M_score"]]
    adata.uns["sample_data"]["cell_cycle"] = sc_obs[["cell_cycle"]]
    return adata if copy else None



def _init_sample_info(data):
    if "sample_index" not in data.uns.keys():
        sample_index = (
            data.obs[["cell", "gene"]].value_counts().index.to_frame(index=False)
        )
        sample_index.columns = ["cell", "gene"]
        sample_index.index = sample_index.index.astype(str)
        data.uns["sample_index"] = sample_index
        data.uns["sample_data"] = dict()


def subsample_points(data, fraction):
    """Randomly subsample data stratified by cell.
    Parameters
    ----------
    data : DataFrame
        DataFrame with columns x, y, cell, gene annotations.
    fraction : float
        Float between (0, 1] to subsample data.
    copy : bool
        Return view of AnnData if False, return copy if True. Default False.
    Returns
    -------
    AnnData
        Returns subsampled view of original AnnData object.
    """
    keep = (
        data.groupby("cell")
        .apply(lambda df: df.sample(frac=fraction))
        .index.droplevel(0)
    )
     
    return data.loc[keep]

