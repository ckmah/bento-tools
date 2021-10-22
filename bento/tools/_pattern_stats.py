import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as sfm
from patsy import PatsyError
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from dask import dataframe as dd
from dask.diagnostics import ProgressBar

from .._utils import PATTERN_NAMES, track
from tqdm.auto import tqdm

tqdm.pandas()


@track
def pattern_stats(data, copy=False):
    """Computes frequencies of input layer values across cells and across genes.
    Assumes layer values are categorical.

    Parameters
    ----------
    data : [type]
        [description]
    layer : [type]
        [description]
    copy : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy() if copy else data

    detected = np.ones(adata.shape)

    for c in PATTERN_NAMES:
        detected = detected & ~data.to_df(c).isna()

    adata.var["n_detected"] = detected.sum().astype(int)
    adata.var[f"fraction_detected"] = (adata.var["n_detected"] / adata.n_obs).astype(
        float
    )

    adata.obs["n_detected"] = detected.sum(axis=1).astype(int)
    adata.obs[f"fraction_detected"] = (adata.obs["n_detected"] / adata.n_vars).astype(
        float
    )

    for c in PATTERN_NAMES:
        # Save frequencies across genes
        counts = adata.to_df(c)

        # Save frequencies across cells
        adata.var[f"{c}_count"] = counts.fillna(0).sum(axis=0).astype(int)
        adata.var[f"{c}_fraction"] = (
            adata.var[f"{c}_count"] / adata.var["n_detected"]
        ).astype(float)

        adata.obs[f"{c}_count"] = counts.fillna(0).sum(axis=1).astype(int)
        adata.obs[f"{c}_fraction"] = (
            adata.obs[f"{c}_count"] / adata.obs["n_detected"]
        ).astype(float)

    return adata if copy else None


def _pattern_log2fc(data, phenotype=None):
    """Compute pairwise log2 fold change of patterns between groups in phenotype.

    Parameters
    ----------
    data : AnnData
        Anndata formatted spatial data.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs_names.
    copy : bool, optional
        Return view of AnnData if False, return copy if True. By default False.
    """

    if phenotype not in data.obs.columns:
        raise ValueError("Phenotype is invalid.")

    phenotype_vector = data.obs[phenotype]

    gene_fc_stats = []
    for c in PATTERN_NAMES:

        # save pattern frequency to new column, one for each group
        group_freq = (
            data.to_df(c)
            .replace("none", np.nan)
            .astype(float)
            .groupby(phenotype_vector)
            .sum()
            .T
        )

        def log2fc(group_col):
            """
            Return
            ------
            log2fc : int
                log2fc of group_count / rest, pseudocount of 1
            group_count : int
            rest_mean_count : int
            """
            group_name = group_col.name
            rest_cols = group_freq.columns[group_freq.columns != group_name]
            rest_mean = group_freq[rest_cols].mean(axis=1)

            # log2fc(group frequency / mean other frequency)
            log2fc = np.log2((group_col + 1) / (rest_mean + 1))
            results = log2fc.to_frame("log2fc")
            results["phenotype"] = f"{phenotype}_{group_name}"
            return results

        # Compute log2fc of group / mean(rest) for each group
        p_fc = []
        for g in group_freq.columns:
            p_fc.append(log2fc(group_freq[g]))

        p_fc = pd.concat(p_fc)
        p_fc["pattern"] = c   
        
        gene_fc_stats.append(p_fc)

    gene_fc_stats = pd.concat(gene_fc_stats)
    
    gene_fc_stats = gene_fc_stats.reset_index()

    return gene_fc_stats


def _pattern_diff_gene(cell_by_pattern, phenotype, phenotype_vector):
    """Perform pairwise comparison between groupby and every class.
    Parameters
    ----------
    Returns
    -------
    DataFrame
        Differential localization test results. [# of patterns, ]
    """
    cell_by_pattern = cell_by_pattern.dropna().reset_index(drop=True)

    # One hot encode categories
    group_dummies = pd.get_dummies(pd.Series(phenotype_vector))
    group_dummies.columns = [f"{phenotype}_{g}" for g in group_dummies.columns]
    group_names = group_dummies.columns.tolist()
    group_data = pd.concat([cell_by_pattern, group_dummies], axis=1)
    group_data.columns = group_data.columns.astype(str)

    # Perform one group vs rest logistic regression
    results = []
    for g in group_names:
        try:
            res = sfm.logit(
                formula=f"{g} ~ {' + '.join(PATTERN_NAMES)}", data=group_data
            ).fit(disp=0)

            # Look at marginal effect of each pattern coefficient
            r = res.get_margeff(dummy=True).summary_frame()
            r["phenotype"] = g

            r.columns = [
                "dy/dx",
                "std_err",
                "z",
                "pvalue",
                "ci_low",
                "ci_high",
                "phenotype",
            ]
            r = r.reset_index().rename({"index": "pattern"}, axis=1)

            results.append(r)
        #
        except (
            np.linalg.LinAlgError,
            ValueError,
            PerfectSeparationError,
            PatsyError,
        ) as e:
            continue

    if len(results) > 0:
        results = pd.concat(results)

    return results if len(results) > 0 else None


@track
def pattern_diff(data, phenotype=None, continuous=False, min_cells=10, copy=False):
    """Gene-wise test for differential localization across phenotype of interest.
    Parameters
    ----------
    data : AnnData
        Anndata formatted spatial data.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs_names.
    continuous : bool
        Whether the phenotype is continuous or categorical. By default False.
    n_cores : int, optional
        cores used for multiprocessing, by default 1
    copy : bool, optional
        Return view of AnnData if False, return copy if True. By default False.
    """
    adata = data.copy() if copy else data

    # Note which samples were detected and classified
    detected = np.ones(data.shape)
    for c in PATTERN_NAMES:
        detected = detected & ~data.to_df(c).isna()

    # Only look at genes detected in >= min_cells
    valid_genes = detected.sum(axis=0) >= min_cells
    print(f"{sum(valid_genes)} genes detected in at least {min_cells} cells.")

    # Retrieve cell phenotype
    phenotype_vector = adata.obs[phenotype].tolist()

    if continuous:
        pattern_dfs = {}
        for p in PATTERN_NAMES:
            p_df = adata.to_df(p).loc[:, valid_genes]
            p_corr = p_df.corrwith(phenotype_vector, drop=True)
            pattern_dfs[p] = p_df
            
            
    else:
        # Load and flatten pattern layers
        pattern_df = []
        for p in PATTERN_NAMES:
            p_df = adata.to_df(p).loc[:, valid_genes].reset_index().melt(id_vars="cell")
            p_df["pattern"] = p
            pattern_df.append(p_df)

        # [Sample by patterns] where sample id = [cell, gene] pair
        pattern_df = pd.concat(pattern_df)
        pattern_df = pattern_df.pivot(
            index=["cell", "gene"], columns="pattern", values="value"
        ).reset_index()
        
        # Fit logit for each gene
        meta = {
            "pattern": str,
            "dy/dx": float,
            "std_err": float,
            "z": float,
            "pvalue": float,
            "ci_low": float,
            "ci_high": float,
            "phenotype": str,
        }

    #     diff_output = pattern_df.groupby("gene").progress_apply(
    #         lambda gp: _pattern_diff_gene(gp, phenotype, phenotype_vector)
    #     )

        with ProgressBar():
            diff_output = (
                dd.from_pandas(pattern_df, chunksize=100)
                .groupby("gene")
                .apply(
                    lambda gp: _pattern_diff_gene(gp, phenotype, phenotype_vector), meta=meta
                )
                .reset_index()
                .compute()
            )

    # Format pattern column
    # diff_output = pd.concat(diff_output)

    # FDR correction
    diff_output["padj"] = diff_output["pvalue"] * diff_output["gene"].nunique()

    results = diff_output.dropna()

    # -log10pvalue, padj
    results["-log10p"] = -np.log10(results["pvalue"].astype(np.float32))
    results["-log10padj"] = -np.log10(results["padj"].astype(np.float32))

    # Cap significance values
    results.loc[results["-log10p"] > 20, "-log10p"] = 20
    results.loc[results["-log10padj"] > 12, "-log10padj"] = 12

    # Group-wise log2 fold change values
    log2fc_stats = _pattern_log2fc(adata, phenotype)

    # Join log2fc results to p value df
    results = (
        results.set_index(["gene", "pattern", "phenotype"])
        .join(log2fc_stats.set_index(["gene", "pattern", "phenotype"]))
        .reset_index()
    )
    
    # Sort results
    results = results.sort_values("pvalue")

    # Save back to AnnData
    adata.uns[f"diff_{phenotype}"] = results

    return adata if copy else None
