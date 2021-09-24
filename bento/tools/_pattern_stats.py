import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as sfm
from patsy import PatsyError
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .._utils import PATTERN_NAMES, track


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


@track
def pattern_diff(data, phenotype=None, copy=False):
    """Gene-wise test for differential localization across phenotype of interest.

    Parameters
    ----------
    data : AnnData
        Anndata formatted spatial data.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs_names.
    copy : bool, optional
        Return view of AnnData if False, return copy if True. By default False.
    """
    adata = data.copy() if copy else data

    if phenotype not in adata.obs.columns:
        raise ValueError("Phenotype is invalid.")

    detected = np.ones(data.shape)
    for c in PATTERN_NAMES:
        detected = detected & ~data.to_df(c).isna()

    gene_fc_stats = []
    for c in PATTERN_NAMES:
        a = data.var[f"{c}_count"].to_frame()
        a["n_cells_detected"] = detected[a.index].sum()

        a.columns = [
            "pattern_count",
            "n_cells_detected",
        ]

        a["pattern"] = c

        group_n_cells = pd.DataFrame(detected).groupby(data.obs[phenotype]).sum().T
        group_n_cells.columns += "_n_cells"
        a = a.join(group_n_cells)

        # save pattern frequency to new column, one for each group
        group_freq = (
            data.to_df(c)
            .replace("none", np.nan)
            .astype(float)
            .groupby(data.obs[phenotype])
            .sum()
            .T
        )

        all_mean = group_freq.mean(axis=1)
        all_mean.name = f"{phenotype}_mean_pcount"
        a = a.join(all_mean)

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
            log2fc = np.log2((group_col + 1) / (rest_mean + 1))

            # Average rank across log2fc, # of cells detected in group, # of cells deteted in rest
            ranks = (
                pd.concat([all_mean, log2fc], axis=1)
                .apply(lambda col: col.rank(ascending=False, method="min"), axis=0)
                .mean(axis=1)
            )
            results = pd.DataFrame(
                [ranks, log2fc],
                index=[f"{group_name}_rank", f"{group_name}_log2fc"],
            ).T
            return results

        # log2fc of group / mean(rest)
        for g in group_freq.columns:
            a = a.join(log2fc(group_freq[g]))

        gene_fc_stats.append(a)

    gene_fc_stats = pd.concat(gene_fc_stats)
    gene_fc_stats = gene_fc_stats.reset_index()
    adata.uns[f"{phenotype}_dp"] = gene_fc_stats

    return adata if copy else None


def _spots_diff_gene(cell_by_pattern, phenotype, phenotype_vector):
    """Perform pairwise comparison between groupby and every class.
    Parameters
    ----------
    Returns
    -------
    DataFrame
        Differential localization test results. [# of patterns, ]
    """
    results = []

    # One hot encode categories
    group_dummies = pd.get_dummies(pd.Series(phenotype_vector))
    group_dummies.columns = [f"{phenotype}_{g}" for g in group_dummies.columns]
    group_names = group_dummies.columns.tolist()
    group_data = pd.concat([cell_by_pattern, group_dummies], axis=1)
    group_data.columns = group_data.columns.astype(str)

    # Perform one group vs rest logistic regression
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
                "gene",
                "phenotype",
            ]
            r = r.reset_index().rename({"index": "pattern"}, axis=1)

            results.append(r)
        except (np.linalg.LinAlgError, PerfectSeparationError, PatsyError):
            continue

    return pd.concat(results) if len(results) > 0 else None


def spots_diff(
    data, phenotype=None, continuous=False, combined=False, n_cores=1, copy=False
):
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

    # need patterns as columns, cells and genes as index
    pattern_df = []
    for p in PATTERN_NAMES:
        p_df = pd.DataFrame(
            adata.layers[p], index=adata.obs_names, columns=adata.var_names
        )
        p_df = p_df.reset_index().melt(id_vars="cell")
        p_df["pattern"] = p
        pattern_df.append(p_df)

    pattern_df = pd.concat(pattern_df)
    pattern_df = pattern_df.pivot(
        index=["cell", "gene"], columns="pattern", values="value"
    ).reset_index()

    phenotype_vector = adata.obs[phenotype].tolist()

    diff_output = pattern_df.groupby("gene").apply(
        lambda gp: _spots_diff_gene(gp, phenotype, phenotype_vector)
    )

    # Format pattern column
    diff_output = pd.concat(diff_output)

    # FDR correction
    results_adj = []
    for _, df in diff_output.groupby("pattern"):
        df["padj"] = multipletests(df["pvalue"], method="hs")[1]
        results_adj.append(df)

    results_adj = pd.concat(results_adj).dropna()

    # -log10pvalue, padj
    results_adj["-log10p"] = -np.log10(results_adj["pvalue"].astype(np.float32))
    results_adj["-log10padj"] = -np.log10(results_adj["padj"].astype(np.float32))

    # Cap significance values
    results_adj.loc[results_adj["-log10p"] > 20, "-log10p"] = 20
    results_adj.loc[results_adj["-log10padj"] > 12, "-log10padj"] = 12

    # Sort results
    results_adj = results_adj.sort_values("pvalue")

    # Save back to AnnData
    adata.uns[f"diff_{phenotype}"] = results_adj

    return adata if copy else None
