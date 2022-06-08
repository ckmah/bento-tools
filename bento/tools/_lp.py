import pickle

import bento
import numpy as np
import pandas as pd
import statsmodels.formula.api as sfm
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from patsy import PatsyError
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from tqdm.auto import tqdm

from .._utils import PATTERN_NAMES, PATTERN_PROBS, track
from ..preprocessing import get_layers

tqdm.pandas()


@track
def lp(data, min_count=5, copy=False):
    """Predict transcript subcellular localization patterns.
    Patterns include: cell edge, cytoplasmic, nuclear edge, nuclear, none

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    min_count : int
        Minimum expression count per sample; otherwise ignore sample
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    Depending on `copy`, returns or updates `adata.layers` with the
    `'cell_edge'`, `'cytoplasm'`, `'none'`, `'nuclear'`, and `'nuclear_edge'`
    fields for their respective localization pattern labels.
    """
    adata = data.copy() if copy else data

    # Compute features if missing TODO currently recomputes everything
    if not all(f in data.layers.keys() for f in PATTERN_MODEL_FEATURE_NAMES):
        features = [
            "cell_proximity",
            "nucleus_proximity",
            "cell_asymmetry",
            "nucleus_asymmetry",
            "ripley_stats",
            "point_dispersion",
            "nucleus_dispersion",
        ]
        bento.tl.analyze_samples(data, features, chunksize=100)

    X_df = get_layers(adata, PATTERN_MODEL_FEATURE_NAMES, min_count)

    model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models"
    model = pickle.load(open(f"{model_dir}/rf_calib_20220514.pkl", "rb"))

    pattern_prob = pd.DataFrame(
        model.predict_proba(X_df.values), index=X_df.index, columns=PATTERN_NAMES
    )
    thresholds = [0.45300, 0.43400, 0.37900, 0.43700, 0.50500]
    # Save each pattern to adata
    for p, pp, thresh in tqdm(
        zip(PATTERN_NAMES, PATTERN_PROBS, thresholds), total=len(PATTERN_NAMES)
    ):
        indicator_df = (
            (pattern_prob >= thresh)
            .reset_index()
            .pivot(index="cell", columns="gene", values=p)
            .replace({True: 1, False: 0})
            .reindex(index=adata.obs_names, columns=adata.var_names)
            .astype(float)
        )
        indicator_df.columns.name = "gene"

        prob_df = (
            pattern_prob.reset_index()
            .pivot(index="cell", columns="gene", values=p)
            .reindex(index=adata.obs_names, columns=adata.var_names)
            .astype(float)
        )

        # Save to adata.layers
        adata.layers[p] = indicator_df
        adata.layers[pp] = prob_df

    # Run without decorator
    lp_stats.__wrapped__(adata)

    return adata if copy else None


@track
def lp_stats(data, copy=False):
    """Computes frequencies of localization patterns across cells and genes.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.
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
        counts = adata.to_df(c)

        # Save frequencies across cells
        adata.var[f"{c}_count"] = counts.fillna(0).sum(axis=0).astype(int)
        adata.var[f"{c}_fraction"] = (adata.var[f"{c}_count"] / adata.n_obs).astype(
            float
        )

        # Save frequencies across genes
        adata.obs[f"{c}_count"] = counts.fillna(0).sum(axis=1).astype(int)
        adata.obs[f"{c}_fraction"] = (adata.obs[f"{c}_count"] / adata.n_vars).astype(
            float
        )

    return adata if copy else None


def _lp_logfc(data, phenotype=None):
    """Compute pairwise log2 fold change of patterns between groups in phenotype.

    Parameters
    ----------
    data : AnnData
        Anndata formatted spatial data.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs.columns.
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


def _lp_diff_gene(cell_by_pattern, phenotype, phenotype_vector):
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
def lp_diff(data, phenotype=None, continuous=False, min_cells=10, copy=False):
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
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.
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
        #         lambda gp: _lp_diff_gene(gp, phenotype, phenotype_vector)
        #     )

        with ProgressBar():
            diff_output = (
                dd.from_pandas(pattern_df, chunksize=100)
                .groupby("gene")
                .apply(
                    lambda gp: _lp_diff_gene(gp, phenotype, phenotype_vector), meta=meta
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
    log2fc_stats = _lp_logfc(adata, phenotype)

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


PATTERN_MODEL_FEATURE_NAMES = [
    "cell_inner_proximity",
    "nucleus_inner_proximity",
    "nucleus_outer_proximity",
    "cell_inner_asymmetry",
    "nucleus_inner_asymmetry",
    "nucleus_outer_asymmetry",
    "l_max",
    "l_max_gradient",
    "l_min_gradient",
    "l_monotony",
    "l_half_radius",
    "point_dispersion",
    "nucleus_dispersion",
]
