import pickle

import bento
import numpy as np
import pandas as pd
import statsmodels.formula.api as sfm
from patsy import PatsyError
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from tqdm.auto import tqdm
from anndata import AnnData
from .._utils import track
from .._constants import PATTERN_NAMES, PATTERN_FEATURES

tqdm.pandas()


@track
def lp(data: AnnData, groupby: str = "gene", copy: bool = False):
    """Predict transcript subcellular localization patterns.
    Patterns include: cell edge, cytoplasmic, nuclear edge, nuclear, none

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    groupby : str or list of str, optional (default: None)
        Key in `data.uns['points'] to groupby, by default None. Always treats each cell separately
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : AnnData
        .uns['lp']: DataFrame
            Localization pattern indicator matrix.
        .uns['lpp']: DataFrame
            Localization pattern probabilities.
    """
    adata = data.copy() if copy else data

    if isinstance(groupby, str):
        groupby = [groupby]

    # Load trained model
    model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models"
    model = pickle.load(open(f"{model_dir}/rf_calib_20220514.pkl", "rb"))

    # Compatibility with newer versions of scikit-learn
    for cls in model.calibrated_classifiers_:
        cls.estimator = cls.base_estimator

    # Compute features
    feature_key = f"cell_{'_'.join(groupby)}_features"
    if feature_key not in adata.uns.keys() or not all(
        f in adata.uns[feature_key].columns for f in PATTERN_FEATURES
    ):
        bento.tl.analyze_points(
            adata,
            "cell_shape",
            ["proximity", "asymmetry", "ripley", "point_dispersion_norm"],
            groupby=groupby,
        )
        bento.tl.analyze_points(
            adata,
            "nucleus_shape",
            ["proximity", "asymmetry", "shape_dispersion_norm"],
            groupby=groupby,
        )

    X_df = adata.uns[feature_key][PATTERN_FEATURES]

    pattern_prob = pd.DataFrame(
        model.predict_proba(X_df.values),
        columns=PATTERN_NAMES,
    )
    pattern_prob.index = adata.uns[feature_key].set_index(["cell", *groupby]).index
    thresholds = [0.45300, 0.43400, 0.37900, 0.43700, 0.50500]

    indicator_df = (pattern_prob >= thresholds).replace({True: 1, False: 0})

    adata.uns["lp"] = indicator_df.reset_index()
    adata.uns["lpp"] = pattern_prob.reset_index()
    return adata if copy else None


@track
def lp_stats(data: AnnData, copy: bool = False):
    """Computes frequencies of localization patterns across cells and genes.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    copy : bool
        Whether to return a copy of the AnnData object. Default False.
    Returns
    -------
    adata : AnnData
        .uns['lp_stats']: DataFrame of localization pattern frequencies.
    """
    adata = data.copy() if copy else data

    lp = adata.uns["lp"]

    cols = lp.columns
    groupby = list(cols[~cols.isin(PATTERN_NAMES)])
    groupby.remove("cell")

    g_pattern_counts = lp.groupby(groupby).apply(lambda df: df[PATTERN_NAMES].sum())
    adata.uns["lp_stats"] = g_pattern_counts

    return adata if copy else None


def _lp_logfc(data, phenotype=None):
    """Compute pairwise log2 fold change of patterns between groups in phenotype.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs.columns.

    Returns
    -------
    gene_fc_stats : DataFrame
        log2 fold change of patterns between groups in phenotype.
    """
    stats = data.uns["lp_stats"]

    if phenotype not in data.obs.columns:
        raise ValueError("Phenotype is invalid.")

    phenotype_vector = data.obs[phenotype]

    pattern_df = data.uns["lp"].copy()
    groups_name = stats.index.name
    pattern_df[["cell", groups_name]] = data.uns[f"cell_{groups_name}_features"][
        ["cell", groups_name]
    ]

    gene_fc_stats = []
    for c in PATTERN_NAMES:

        # save pattern frequency to new column, one for each group
        group_freq = (
            pattern_df.pivot(index="cell", columns=groups_name, values=c)
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
            results["phenotype"] = group_name
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


def _lp_diff_gene(cell_by_pattern, phenotype_vector):
    """Perform pairwise comparison between groupby and every class.

    Parameters
    ----------
    cell_by_pattern : DataFrame
        Cell by pattern matrix.
    phenotype_vector : Series
        Series of cell groupings.

    Returns
    -------
    DataFrame
        Differential localization test results. [# of patterns, ]
    """
    cell_by_pattern = cell_by_pattern.dropna().reset_index(drop=True)

    # One hot encode categories
    group_dummies = pd.get_dummies(pd.Series(phenotype_vector))
    # group_dummies.columns = [f"{phenotype}_{g}" for g in group_dummies.columns]
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
def lp_diff(
    data: AnnData, phenotype: str = None, continuous: bool = False, copy: bool = False
):
    """Gene-wise test for differential localization across phenotype of interest.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs.columns.
    continuous : bool
        Whether the phenotype is continuous or categorical. By default False.
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : AnnData
        Spatial formatted AnnData object.
        .uns['diff_{phenotype}'] : DataFrame
            Long DataFrame with differential localization test results across phenotype groups.
    """
    adata = data.copy() if copy else data

    stats = adata.uns["lp_stats"]

    # Retrieve cell phenotype
    phenotype_vector = adata.obs[phenotype].tolist()

    # TODO untested/incomplete
    if continuous:
        pattern_dfs = {}

        # Compute correlation for each point group along cells
        for p in PATTERN_NAMES:
            p_labels = adata.uns["lp"][p]
            groups_name = stats.index.name
            p_labels[["cell", groups_name]] = adata.uns[f"cell_{groups_name}_features"][
                ["cell", groups_name]
            ]
            p_labels = p_labels.pivot(index="cell", columns="gene", values=p)
            p_corr = p_df.corrwith(phenotype_vector, drop=True)
            pattern_dfs[p] = p_labels

    else:
        # [Sample by patterns] where sample id = [cell, group] pair
        pattern_df = adata.uns["lp"].copy()
        groups_name = stats.index.name
        pattern_df[["cell", groups_name]] = adata.uns[f"cell_{groups_name}_features"][
            ["cell", groups_name]
        ]

        diff_output = (
            pattern_df.groupby(groups_name)
            .progress_apply(lambda gp: _lp_diff_gene(gp, phenotype_vector))
            .reset_index()
        )

    # FDR correction
    diff_output["padj"] = diff_output["pvalue"] * diff_output[groups_name].nunique()

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
        results.set_index([groups_name, "pattern", "phenotype"])
        .join(log2fc_stats.set_index([groups_name, "pattern", "phenotype"]))
        .reset_index()
    )

    # Sort results
    results = results.sort_values("pvalue")

    # Save back to AnnData
    adata.uns[f"diff_{phenotype}"] = results

    return adata if copy else None
