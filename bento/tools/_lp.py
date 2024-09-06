from typing import List, Optional, Union
import pickle
import warnings

warnings.filterwarnings("ignore")

import bento
import numpy as np
import pandas as pd
import statsmodels.formula.api as sfm
from pandas.api.types import is_numeric_dtype
from patsy import PatsyError
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from tqdm.auto import tqdm
from spatialdata._core.spatialdata import SpatialData

from .._constants import PATTERN_NAMES, PATTERN_THRESHOLDS_CALIB
from .._utils import get_instance_key, get_feature_key, get_points, set_points_metadata

tqdm.pandas()


def lp(
    sdata: SpatialData,
    instance_key: str = "cell_boundaries",
    nucleus_key: str = "nucleus_boundaries",
    groupby: Union[str, List[str]] = "feature_name",
    num_workers=1,
    recompute=False,
):
    """Predict transcript subcellular localization patterns.
    Patterns include: cell edge, cytoplasmic, nuclear edge, nuclear, none

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object

    groupby : str or list of str
        Key in `sdata.points[points_key] to groupby, by default None. Always treats each cell separately

    Returns
    -------
    sdata : SpatialData
        .tables["table"].uns['lp']
            Localization pattern indicator matrix.
        .tables["table"].uns['lpp']
            Localization pattern probabilities.
    """

    if isinstance(groupby, str):
        groupby = [groupby]

    pattern_features = [  # Do not change order of features!
        f"{instance_key}_inner_proximity",
        f"{nucleus_key}_inner_proximity",
        f"{nucleus_key}_outer_proximity",
        f"{instance_key}_inner_asymmetry",
        f"{nucleus_key}_inner_asymmetry",
        f"{nucleus_key}_outer_asymmetry",
        "l_max",
        "l_max_gradient",
        "l_min_gradient",
        "l_monotony",
        "l_half_radius",
        "point_dispersion_norm",
        f"{nucleus_key}_dispersion_norm",
    ]

    # Compute features
    feature_key = f"{instance_key}_{'_'.join(groupby)}_features"
    if (
        feature_key not in sdata.tables["table"].uns.keys()
        or not all(
            f in sdata.tables["table"].uns[feature_key].columns
            for f in pattern_features
        )
        or recompute
    ):
        bento.tl.analyze_points(
            sdata,
            instance_key,
            ["proximity", "asymmetry", "ripley", "point_dispersion_norm"],
            groupby=groupby,
            recompute=recompute,
            num_workers=num_workers,
        )
        bento.tl.analyze_points(
            sdata,
            nucleus_key,
            ["proximity", "asymmetry", "shape_dispersion_norm"],
            groupby=groupby,
            recompute=recompute,
            num_workers=num_workers,
        )

    X_df = sdata.tables["table"].uns[feature_key][pattern_features]

    # Save which samples have nan feature values
    invalid_samples = X_df.isna().any(axis=1)

    # Load trained model
    model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models"
    model = pickle.load(open(f"{model_dir}/rf_calib_20220514.pkl", "rb"))

    # Compatibility with newer versions of scikit-learn
    for cls in model.calibrated_classifiers_:
        cls.estimator = cls.base_estimator

    # Predict patterns
    pattern_prob = pd.DataFrame(
        model.predict_proba(X_df.values),
        columns=PATTERN_NAMES,
    )

    # Add cell and groupby identifiers
    pattern_prob.index = (
        sdata.tables["table"].uns[feature_key].set_index([instance_key, *groupby]).index
    )

    # Set to no class if sample had nan feature values
    pattern_prob.loc[:, invalid_samples] = 0

    # Threshold probabilities to get indicator matrix
    indicator_df = (pattern_prob >= PATTERN_THRESHOLDS_CALIB).astype(np.uint8)

    lp_df = indicator_df.reset_index()[PATTERN_NAMES]

    invalid_patterns = (lp_df == 0).all(axis=0)
    invalid_patterns = invalid_patterns[invalid_patterns].index.tolist()
    n_invalid_cells = (lp_df == 0).all(axis=1).sum()

    if len(invalid_patterns) > 0:
        warnings.warn(
            f"Patterns {invalid_patterns} are not present in any cells.", UserWarning
        )

    if n_invalid_cells > 0:
        warnings.warn(
            f"Patterns not detected in {n_invalid_cells} / {lp_df.shape[0]} cells.",
            UserWarning,
        )

    sdata.tables["table"].uns["lp"] = indicator_df.reset_index()
    sdata.tables["table"].uns["lpp"] = pattern_prob.reset_index()

    lp_stats(sdata)


def lp_stats(sdata: SpatialData):
    """Computes frequencies of localization patterns across cells and genes.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    instance_key : str
        cell boundaries instance key

    Returns
    -------
    sdata : SpatialData
        .tables["table"].uns['lp_stats']: DataFrame of localization pattern frequencies.
    """
    instance_key = get_instance_key(sdata)
    feature_key = get_feature_key(sdata)
    lp = sdata.table.uns["lp"]

    cols = lp.columns
    groupby = list(cols[~cols.isin(PATTERN_NAMES)])
    groupby.remove(instance_key)

    g_pattern_counts = lp.groupby(groupby, observed=True).apply(
        lambda df: df[PATTERN_NAMES].sum().astype(int)
    )
    sdata.table.uns["lp_stats"] = g_pattern_counts

    lpp = sdata["table"].uns["lpp"]
    top_pattern = lpp[[instance_key, feature_key]]
    top_pattern["pattern"] = (
        lpp[PATTERN_NAMES].mask(lp[PATTERN_NAMES] == 0).idxmax(axis=1)
    )

    points = get_points(sdata)
    top_pattern_long = points.set_index(["cell_boundaries", "feature_name"]).merge(
        top_pattern,
        on=["cell_boundaries", "feature_name"],
        how="left",
        suffixes=("", "_y"),
    )["pattern"]
    set_points_metadata(sdata, "transcripts", top_pattern_long, "pattern")


def _lp_logfc(sdata, instance_key, phenotype=None):
    """Compute pairwise log2 fold change of patterns between groups in phenotype.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object.
    instance_key: str
        cell boundaries instance key
    phenotype : str
        Variable grouping cells for differential analysis. Must be in sdata.shapes["cell_boundaries"].columns.

    Returns
    -------
    gene_fc_stats : DataFrame
        log2 fold change of patterns between groups in phenotype.
    """
    stats = sdata.tables["table"].uns["lp_stats"]

    if phenotype not in sdata.shapes[instance_key].columns:
        raise ValueError("Phenotype is invalid.")

    phenotype_vector = sdata.shapes[instance_key][phenotype]

    pattern_df = sdata.tables["table"].uns["lp"].copy()
    groups_name = stats.index.name

    gene_fc_stats = []
    for c in PATTERN_NAMES:
        # save pattern frequency to new column, one for each group
        group_freq = (
            pattern_df.pivot(index=instance_key, columns=groups_name, values=c)
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


def _lp_diff_gene(cell_by_pattern, phenotype_series, instance_key):
    """Perform pairwise comparison between groupby and every class.

    Parameters
    ----------
    cell_by_pattern : DataFrame
        Cell by pattern matrix.
    phenotype_series : Series
        Series of cell groupings.
    instance_key : str
        cell boundaries instance key

    Returns
    -------
    DataFrame
        Differential localization test results. [# of patterns, ]
    """
    cell_by_pattern = cell_by_pattern.dropna().reset_index(drop=True)

    # One hot encode categories
    group_dummies = pd.get_dummies(phenotype_series)
    group_names = group_dummies.columns.tolist()
    group_data = cell_by_pattern.set_index(instance_key).join(
        group_dummies, how="inner"
    )
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
            r = r.reset_index()

            results.append(r)

        # Append empty Dataframe if empty groups or patterns missing from groups
        except (
            np.linalg.LinAlgError,
            ValueError,
            PerfectSeparationError,
            PatsyError,
        ):
            r = pd.DataFrame(
                columns=[
                    "index",
                    "dy/dx",
                    "Std. Err.",
                    "z",
                    "Pr(>|z|)",
                    "Conf. Int. Low",
                    "Cont. Int. Hi.",
                    "phenotype",
                ]
            )
            results.append(r)

    results = pd.concat(results)

    col_map = {
        "index": "pattern",
        "dy/dx": "dy/dx",
        "Std. Err.": "std_err",
        "z": "z",
        "Pr(>|z|)": "pvalue",
        "Conf. Int. Low": "ci_low",
        "Cont. Int. Hi.": "ci_high",
        "phenotype": "phenotype",
    }
    results = results.rename(columns=col_map)

    return results


def lp_diff_discrete(
    sdata: SpatialData, instance_key: str = "cell_boundaries", phenotype: str = None
):
    """Gene-wise test for differential localization across phenotype of interest.

    Scenarios:
    Missing patterns within phenotype groupings
    Solution:
    - Warn user about missing patterns
    - Remove missing patterns from analysis
    - Return results with missing patterns removed

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    instance_key : str
        cell boundaries instance key.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in sdata.shape["cell_boundaries].columns.

    Returns
    -------
    sdata : SpatialData
        .tables["table"].uns['diff_{phenotype}']
            Long DataFrame with differential localization test results across phenotype groups.
    """
    lp_df = sdata.tables["table"].uns["lp"]

    invalid_patterns = (lp_df == 0).all(axis=0)
    invalid_patterns = invalid_patterns[invalid_patterns].index.tolist()
    n_invalid_cells = (lp_df == 0).all(axis=1).sum()

    if len(invalid_patterns) > 0:
        warnings.warn(
            f"Patterns {invalid_patterns} are not present in any cells.", UserWarning
        )

    if n_invalid_cells > 0:
        warnings.warn(
            f"Patterns not detected in {n_invalid_cells} / {lp_df.shape[0]} cells.",
            UserWarning,
        )

    lp_stats(sdata)
    stats = sdata.tables["table"].uns["lp_stats"]

    # Retrieve cell phenotype
    phenotype_series = sdata.shapes[instance_key][phenotype]
    if is_numeric_dtype(phenotype_series):
        warnings.simplefilter("always", UserWarning)
        warnings.warn(
            f"Phenotype dtype must not be numeric | dtype: {phenotype_series.dtype}"
        )
        warnings.filterwarnings("ignore")
        return

    # [Sample by patterns] where sample id = [cell, group] pair
    pattern_df = sdata.tables["table"].uns["lp"].copy()
    groups_name = stats.index.name

    diff_output = (
        pattern_df.groupby(groups_name, observed=True)
        .progress_apply(lambda gp: _lp_diff_gene(gp, phenotype_series, instance_key))
        .reset_index()
    )

    # FDR correction
    diff_output["padj"] = diff_output["pvalue"] * diff_output[groups_name].nunique()

    results = diff_output.dropna()

    # -log10pvalue, padj
    results["-log10p"] = -np.log10(results["pvalue"].astype(np.float32))
    results["-log10padj"] = -np.log10(results["padj"].astype(np.float32))

    # Cap significance values
    results.loc[results["-log10p"] == np.inf, "-log10p"] = results.loc[
        results["-log10p"] != np.inf
    ]["-log10p"].max()
    results.loc[results["-log10padj"] == np.inf, "-log10padj"] = results.loc[
        results["-log10padj"] != np.inf
    ]["-log10padj"].max()

    # Group-wise log2 fold change values
    log2fc_stats = _lp_logfc(sdata, instance_key, phenotype)

    # Join log2fc results to p value df
    results = (
        results.set_index([groups_name, "pattern", "phenotype"])
        .join(log2fc_stats.set_index([groups_name, "pattern", "phenotype"]))
        .reset_index()
    )

    # Sort results
    results = results.sort_values("pvalue").reset_index(drop=True)
    del results["level_1"]
    # Save back to SpatialData
    sdata.tables["table"].uns[f"diff_{phenotype}"] = results


def lp_diff_continuous(
    sdata: SpatialData, instance_key: str = "cell_boundaries", phenotype: str = None
):
    """Gene-wise test for differential localization across phenotype of interest.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    instance_key : str
        cell boundaries instance key.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in sdata.shape["cell_boundaries].columns.

    Returns
    -------
    sdata : SpatialData
        .tables["table"].uns['diff_{phenotype}']
            Long DataFrame with differential localization test results across phenotype groups.
    """
    stats = sdata.tables["table"].uns["lp_stats"]
    lpp = sdata.tables["table"].uns["lpp"]
    # Retrieve cell phenotype
    phenotype_series = sdata.shapes[instance_key][phenotype]

    pattern_dfs = {}
    # Compute correlation for each point group along cells
    for p in PATTERN_NAMES:
        groups_name = stats.index.name
        p_labels = lpp.pivot(index=instance_key, columns=groups_name, values=p)
        p_corr = p_labels.corrwith(phenotype_series, axis=0, drop=True)

        pattern_df = pd.DataFrame(p_corr).reset_index(drop=False)
        pattern_df.insert(loc=1, column="pattern", value=p)
        pattern_df = pattern_df.rename(columns={0: "pearson_correlation"})
        pattern_dfs[p] = pattern_df

    # Concatenate all pattern_dfs into one
    pattern_dfs = (
        pd.concat(pattern_dfs.values(), ignore_index=True)
        .sort_values(by=["pearson_correlation"], ascending=False)
        .reset_index(drop=True)
    )

    pattern_dfs = pattern_dfs.loc[~pattern_dfs["pearson_correlation"].isna()]
    sdata.tables["table"].uns[f"diff_{phenotype}"] = pattern_dfs
