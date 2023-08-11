import pickle
import warnings

warnings.filterwarnings("ignore")

import bento
import numpy as np
import pandas as pd
import statsmodels.formula.api as sfm
from patsy import PatsyError
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from tqdm.auto import tqdm
from spatialdata._core.spatialdata import SpatialData

#from .._utils import track
from .._constants import PATTERN_NAMES, PATTERN_FEATURES

tqdm.pandas()

# @track
def lp(sdata: SpatialData, groupby: str = "feature_name", copy: bool = False):
    """Predict transcript subcellular localization patterns.
    Patterns include: cell edge, cytoplasmic, nuclear edge, nuclear, none

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object
    groupby : str or list of str, optional (default: None)
        Key in `data.points['transcripts'] to groupby, by default None. Always treats each cell separately
    copy : bool
        Return a copy of `sdata` instead of writing to data, by default False.

    Returns
    -------
    sdata : SpatialData
        .table.uns['lp']: DataFrame
            Localization pattern indicator matrix.
        .table.uns['lpp']: DataFrame
            Localization pattern probabilities.
    """

    if isinstance(groupby, str):
        groupby = [groupby]

    # Compute features
    feature_key = f"cell_{'_'.join(groupby)}_features"
    if feature_key not in sdata.table.uns.keys() or not all(
        f in sdata.table.uns[feature_key].columns for f in PATTERN_FEATURES
    ):
        bento.tl.analyze_points(
            sdata,
            "cell_boundaries",
            ["proximity", "asymmetry", "ripley", "point_dispersion_norm"],
            groupby=groupby,
            recompute=True,
        )
        bento.tl.analyze_points(
            sdata,
            "nucleus_boundaries",
            ["proximity", "asymmetry", "shape_dispersion_norm"],
            groupby=groupby,
            recompute=True,
        )

    X_df = sdata.table.uns[feature_key][PATTERN_FEATURES]

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
    pattern_prob.index = sdata.table.uns[feature_key].set_index(["cell", *groupby]).index

    # Threshold probabilities to get indicator matrix
    thresholds = [0.45300, 0.43400, 0.37900, 0.43700, 0.50500]
    indicator_df = (pattern_prob >= thresholds).replace({True: 1, False: 0})

    sdata.table.uns["lp"] = indicator_df.reset_index()
    sdata.table.uns["lpp"] = pattern_prob.reset_index()

#@track
def lp_stats(sdata: SpatialData, copy: bool = False):
    """Computes frequencies of localization patterns across cells and genes.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object.
    copy : bool
        Whether to return a copy of the SpatialData object. Default False.
    Returns
    -------
    sdata : SpatialData
        .table.uns['lp_stats']: DataFrame of localization pattern frequencies.
    """
    lp = sdata.table.uns["lp"]

    cols = lp.columns
    groupby = list(cols[~cols.isin(PATTERN_NAMES)])
    groupby.remove("cell")

    g_pattern_counts = lp.groupby(groupby).apply(lambda df: df[PATTERN_NAMES].sum().astype(int))
    sdata.table.uns["lp_stats"] = g_pattern_counts

def _lp_logfc(sdata, phenotype=None):
    """Compute pairwise log2 fold change of patterns between groups in phenotype.

    Parameters
    ----------
    data : SpatialData
        Spatial formatted SpatialData object.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in sdata.shapes["cell_boundaries"].columns.

    Returns
    -------
    gene_fc_stats : DataFrame
        log2 fold change of patterns between groups in phenotype.
    """
    stats = sdata.table.uns["lp_stats"]

    if phenotype not in sdata.shapes["cell_boundaries"].columns:
        raise ValueError("Phenotype is invalid.")

    phenotype_vector = sdata.shapes["cell_boundaries"][phenotype]

    pattern_df = sdata.table.uns["lp"].copy()
    groups_name = stats.index.name
    '''pattern_df[["cell", groups_name]] = data.uns[f"cell_{groups_name}_features"][
        ["cell", groups_name]
    ]'''

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

def _lp_diff_gene(cell_by_pattern, phenotype_series):
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
    group_dummies = pd.get_dummies(phenotype_series)
    group_names = group_dummies.columns.tolist()
    group_data = cell_by_pattern.set_index("cell").join(group_dummies, how='inner')
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

#@track
def lp_diff_discrete(
    sdata: SpatialData, phenotype: str = None, copy: bool = False
):
    """Gene-wise test for differential localization across phenotype of interest.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs.columns.
    copy : bool
        Return a copy of `sdata` instead of writing to data, by default False.

    Returns
    -------
    sdata : SpatialData
        Spatial formatted SpatialData object.
        .table.uns['diff_{phenotype}'] : DataFrame
            Long DataFrame with differential localization test results across phenotype groups.
    """
    stats = sdata.table.uns["lp_stats"]

    # Retrieve cell phenotype
    phenotype_series = sdata.shapes["cell_boundaries"][phenotype]

    # [Sample by patterns] where sample id = [cell, group] pair
    pattern_df = sdata.table.uns["lp"].copy()
    groups_name = stats.index.name

    diff_output = (
        pattern_df.groupby(groups_name)
        .progress_apply(lambda gp: _lp_diff_gene(gp, phenotype_series))
        .reset_index()
    )

    # FDR correction
    diff_output["padj"] = diff_output["pvalue"] * diff_output[groups_name].nunique()

    results = diff_output.dropna()

    # -log10pvalue, padj
    results["-log10p"] = -np.log10(results["pvalue"].astype(np.float32))
    results["-log10padj"] = -np.log10(results["padj"].astype(np.float32))

    '''# Cap significance values
    results.loc[results["-log10p"] > 20, "-log10p"] = 20
    results.loc[results["-log10padj"] > 12, "-log10padj"] = 12'''

    # Group-wise log2 fold change values
    log2fc_stats = _lp_logfc(sdata, phenotype)

    # Join log2fc results to p value df
    results = (
        results.set_index([groups_name, "pattern", "phenotype"])
        .join(log2fc_stats.set_index([groups_name, "pattern", "phenotype"]))
        .reset_index()
    )

    # Sort results
    results = results.sort_values("pvalue")
    del results["level_1"]
    # Save back to SpatialData
    sdata.table.uns[f"diff_{phenotype}"] = results
