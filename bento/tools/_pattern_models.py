import pickle

import bento
import pandas as pd
import xgboost
from tqdm.auto import tqdm
from xgboost import XGBRFClassifier

from .._utils import PATTERN_NAMES, PATTERN_PROBS, track
from ._pattern_stats import pattern_stats

OneHotEncoder = None
LabelBinarizer = None

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

def pattern_features(data):
    models = [
        bento.tl.ShapeProximity("cell_shape"),
        bento.tl.ShapeProximity("nucleus_shape"),
        bento.tl.ShapeAsymmetry("cell_shape"),
        bento.tl.ShapeAsymmetry("nucleus_shape"),
        bento.tl.Ripley(),
        bento.tl.PointDispersion(),
        bento.tl.ShapeDispersion("nucleus_shape"),
    ]
    for model in models:
        model.transform(data)


@track
def intracellular_patterns(data, min_count=5, copy=False):
    """Predict transcript subcellular localization patterns.
    Patterns include: cell edge, cytoplasmic, nuclear edge, nuclear, none

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    min_count : int
        Minimum expression count per sample; otherwise ignore sample
    copy : bool
        Return a copy instead of writing to data

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    """
    adata = data.copy() if copy else data

    # Compute features if missing TODO currently recomputes everything 
    if not all(f in data.layers.keys() for f in PATTERN_MODEL_FEATURE_NAMES):
        pattern_features(data)

    X_df = get_features(adata, PATTERN_MODEL_FEATURE_NAMES, min_count)

    model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models"
    model = pickle.load(open(f"{model_dir}/rf_model_20211102.pkl", "rb"))

    pattern_prob = pd.DataFrame(
        model.predict_proba(X_df.values), index=X_df.index, columns=PATTERN_NAMES
    )

    # Save each pattern to adata
    for p, pp in tqdm(zip(PATTERN_NAMES, PATTERN_PROBS), total=len(PATTERN_NAMES)):
        indicator_df = (
            (pattern_prob >= 0.5)
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
    pattern_stats.__wrapped__(adata)

    return adata if copy else None


def get_features(data, feature_names, min_count):
    """Get features for all samples by melting features from data.layers.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    feature_names : list of str
        all values must to be keys in data.layers
    min_count : int
        Minimum expression count per sample; otherwise ignore sample

    Returns
    -------
    DataFrame
        rows are samples indexed as (cell, gene) and columns are features
    """
    sample_index = (
        data.to_df()
        .reset_index()
        .melt(id_vars="cell")
        .dropna()
        .set_index(["cell", "gene"])
    )

    sample_index = sample_index[sample_index["value"] >= min_count].drop(
        "value", axis=1
    )

    for f in feature_names:
        values = (
            data.to_df(f).reset_index().melt(id_vars="cell").set_index(["cell", "gene"])
        )
        values.columns = [f]
        sample_index = sample_index.join(values)

    return sample_index[feature_names]
