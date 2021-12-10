import pickle

OneHotEncoder = None
LabelBinarizer = None
import bento
import pandas as pd
from tqdm.auto import tqdm
from .._utils import PATTERN_NAMES, PATTERN_PROBS, track
from ._pattern_stats import pattern_stats


@track
def intracellular_patterns(data, min_count=5, copy=False):
    """Predict transcript subcellular localization patterns.
    Patterns include: cell edge, cytoplasmic, nuclear edge, nuclear, none
    """
    adata = data.copy() if copy else data

    features = [
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

    X_df = get_features(adata, features, min_count)

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


def get_features(data, features, min_count):

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

    for f in features:
        values = (
            data.to_df(f).reset_index().melt(id_vars="cell").set_index(["cell", "gene"])
        )
        values.columns = [f]
        sample_index = sample_index.join(values)

    return sample_index[features]
