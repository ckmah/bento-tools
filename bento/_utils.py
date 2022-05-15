import inspect
from functools import wraps

from anndata import AnnData

import seaborn as sns

PATTERN_NAMES = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]
PATTERN_PROBS = [f"{p}_p" for p in PATTERN_NAMES]
TENSOR_DIM_NAMES = ["layers", "cells", "genes"]

# Colors correspond to order of PATTERN_NAMES: cyan, blue, gray, orange, red
PATTERN_COLORS = ['#17becf', '#1f77b4', '#7f7f7f', '#ff7f0e', '#d62728']

# Colors to represent each dimension (features, cells, genes); Set2 palette n_colors=3
DIM_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb']
# ['#AD6A6C', '#f5b841', '#0cf2c9']

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def track(func):
    """
    Track changes in AnnData object after applying function.
    """

    @wraps(func)
    def wrapper(*args, **kwds):
        kwargs = get_default_args(func)
        kwargs.update(kwds)

        if type(args[0]) == AnnData:
            adata = args[0]
        else:
            adata = args[1]

        old_attr = list_attributes(adata)

        if kwargs["copy"]:
            out_adata = func(*args, **kwds)
            new_attr = list_attributes(out_adata)
        else:
            func(*args, **kwds)
            new_attr = list_attributes(adata)

        # Print differences between new and old adata
        out = ""
        out += "AnnData object modified:"

        if old_attr["n_obs"] != new_attr["n_obs"]:
            out += f"\nn_obs: {old_attr['n_obs']} -> {new_attr['n_obs']}"

        if old_attr["n_vars"] != new_attr["n_vars"]:
            out += f"\nn_vars: {old_attr['n_vars']} -> {new_attr['n_vars']}"

        modified = False
        for attr in old_attr.keys():

            if attr == "n_obs" or attr == "n_vars":
                continue

            removed = list(old_attr[attr] - new_attr[attr])
            added = list(new_attr[attr] - old_attr[attr])

            if len(removed) > 0 or len(added) > 0:
                modified = True
                out += f"\n    {attr}:"
                if len(removed) > 0:
                    out += f"\n        - {', '.join(removed)}"
                if len(added) > 0:
                    out += f"\n        + {', '.join(added)}"

        if modified:
            print(out)

        return out_adata if kwargs["copy"] else None

    return wrapper


def list_attributes(adata):
    found_attr = dict(n_obs=adata.n_obs, n_vars=adata.n_vars)
    for attr in [
        "obs",
        "var",
        "uns",
        "obsm",
        "varm",
        "layers",
        "obsp",
        "varp",
    ]:
        keys = set(getattr(adata, attr).keys())
        found_attr[attr] = keys

    return found_attr


def pheno_to_color(pheno, palette):
    """
    Maps list of categorical labels to a color palette.
    Input values are first sorted alphanumerically least to greatest before mapping to colors.
    This ensures consistent colors regardless of input value order.

    Parameters
    ----------
    pheno : pd.Series
        Categorical labels to map
    palette: None, string, or sequence, optional
        Name of palette or None to return current palette.
        If a sequence, input colors are used but possibly cycled and desaturated.
        Taken from sns.color_palette() documentation.

    Returns
    -------
    dict
        Mapping of label to color in RGBA
    tuples
        List of converted colors for each sample, formatted as RGBA tuples.

    """
    import seaborn as sns

    if type(palette) is str:
        palette = sns.color_palette(palette)
    else:
        palette = palette

    values = list(set(pheno))
    values.sort()
    palette = sns.color_palette(palette, n_colors=len(values))
    study2color = dict(zip(values, palette))
    sample_colors = [study2color[v] for v in pheno]
    return study2color, sample_colors
