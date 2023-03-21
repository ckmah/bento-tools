import inspect
from functools import wraps

import geopandas as gpd
import pandas as pd
import seaborn as sns
from anndata import AnnData
from shapely import wkt


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

    1. First remembers a shallow list of AnnData attributes by listing keys from obs, var, etc.
    2. Perform arbitrary task
    3. List attributes again, perform simple diff between list of old and new attributes
    4. Print to user added and removed keys

    Parameters
    ----------
    func : function
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
    """Traverse AnnData object attributes and list keys.

    Parameters
    ----------
    adata : AnnData
        AnnData object

    Returns
    -------
    dict
        Dictionary of keys for each AnnData attribute.
    """
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
    if isinstance(palette, str):
        palette = sns.color_palette(palette)

    values = list(set(pheno))
    values.sort()
    palette = sns.color_palette(palette, n_colors=len(values))
    study2color = dict(zip(values, palette))
    sample_colors = [study2color[v] for v in pheno]
    return study2color, sample_colors


def sync(data, copy=False):
    """
    Sync existing point sets and associated metadata with data.obs_names and data.var_names

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    copy : bool, optional
    """
    adata = data.copy() if copy else data

    if "point_sets" not in adata.uns.keys():
        adata.uns["point_sets"] = dict(points=[])

    # Iterate over point sets
    for point_key in adata.uns["point_sets"]:
        points = adata.uns[point_key]

        # Subset for cells
        cells = adata.obs_names.tolist()
        in_cells = points["cell"].isin(cells)

        # Subset for genes
        in_genes = [True] * points.shape[0]
        if "gene" in points.columns:
            genes = adata.var_names.tolist()
            in_genes = points["gene"].isin(genes)

        # Combine boolean masks
        valid_mask = (in_cells & in_genes).values

        # Sync points using mask
        points = points.loc[valid_mask]

        # Remove unused categories for categorical columns
        for col in points.columns:
            if points[col].dtype == "category":
                points[col].cat.remove_unused_categories(inplace=True)

        adata.uns[point_key] = points

        # Sync point metadata using mask
        for metadata_key in adata.uns["point_sets"][point_key]:
            metadata = adata.uns[metadata_key]

            if isinstance(metadata, pd.DataFrame):
                adata.uns[metadata_key] = metadata.loc[valid_mask, :]
            else:
                adata.uns[metadata_key] = adata.uns[metadata_key][valid_mask]

    return adata if copy else None


def _register_points(data, point_key, metadata_keys):
    required_cols = ["x", "y", "cell"]

    if point_key not in data.uns.keys():
        raise ValueError(f"Key {point_key} not found in data.uns")

    points = data.uns[point_key]

    if not all([col in points.columns for col in required_cols]):
        raise ValueError(
            f"Point DataFrame must have columns {', '.join(required_cols)}"
        )

    # Check for valid cells
    cells = data.obs_names.tolist()
    if not points["cell"].isin(cells).all():
        raise ValueError("Invalid cells in point DataFrame")

    # Initialize/add to point registry
    if "point_sets" not in data.uns.keys():
        data.uns["point_sets"] = dict()

    if point_key not in data.uns["point_sets"].keys():
        data.uns["point_sets"][point_key] = []

    if len(metadata_keys) < 0:
        return

    # Register metadata
    for key in metadata_keys:
        # Check for valid metadata
        if key not in data.uns.keys():
            raise ValueError(f"Key {key} not found in data.uns")

        n_points = data.uns[point_key].shape[0]
        metadata_len = data.uns[key].shape[0]
        if metadata_len != n_points:
            raise ValueError(
                f"Metadata {key} must have same length as points {point_key}"
            )

        # Add metadata key to registry
        if key not in data.uns["point_sets"][point_key]:
            data.uns["point_sets"][point_key].append(key)


def register_points(point_key: str, metadata_keys: list):
    """Decorator function to register points to the current `AnnData` object.
    This keeps track of point sets and keeps them in sync with `AnnData` object.

    Parameters
    ----------
    point_key : str
        Key where points are stored in `data.uns`
    metadata_keys : list
        Keys where point metadata are stored in `data.uns`
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            kwargs = get_default_args(func)
            kwargs.update(kwds)

            func(*args, **kwds)
            data = args[0]
            # Check for required columns
            return _register_points(data, point_key, metadata_keys)

        return wrapper

    return decorator


def sc_format(data, copy=False):
    """
    Convert data.obs GeoPandas columns to string for compatibility with scanpy.
    """
    adata = data.copy() if copy else data

    shape_names = data.obs.columns.str.endswith("_shape")

    for col in data.obs.columns[shape_names]:
        adata.obs[col] = adata.obs[col].astype(str)

    return adata if copy else None


def geo_format(data, copy=False):
    """
    Convert data.obs scanpy columns to GeoPandas compatible types.
    """
    adata = data.copy() if copy else data

    shape_names = adata.obs.columns[adata.obs.columns.str.endswith("_shape")]

    adata.obs[shape_names] = adata.obs[shape_names].apply(
        lambda col: gpd.GeoSeries(
            col.astype(str).apply(lambda val: wkt.loads(val) if val != "None" else None)
        )
    )

    return adata if copy else None
