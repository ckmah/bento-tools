import geopandas as gpd

from .._utils import track

# from ..tools import analyze_shapes
# import scanpy as sc

# TODO resolve circular import
# def qc_metrics(data, copy=False):

#     adata = data.copy() if copy else data

#     sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
#     analyze_shapes(adata, "cell_shape", ["area", "density"])

#     return adata if copy else None


def get_layers(data, layers, min_count=None):
    """Get values of layers reformatted as a long-form dataframe.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object
    layers : list of str
        all values must to be keys in data.layers
    min_count : int, default None
        minimum number of molecules (count) required to include in output
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

    if min_count:
        sample_index = sample_index[sample_index["value"] >= min_count].drop(
            "value", axis=1
        )

    for layer in layers:
        values = (
            data.to_df(layer)
            .reset_index()
            .melt(id_vars="cell")
            .set_index(["cell", "gene"])
        )
        values.columns = [layer]
        sample_index = sample_index.join(values)

    return sample_index[layers]
