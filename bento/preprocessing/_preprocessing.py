def get_points(data, cells=None, genes=None):

    points = data.uns["points"]

    if cells is not None:
        cells = [cells] if type(cells) is str else cells

        points = points.loc[points["cell"].isin(cells)]

    if genes is not None:
        genes = [genes] if type(genes) is str else genes

        points = points.loc[points["gene"].isin(genes)]

    return points


def set_points(data, cells=None, genes=None, copy=False):
    adata = data.copy() if copy else data
    points = get_points(adata, cells, genes)
    adata.uns["points"] = points
    return adata if copy else None