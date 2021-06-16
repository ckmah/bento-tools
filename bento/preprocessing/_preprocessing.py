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


def subsample(data, frac=0.2, copy=True):
    adata = data.copy() if copy else data
    points = get_points(data)

    sampled_pts = points.groupby(['cell', 'gene']).sample(frac=frac)
    
    X = sampled_pts[['cell', 'gene']].pivot_table(
        index="cell", columns="gene", aggfunc=len, fill_value=0
    )

    adata.uns['points'] = sampled_pts

    adata = adata[X.index, X.columns]
    adata.X = X

    return adata if copy else None
