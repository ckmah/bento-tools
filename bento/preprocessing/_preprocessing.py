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


def filter_points(data, min=None, max=None, copy=False):
    """Select samples with at least min_count and at most max_count points.

    Parameters
    ----------
    data : AnnData
        bento loaded AnnData
    min : int, optional
        minimum points needed to keep sample, by default None
    max : int, optional
        maximum points needed to keep sample, by default None
    copy : bool, optional
        True modifies data in place while False returns a copy of the modified data, by default False
    """
    adata = data.copy() if copy else data
    points = get_points(data)

    expr_flat = adata.to_df().reset_index().melt(id_vars='cell')
    
    if min:
        expr_flat = expr_flat.query(f'value >= {min}')
    
    if max:
        expr_flat = expr_flat.query(f'value <= {max}')

    expr_flat = set(tuple(x) for x in expr_flat[['cell', 'gene']].values)

    sample_ids = [tuple(x) for x in points[['cell', 'gene']].values]
    keep = [True if x in expr_flat else False for x in sample_ids]

    points = points.loc[keep]

    # points = points.groupby(['cell', 'gene']).apply(lambda df: df if df.shape[0] >= 5 else None).reset_index(drop=True)
    adata.uns['points'] = points
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
