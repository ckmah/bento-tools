# for enrichment analyses, plot cells colored by number of genes present in each cell

from ..plotting import plot


def fe(data, gs, fraction=False, cmap=None, copy=False, **kwargs):
    """For a given pathway, plot cells colored by the number of genes present in each cell.
    
    Parameters
    ----------
    data : AnnData
        Spatially formatted AnnData
    gs : DataFrame
        Gene set to be plotted
    
    """

    adata = data.copy() if copy else data

    if 'fe' not in data.uns:
        print('Run bento.tl.fe first.')
        return

    data.obs[gs] = data.uns['fe_stats'][gs]
    
    if fraction:
        data.obs[gs] = data.obs[gs] / data.uns['fe_size']


    plot(data, kind="cell", hue=gs, cmap=cmap, **kwargs)

    return adata if copy else None