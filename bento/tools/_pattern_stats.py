import numpy as np
import pandas as pd

from ..utils import track

@track
def pattern_stats(data, copy=False):
    """Computes frequencies of input layer values across cells and across genes.
    Assumes layer values are categorical.

    Parameters
    ----------
    data : [type]
        [description]
    layer : [type]
        [description]
    copy : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy() if copy else data

    detected = np.ones(adata.shape)

    for c in PATTERN_NAMES:
        detected = detected & ~data.to_df(c).isna()
    
    adata.var['n_detected'] = detected.sum().astype(int)
    adata.var[f'fraction_detected'] = (adata.var['n_detected'] / adata.n_obs).astype(float)
    
    adata.obs['n_detected'] = detected.sum(axis=1).astype(int)
    adata.obs[f'fraction_detected'] = (adata.obs['n_detected'] / adata.n_vars).astype(float)

    for c in PATTERN_NAMES:
        # Save frequencies across genes
        counts = adata.to_df(c)

        # Save frequencies across cells
        adata.var[f'{c}_count'] = counts.fillna(0).sum(axis=0).astype(int)
        adata.var[f'{c}_fraction'] = (adata.var[f'{c}_count'] / adata.var['n_detected']).astype(float)

        adata.obs[f'{c}_count'] = counts.fillna(0).sum(axis=1).astype(int)
        adata.obs[f'{c}_fraction'] = (adata.obs[f'{c}_count'] / adata.obs['n_detected']).astype(float)

    return adata if copy else None


@track
def pattern_diff(data, phenotype=None, copy=False):
    """Gene-wise test for differential localization across phenotype of interest.

    Parameters
    ----------
    data : AnnData
        Anndata formatted spatial data.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs_names.
    copy : bool, optional
        Return view of AnnData if False, return copy if True. By default False.
    """
    adata = data.copy() if copy else data

    if phenotype not in adata.obs.columns:
        raise ValueError("Phenotype is invalid.")

    detected = np.ones(data.shape)
    for c in PATTERN_NAMES:
        detected = detected & ~data.to_df(c).isna()

    gene_fc_stats = []
    for c in PATTERN_NAMES:
        a = data.var[f'{c}_count'].to_frame()
        a["n_cells_detected"] = detected[a.index].sum()
        a["cell_fraction"] = a["n_cells_detected"] / data.n_obs

        a.columns = [
            "pattern_count",
            "n_cells_detected",
            "cell_fraction",
        ]

        a["pattern"] = c

        # save each sum to a new column, one for each group
        group_freq = (
            data.to_df(c)
            .replace("none", np.nan)
            .astype(float)
            .groupby(data.obs[phenotype])
            .sum()
            .T
        )

        def log2fc(group_count):
            rest_cols = group_freq.columns[group_freq.columns != group_count.name]
            rest_mean = group_freq[rest_cols].mean(axis=1)

            return np.log2((group_count + 1) / (rest_mean + 1))

        # log2fc of group / mean(rest)
        group_fc = group_freq.apply(log2fc, axis=0)
        group_fc.columns = group_fc.columns.astype(str)
        group_fc.columns += "_log2fc"
        a = a.join(group_fc)
        gene_fc_stats.append(a)

    gene_fc_stats = pd.concat(gene_fc_stats)
    gene_fc_stats = gene_fc_stats.reset_index()
    adata.uns[f"{phenotype}_dp"] = gene_fc_stats

    return adata if copy else None
