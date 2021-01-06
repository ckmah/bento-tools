from collections import defaultdict

import bento
import geopandas
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

from umap import UMAP

from .._settings import pandarallel, settings

from tqdm.auto import tqdm

def subsample(data, fraction, copy=False):
    """Randomly subsample data stratified by cell.
    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial transcriptomics data.
    fraction : float
        Float between (0, 1] to subsample data.
    copy : bool
        Return view of AnnData if False, return copy if True. Default False.
    Returns
    -------
    AnnData
        Returns subsampled view of original AnnData object.
    """
    keep = data.obs.groupby('cell').apply(
        lambda df: df.sample(frac=fraction)).index.droplevel(0)

    if copy:
        return data[keep, :].copy()
    else:
        return data[keep, :]


def _test_gene(data, groupby):
    """Perform pairwise comparison between groupby and every class.

    Parameters
    ----------
    data : DataFrame
        Phenotype and localization pattern labels across cells for a single gene.
    groupby : str
        Variable grouping cells for differential analysis. Should be present in data.columns.

    Returns
    -------
    DataFrame
        Differential localization test results. [# of patterns, ]
    """
    # GLM approach
    # Discrete https://www.statsmodels.org/stable/discretemod.html
    #     res = glm(formula=f'{groupby} ~ cell2D + cellext + foci + nuc2D + polarized + random',
    #               data=data,
    #               family=sm.families.Binomial(link=sm.families.links.probit()),).fit()
    #     return res.wald_test_terms().table

    results = []
    for c in ["cell2D", "cellext", "foci", "nuc2D", "polarized", "random"]:

        # Make dummy columns and add to table
        group_dummies = pd.get_dummies(data[groupby])
        group_names = group_dummies.columns.tolist()
        group_data = pd.concat([data, group_dummies], axis=1)
        
        cont_table_empty = pd.DataFrame([[0,0], [0,0]])
        for g in group_names:
            cont_table = pd.crosstab(group_data[g], group_data[c]).fillna(0).add(cont_table_empty, fill_value=0)
            oddsratio, p_value = stats.fisher_exact(cont_table)
            results.append([c, g, oddsratio, p_value])
    return pd.DataFrame(results, columns=["pattern", "group", "oddsratio", "pvalue"],
    )


def diff_spots(data, groupby, copy=False):
    """Test for differential localization across phenotype of interest.

    Parameters
    ----------
    data : AnnData 
        Anndata formatted spatial transcriptomics data.
    groupby : str, pd.Series
        Variable grouping cells for differential analysis. If str, needs to be a key in data.uns['sample_data']. If pandas Series, must be same length as number of cells in 'data'. 
    copy : bool, optional
        Return view of AnnData if False, return copy if True. By default False.
    """
    adata = data.copy() if copy else data

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    group_data = adata.uns['sample_index'].reset_index(drop=True).join(adata.uns['sample_data']['patterns'])

    if type(groupby) is pd.Series:
        group_data = group_data.reset_index(drop=True).join(groupby)
    else:
        group_data = group_data.reset_index(drop=True).join(adata.uns['sample_data'][groupby])
    
    if settings.n_cores > 1:
        results = group_data.groupby("gene").parallel_apply(lambda gene_df: _test_gene(gene_df, groupby))
    else:
        tqdm.pandas(desc=f'Testing {groupby}')
        results = group_data.groupby("gene").progress_apply(lambda gene_df: _test_gene(gene_df, groupby))
        
    # Formatting
    results = results.reset_index().drop("level_1", axis=1)

    # FDR correction
    results_adj = []
    for _, df in results.groupby("pattern"):
        df["padj"] = multipletests(df["pvalue"], method="hs")[1]
        results_adj.append(df)

    results_adj = pd.concat(results_adj)
    results_adj = results_adj.dropna()

    # -log10pvalue, padj
    results_adj["-log10p"] = -np.log10(results_adj["pvalue"])
    results_adj["-log10padj"] = -np.log10(results_adj["padj"])

    return results_adj.sort_values('pvalue')


def score_genes_cell_cycle(data, copy=False, **kwargs):
    """Wrapper to preprocess counts similar to scRNA-seq and score cell cycle genes via scanpy.

    Parameters
    ----------
    data : AnnData
        Spatial anndata object.
    copy : bool, optional
        Copy data or modify inplace, by default False.
    **kwargs
        Passed to scanpy.tl.score_genes().
    """
    adata = data.copy() if copy else data

    # Extract points
    expression = pd.DataFrame(adata.X, index=pd.MultiIndex.from_frame(adata.obs[['cell', 'gene']]))

    # Aggregate points to counts
    expression = adata.obs[['cell','gene']].groupby(['cell', 'gene']).apply(lambda x: x.shape[0]).to_frame()
    expression = expression.reset_index()

    # Remove extracellular points
    expression = expression.loc[expression['cell'] != '-1']
    
    # Format as dense cell x gene counts matrix
    expression = expression.pivot(index='cell', columns='gene').fillna(0)
    expression.columns = expression.columns.droplevel(0)
    expression.columns = expression.columns.str.upper()

    # Create scanpy anndata object to use scoring function
    sc_data = sc.AnnData(expression)

    # Perform standard scRNA-seq filtering
    # sc.pp.filter_cells(sc_data, min_genes=200)
    # sc.pp.filter_genes(sc_data, min_cells=3)
    
    # Standard scaling
    sc.pp.normalize_per_cell(sc_data, counts_per_cell_after=1e4)
    sc.pp.log1p(sc_data)
    sc.pp.scale(sc_data)

    # Get cell cycle genes
    cell_cycle_genes = pd.read_csv('https://github.com/theislab/scanpy_usage/raw/master/180209_cell_cycle/data/regev_lab_cell_cycle_genes.txt', header=None)
    cell_cycle_genes = cell_cycle_genes.values.flatten().tolist()

    # Define genes for each phase
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]

    # Only score genes present in data
    cell_cycle_genes = [x for x in cell_cycle_genes if x in sc_data.var_names]

    # Score cell cycle genes
    sc.tl.score_genes_cell_cycle(sc_data, s_genes=s_genes, g2m_genes=g2m_genes)

    # Get indexed cell cycle scores and phase labels
    sc_obs = sc_data.obs.reset_index()
    sc_obs.index = sc_obs.index.astype(str)
    sc_obs['cell'] = sc_obs['cell'].astype(str)

    # Save to spatial anndata object
    _init_sample_info(adata)
    adata.uns['sample_data']['cell_cycle'] = adata.uns['sample_index'].merge(sc_obs, on='cell', how='left')[['phase']]
    adata.uns['sample_data']['cell_cycle'].columns = ['cell_cycle']
    return adata if copy else None


def pca(data, features, n_components=2, copy=False):
    """Perform principal component analysis on samples using specified features.

    Parameters
    ----------
    data : AnnData
        [description]
    features : [type]
        [description]
    n_components : int, optional
        [description], by default 2
    copy : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy() if copy else data

    if type(features) == str:
        features = [features]
    
    # Initialize PCA
    pca = PCA(n_components=n_components)

    # Compute pca
    features_x = np.array([bento.get_feature(adata, f) for f in features])
    data.uns['sample_pca'] = pca.fit_transform(features_x)

    # Save PCA outputs
    data.uns['pca'] = dict()
    data.uns['pca']['features_used'] = features
    data.uns['pca']['components_'] = PCA.components_
    data.uns['pca']['explained_variance_'] = PCA.explained_variance_
    data.uns['pca']['explained_variance_ratio_'] = PCA.explained_variance_ratio_
    return adata if copy else None


def umap(data, n_components=2, n_neighbors=15, **kwargs):
    """
    """
    fit = u.UMAP(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
    umap_components = fit.fit_transform(data.uns['features'])
    columns = [str(c) for c in range(0, umap_components.shape[1])]
    umap_components = pd.DataFrame(umap_components,
                         index=data.uns['features'].index,
                         columns=columns)
    data.uns['umap_components'] = umap_components
    return data


def _map_to_obs(data, name):

    if name not in data.uns['sample_data'].keys():
        print(f'{name} not found.')
        return
        
    data.obs[name] = data.uns['sample_data'][name][adata.obs['sample_id']]
    
    
def _init_sample_info(data):
    if 'sample_index' not in data.uns.keys():
        sample_index = data.obs[['cell', 'gene']].value_counts().index.to_frame(index=False)
        sample_index.columns = ['cell', 'gene']
        sample_index.index = sample_index.index.astype(str)
        data.uns['sample_index'] = sample_index
        data.uns['sample_data'] = dict()
        