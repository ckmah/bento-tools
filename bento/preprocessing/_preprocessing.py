

def filter_cells(data, min_points=5):
    if not min_points:
        print('`min_points` can not be None')
        return data
    else:
        npoints_cell = data.obs['cell'].value_counts()
        filt_cells = npoints_cell[npoints_cell >= min_points].index
        filt_points = data.obs['cell'].isin(filt_cells)
        return data[filt_points, :]


def filter_genes(data, min_points):

    def _filter_genes(obs_gene_df, min_points):
        """
        Return
        """
        gene_expr = obs_gene_df.groupby('gene').apply(len)
        genes_keep = gene_expr[gene_expr >= min_points].index
        obs_gene_df_filt = obs_gene_df.loc[obs_gene_df['gene'].isin(genes_keep)]
        return obs_gene_df_filt

    # For each cell, select genes that pass min. threshold
    gene_by_cell = data.obs.groupby('cell')[['gene']]
    obs_filt = gene_by_cell.apply(lambda obs_gene_df: _filter_genes(obs_gene_df, min_points))
    obs_keep = obs_filt.index.get_level_values(1)

    return data[obs_keep, :]
