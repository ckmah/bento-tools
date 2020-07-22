import geopandas
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from .._utils import quantify_variable

alt.themes.enable('opaque')
alt.renderers.enable('mimetype')
alt.data_transformers.enable('json')


def plot_cells(data, cells):
    if type(cells) != list:
        cells = [cells]

    ncells = len(cells)
    view = data[data.obs['cell'].isin(cells), :]
    fig, axes = plt.subplots(1, len(cells), figsize=(ncells*2, 2))
    for i, c in enumerate(cells):
        if len(cells) == 1:
            ax = axes
        else:
            ax = axes[i]

        # Plot masks
        for m in view.uns['masks']:
            if m == 'cell':
                view.uns['masks'][m][view.uns['masks'][m].index == c].plot(
                    ax=ax, color='white', edgecolor='grey', linewidth=1)
            else:
                view.uns['masks'][m][view.uns['mask_index'][m]['cell'] == c].plot(
                    ax=ax, color='white', edgecolor='grey', linewidth=1)

        # Plot points
        points = view[view.obs['cell'] == c]
        geopandas.GeoDataFrame(points.X, geometry=geopandas.points_from_xy(points.X[:, 0], points.X[:, 1])).plot(
            ax=ax, markersize=2, column=points.obs['gene'], categorical=True)
        ax.axis('off')


def quality_metrics(data, width=900, height=250):
    # TODO refactor
    # Count points per cell
    n_points = data['points']['cell'].value_counts().to_frame().reset_index()
    n_points.columns = ['Cell ID', 'Transcript Count']
    # Plot points per cell
    cell_count_chart = alt.Chart(n_points).mark_bar(opacity=0.8).encode(
        alt.X("Transcript Count", bin=True, axis=alt.Axis(title='Transcript Count')),
        alt.Y('count()', stack=None, axis=alt.Axis(title='Number of Cells'))
    ).properties(
        title='# of Transcripts per Cell',
        width=width/3,
        height=height
    )

    # Count genes per cell
    n_genes = data['points'].groupby('cell').apply(lambda df: len(df['gene'].unique())).reset_index()
    n_genes.columns = ['Cell ID', 'Gene Count']
    # Plot genes per cell
    gene_count_chart = alt.Chart(n_genes).mark_bar(opacity=0.8).encode(
        alt.X("Gene Count", bin=True, axis=alt.Axis(title='Gene Count')),
        alt.Y('count()', stack=None, axis=alt.Axis(title='Number of Cells'))
    ).properties(
        title='# of Genes per Cell',
        width=width/3,
        height=height
    )

    # Count points per gene per cell
    n_points_per_gc = data['points'].groupby(['cell', 'gene']).apply(len).to_frame().reset_index()
    n_points_per_gc.columns = ['Cell', 'Gene', 'Transcript Count']
    # Plot points per gene per cell
    gene_by_cell_count_chart = alt.Chart(n_points_per_gc).transform_density(
        density='Transcript Count',
        as_=['Transcript Count', 'density']
    ).mark_area(opacity=0.8).encode(
        alt.X("Transcript Count:Q", axis=alt.Axis(title='Transcript Count')),
        alt.Y('density:Q', axis=alt.Axis(title='Fraction of Genes (per Cell)'))
    ).properties(
        title='# of Features',
        width=width/3,
        height=height
    )


    chart = (cell_count_chart | gene_count_chart | gene_by_cell_count_chart).configure_view(
        strokeWidth=0
    )

    return chart


def display(data, quantify_masks=['cell'], variable='points', genes=None, downsample=1.0, draw_masks=['cell'], width=400, height=400):
    # TODO refactor
    """
    Visualize distribution of variable in spatial coordinates.

    Parameters
    ----------
    data : dict
    quantify_masks : list
        Set of masks to quantify variable independently in e.g. cell, nucleus, etc.
    variable : str
        Must be one of ['points', 'npoints', 'ngenes']. See _utils.quantify_variable() function for more info.
    genes: None, list
        Select specified list of genes. Default value of None selects all genes.
    downsample: float (0,1]
        Fraction to downsample when plotting. Useful when dealing with large datasets.
    draw_masks: list
       masks to draw outlines for. Will always include outline for union of `masks`.

    """

    # * Subset points to specified genes
    print('Subsetting genes...')
    if genes:
        points = data['points'][data['points']['gene'].isin(genes)]
    else:
        points = data['points']

    # * Downsample points per cell
    print('Downsampling points...')
    # TODO parallel_apply breaks bc unable to pickle cython
    points = points.groupby('cell').apply(lambda df: df.sample(frac=downsample))
    points.index = points.index.droplevel(0)

    # Restrict to points in masks
    print('Masking points')
    points_mask = [False * len(points)]
    for mask in quantify_masks:
        points_mask = points_mask | (points[mask] != -1)
    points = points[points_mask]

    # * Plot raw points
    if variable == 'points':
        print('Plotting points...')

        # Plot points by mask
        variable_chart = None
        for mask in quantify_masks:
            c = alt.Chart(points).mark_circle(
                opacity=0.5,
                size=10
            ).encode(
                longitude='x:Q',
                latitude='y:Q',
                color='gene',
                tooltip=['cell', 'gene']
            )

            if variable_chart is None:
                variable_chart = c
            else:
                variable_chart = variable_chart + c


    # * Plot variable binned by mask
    else:
        print('Plotting variable...')

        variable_chart = None
        for mask in quantify_masks:
            # Quantify variable wrt mask
            data = quantify_variable(data, mask, variable)

            # Plot quantitative variable as background color
            c = alt.Chart(data['masks'][mask].reset_index().rename(columns={"index": mask})).mark_geoshape(
                opacity=1,
                stroke='black'
            ).encode(
                color=alt.Color(f'{variable}:Q', scale=alt.Scale(scheme='reds', zero=True)),
                tooltip=[f'{mask}:N', f'{variable}:N']
            )

            if variable_chart is None:
                variable_chart = c
            else:
                variable_chart = variable_chart + c


    # * Plot mask outlines
    all_masks = set(quantify_masks).union(set(draw_masks))
    outline_chart = None
    for mask in all_masks:
        # Plot quantify mask outline in case empty
        if mask in quantify_masks:
            c = alt.Chart(data['masks'][mask].reset_index().rename(columns={"index": mask})).mark_geoshape(
                fill="transparent",
                stroke="black",
            ).encode(
                tooltip=[f'{mask}:N']
            )
        # Plot draw_masks outlines with transparent bg
        else:
            c = alt.Chart(data['masks'][mask].reset_index().rename(columns={"index": mask})).mark_geoshape(
                fill="transparent",
                stroke="black",
                strokeDash=[4,2]
            ).encode(
                tooltip=[f'{mask}:N']
            )

        if outline_chart is None:
            outline_chart = c
        else:
            outline_chart = outline_chart + c

    chart = outline_chart + variable_chart

    chart = chart.project(
        type='identity',
        reflectY=True
    ).configure_view(
        strokeWidth=0
    ).properties(
        width=width,
        height=height
    )

    return chart

# TODO refactor data['pca_components']
def pca(data, c1=0, c2=1, width=400, height=400, path=''):
    """
    """
    chart = alt.Chart(data['pca_components'].reset_index()).mark_circle().encode(
        x=f'{c1}:Q',
        y=f'{c2}:Q',
        color='gene:N',
    ).properties(
        width=width,
        height=height
    )

    if path:
        chart.save(path, scale_factor=2)

    return chart

# TODO refactor data['umap_components']
def umap(data, c1=0, c2=1, hue='gene', width=400, height=400, path=''):
    """
    """
    chart = alt.Chart(data['umap_components'].reset_index()).mark_circle().encode(
        x=f'{c1}:Q',
        y=f'{c2}:Q',
        color=f'{hue}:N',
    ).properties(
        width=width,
        height=height
    )

    if path:
        chart.save(path, scale_factor=2)

    return chart
