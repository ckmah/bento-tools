import warnings
import geopandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .._utils import _quantify_variable

from shapely.affinity import translate
from shapely import geometry

import altair as alt


alt.themes.enable('opaque')
alt.renderers.enable('default')
alt.data_transformers.enable('json')


def quality_metrics(data, width=900, height=250):
    # Count points per cell
    n_points = data.obs['cell'].value_counts().to_frame().reset_index()
    n_points.columns = ['Cell ID', 'Transcript Count']
    # Plot points per cell
    cell_count_chart = alt.Chart(n_points).mark_area(
        opacity=0.5,
        interpolate='step').encode(
        alt.X("Transcript Count:Q", bin=alt.Bin(maxbins=25), axis=alt.Axis(title='Transcript Count')),
        alt.Y('count()', stack=None, axis=alt.Axis(title='Number of Cells'))
    ).properties(
        title='Transcripts / cell',
        width=width/3,
        height=height
    )

    # Count genes per cell
    n_genes = data.obs.groupby('cell').apply(lambda df: len(df['gene'].unique())).reset_index()
    n_genes.columns = ['Cell ID', 'Gene Count']
    # Plot genes per cell
    gene_count_chart = alt.Chart(n_genes).mark_area(
        opacity=0.5,
        interpolate='step'
        ).encode(
        alt.X("Gene Count", bin=alt.Bin(maxbins=25), axis=alt.Axis(title='Gene Count')),
        alt.Y('count()', stack=None, axis=alt.Axis(title='Number of Cells'))
    ).properties(
        title='Genes / cell',
        width=width/3,
        height=height
    )

    # Count points per gene per cell
    n_points_per_gc = data.obs.groupby(['cell', 'gene']).apply(len).to_frame().reset_index()
    n_points_per_gc.columns = ['Cell', 'Gene', 'Transcript Count']
    # Plot points per gene per cell
    gene_by_cell_count_chart = alt.Chart(n_points_per_gc).mark_area(
        opacity=0.5,
        interpolate='step'
    ).encode(
        alt.X("Transcript Count:Q", bin=alt.Bin(maxbins=50), axis=alt.Axis(title='Transcript Count')),
        alt.Y('count()', stack=None, scale=alt.Scale(type='log'), axis=alt.Axis(title='Samples'))
    ).properties(
        title='Gene expression distribution',
        width=width/3,
        height=height
    )


    chart = (cell_count_chart | gene_count_chart | gene_by_cell_count_chart).configure_view(
        strokeWidth=0
    )

    return chart


def plot_cells(data, style='points', cells=None, genes=None, downsample=1.0, draw_masks=['cell'], bins=200, tile=False, width=400, height=400):
    """
    Visualize distribution of variable in spatial coordinates.
    Parameters
    ----------
    data : dict
    type : str
        Options include 'points' and 'heatmap'
    cells: None, list
        Select specified list of cells. Default value of None selects all cells.
    genes: None, list
        Select specified list of genes. Default value of None selects all genes.
    downsample: float (0,1]
        Fraction to downsample when plotting. Useful when dealing with large datasets.
    draw_masks: list
       masks to draw outlines for. Will always include outline for union of `masks`.
    """

    # * Subset points to specified cells and genes
    points = data

    if cells is None:
        cells =  set(data.obs_vector('cell'))
    else:
        print('Subsetting cells...')
        if type(cells) != list:
            cells = [cells]

    cells = [str.upper(str(i)) for i in cells]

    if -1 in cells:
        warnings.warn('Detected points outside of cells. TODO write clean fx that drops these points')
        
        
    if genes is None:
        genes = data.obs_vector('gene')
    else:
        print('Subsetting genes...')
        if type(genes) != list:
            genes = [genes]

    genes = [str.upper(str(i)) for i in genes]
        
    points_in_cells = data.obs['cell'].astype(str).str.upper().isin(cells)
    points_in_genes = data.obs['gene'].astype(str).str.upper().isin(genes)
    points = data[points_in_cells & points_in_genes,:]

    
    # * Downsample points per cell
    if downsample < 1:
        print('Downsampling points...')
        downsample_mask = points.obs.groupby('cell').apply(lambda df: df.sample(frac=downsample).index.tolist())
        downsample_mask = downsample_mask.explode().dropna().tolist()
        points = points[downsample_mask,:]


    points_df = pd.DataFrame(points.X, columns=['x', 'y'])
    points_df = pd.concat([points_df, points.obs.reset_index(drop=True)], axis=1)

    # * Plot raw points
    if style == 'points':
        print('Plotting points...')

        print(points_df)
        variable_chart = alt.Chart(points_df).mark_circle(
            opacity=0.5,
            size=30,
            stroke='black',
            strokeWidth=1,
        ).encode(
            longitude='x:Q',
            latitude='y:Q',
            color='gene',
            tooltip=['cell', 'gene']
        )

        # * Plot heatmap
    elif style == 'heatmap':
        print('Plotting heatmap...')

        variable_chart = None

        c = alt.Chart(points_df).mark_rect().encode(
            alt.X('x:Q', bin=alt.Bin(maxbins=bins)),
            alt.Y('y:Q', bin=alt.Bin(maxbins=bins)),
            alt.Color('count(x):Q', scale=alt.Scale(scheme='tealblues'))
        )

        if variable_chart is None:
            variable_chart = c
        else:
            variable_chart = variable_chart + c
    else:
        print('Variable ')

    # * Plot mask outlines
    outline_chart = None
    for mask in draw_masks:

        # Subset masks by gene
        mask_select = cells
        strokeDash = []
        
        if mask == 'cell':
            mask_select = data.uns['masks']['cell'].index.astype(str).isin(cells)
        if mask != 'cell': # use cell index to query other mask index
            mask_select = data.uns['mask_index'][mask]['cell'].str.isin(cells)
            strokeDash = [4, 2]

        mask_df = data.uns['masks'][mask].loc[mask_select].reset_index().rename(columns={
            "index": mask})


        c = alt.Chart(mask_df).mark_geoshape(
            fill="transparent",
            stroke="black",
            strokeDash=strokeDash
        )

        if outline_chart is None:
            outline_chart = c
        else:
            outline_chart = outline_chart + c

    chart = alt.layer(variable_chart, outline_chart)

    chart = chart.project(
        type='identity',
        reflectY=True
    ).properties(
        width=width,
        height=height
    )

    if tile:
        chart = chart.repeat(
            column=cells
        )

    chart = chart.configure_view(
        strokeWidth=0
    )

    return chart

def pca(data, c1=0, c2=1, hue='gene', huetype='nominal', width=400, height=400, path=''):
    return _plot_dim(data, 'pca', hue=hue, huetype=huetype, c1=c1, c2=c2, width=width, height=height, path=path)



def umap(data, c1=0, c2=1, hue='gene', huetype='nominal', width=400, height=400, path=''):
    return _plot_dim(data, 'umap', hue=hue, huetype=huetype, c1=c1, c2=c2, width=width, height=height, path=path)


def _plot_dim(data, dim_type, **kwargs):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    dim_type : str
        'pca' or 'umap'
    """

    df = data.uns[f'{dim_type}_components']

    # Map labels to points
    if kwargs["hue"] == 'label':
        df['label'] = data.uns['labels']['label']

    chart = alt.Chart(df).mark_circle(
        opacity=0.8,
        size=10
    ).encode(
        x=f'{kwargs["c1"]}:Q',
        y=f'{kwargs["c2"]}:Q',
        color=alt.Color(kwargs["hue"], type=kwargs['huetype'], scale=alt.Scale(scheme='dark2'))
    ).properties(
        width=kwargs["width"],
        height=kwargs["height"]
    )

    if kwargs["path"]:
        chart.save(kwargs["path"], scale_factor=2)

    return chart

