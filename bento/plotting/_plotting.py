import geopandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .._utils import quantify_variable

from shapely.affinity import translate
from shapely import geometry

import altair as alt

alt.themes.enable('opaque')
alt.renderers.enable('mimetype')
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


def plot_cells(data, type='points', genes=None, downsample=1.0, draw_masks=['cell'], binsize=200, tile=False, width=400, height=400):
    """
    Visualize distribution of variable in spatial coordinates.
    Parameters
    ----------
    data : dict
    type : str
        Options include 'points' and 'heatmap'
    genes: None, list
        Select specified list of genes. Default value of None selects all genes.
    downsample: float (0,1]
        Fraction to downsample when plotting. Useful when dealing with large datasets.
    draw_masks: list
       masks to draw outlines for. Will always include outline for union of `masks`.
    """

    # * Subset points to specified genes
    points = data
    if genes:
        print('Subsetting genes...')
        if type(genes) != list:
            genes = list(genes)

        genes = [str.upper(i) for i in genes]
        points = data[data.obs['gene'].astype(str).str.upper().isin(genes),:]

    # * Downsample points per cell
    if downsample < 1:
        print('Downsampling points...')
        downsample_mask = points.obs.groupby('cell').apply(lambda df: df.sample(frac=downsample).index.tolist())
        downsample_mask = downsample_mask.explode().dropna().tolist()
        points = points[downsample_mask,:]


    points_df = pd.DataFrame(points.X, columns=['x', 'y'])
    points_df = pd.concat([points_df, points.obs.reset_index(drop=True)], axis=1)
    # * Plot raw points
    if variable == 'points':
        print('Plotting points...')

        variable_chart = alt.Chart(points_df).mark_circle(
            opacity=0.5,
            size=10
        ).encode(
            longitude='x:Q',
            latitude='y:Q',
            color='gene',
            tooltip=['cell', 'gene']
        ).facet(
            column='cell:N'
        )

    # * Plot heatmap
    else:
        print('Plotting variable...')

        variable_chart = None

        c = alt.Chart(points_df).mark_rect().encode(
            alt.X('x:Q', bin=alt.Bin(maxbins=binsize)),
            alt.Y('y:Q', bin=alt.Bin(maxbins=binsize)),
            alt.Color('count(x):Q', scale=alt.Scale(scheme='tealblues'))
        )

        if variable_chart is None:
            variable_chart = c
        else:
            variable_chart = variable_chart + c

        # for mask in quantify_masks:
        #     # Quantify variable wrt mask
        #     data = quantify_variable(data, mask, variable)

        #     # Plot quantitative variable as background color
        #     c = alt.Chart(data['masks'][mask].reset_index().rename(columns={"index": mask})).mark_geoshape(
        #         opacity=1,
        #         stroke='black'
        #     ).encode(
        #         color=alt.Color(f'{variable}:Q', scale=alt.Scale(scheme='reds', zero=True)),
        #         tooltip=[f'{mask}:N', f'{variable}:N']
        #     )

            # if variable_chart is None:
            #     variable_chart = c
            # else:
            #     variable_chart = variable_chart + c


    # * Plot mask outlines
    # outline_chart = None
    # for mask in draw_masks:

    #     c = alt.Chart(data.uns['masks'][mask].reset_index().rename(columns={"index": mask})).mark_geoshape(
    #         fill="transparent",
    #         stroke="black",
    #         strokeDash=[4,2]
    #     )

    #     if outline_chart is None:
    #         outline_chart = c
    #     else:
    #         outline_chart = outline_chart + c

    # chart = variable_chart + outline_chart
    chart = variable_chart

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

