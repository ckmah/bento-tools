import geopandas
import pandas as pd
import altair as alt

alt.data_transformers.enable('json')

def spots(data, width=400, height=400, path='', downsample=1.0, genes=[], size=5):

    # TODO add side histograms
    if len(genes) > 0:
        points = data['points'][data['points']['gene'].isin(genes)]
    else:
        points = data['points']

    # Downsample points
    points = points.sample(frac=downsample)
        
    point_chart = alt.Chart(points).mark_circle(size=size).encode(
        longitude='x:Q',
        latitude='y:Q',
        color='gene',
    )

    cell_chart = alt.Chart(data['cell']).mark_geoshape(
        fill='#EEE',
        stroke='gray',
    )

    nucleus_chart = alt.Chart(data['nucleus']).mark_geoshape(
        fill='#EEE',
        stroke='gray',
    )

    chart = (cell_chart + nucleus_chart + point_chart).project(
        type='identity',
        reflectY=True
    ).configure_view(
        strokeWidth=0
    ).properties(
        width=width,
        height=height
    )

    if path:
        chart.save(path, scale_factor=2)

    return chart


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


def umap(data, c1=0, c2=1, width=400, height=400, path=''):
    """
    """
    chart = alt.Chart(data['umap_components'].reset_index()).mark_circle().encode(
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
