import geopandas
import pandas as pd
import altair as alt

def spots(data, width=800, height=800, path=''):
    """
    """
    # TODO convert for mark_circle? geoshape point size is buggy
    # TODO add side histograms

    point_chart = alt.Chart(data['points']).mark_circle(size=5).encode(
        longitude='x:Q',
        latitude='y:Q',
        color='gene',
    )

    cell_chart = alt.Chart(data['cell']).mark_geoshape(
        fill='#DDD',
        stroke='gray',
        opacity=0.6
    )

    nucleus_chart = alt.Chart(data['nucleus']).mark_geoshape(
        fill='#DDD',
        stroke='gray',
        opacity=0.6
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
