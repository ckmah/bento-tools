import geopandas
import pandas as pd
import altair as alt

def spots(data, width=400, height=400, path=''):
    """
    """
    # TODO convert for mark_circle? geoshape point size is buggy
    # TODO add side histograms
    cell_chart = alt.Chart(data['cell']).mark_geoshape(
        fill=None,
        stroke='black'
    ).project(
        type='identity',
        reflectY=True)

    point_chart = alt.Chart(data['points']).mark_geoshape().encode(
        color='gene',
    ).project(
        type='identity',
        reflectY=True,
        pointRadius=1)

    chart = (cell_chart + point_chart).configure_view(
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
