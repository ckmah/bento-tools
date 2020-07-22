import geopandas
import pandas as pd
import matplotlib.pyplot as plt
from .._utils import quantify_variable

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
