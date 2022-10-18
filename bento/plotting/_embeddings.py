import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import rgb_to_hsv, to_hex
from sklearn.preprocessing import quantile_transform
from tqdm.auto import tqdm
from shapely.geometry import Polygon
import geopandas as gpd

from ..tools._utils import get_shape


def nn_embed(
    data,
    transform="pca",
    color_style="rgb",
    normalization="log",
    facecolor="black",
    ax=None,
    copy=False,
):
    """Color points by first 3 components of embedding. Useful for visualizing embeddings.
    Must run `bento.tl.nn_embed` first.
    """
    adata = data.copy() if copy else data

    if "nn_embed" not in adata.uns:
        raise ValueError("Must run `bento.tl.nn_embed` first.")

    # Get point embeddings
    embed = adata.uns["nn_embed"]
    ndata = AnnData(embed)

    # Transform to 3D space
    color_key = f"{transform}_{color_style}"
    points = adata.uns["points"]
    if color_key not in points:

        print("Normalizing embedding...")
        if normalization == "log":
            sc.pp.log1p(ndata, base=2)
        elif normalization == "total":
            sc.pp.normalize_total(adata, target_sum=1)

        if transform == "pca":
            sc.pp.pca(ndata, n_comps=3)
        elif transform == "umap":
            sc.pp.neighbors(ndata, n_neighbors=30)
            sc.tl.umap(ndata, n_components=3)

        colors = ndata.obsm[f"X_{transform}"][:, :3]
        colors = quantile_transform(colors)

        from scipy.interpolate import griddata

        cell_shapes = gpd.GeoDataFrame(geometry=get_shape(adata, "cell_shape"))
        minx, miny, maxx, maxy = cell_shapes.unary_union.bounds
        grid_x, grid_y = np.mgrid[
            minx : maxx : (maxx - minx) + 0j,
            miny : maxy : (maxy - miny) + 0j,
        ]
        values = []
        for channel in tqdm(range(colors.shape[1])):
            values.append(
                griddata(
                    points[["x", "y"]].values,
                    colors[:, channel],
                    (grid_x, grid_y),
                    method="cubic",
                    fill_value=0,
                ).T
            )

        values = np.stack(values, axis=-1)

        rect_bound = gpd.GeoDataFrame(
            geometry=[
                Polygon(
                    [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
                )
            ]
        )

        # Map to color space
    #         if color_style == "rgb":
    #             pass
    #         elif color_style == "hsv":
    #             colors = rgb_to_hsv(colors)

    #         colors = [to_hex(c) for c in colors]

    #         # Save back to points
    #         points[color_key] = colors
    #         adata.uns["points"] = points

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # crop = points.loc[(points.x>6000) & (points.y>6000)]
    # ax.scatter(
    #     x=points["x"], y=points["y"], c=points[color_key], s=1, linewidths=0
    # )

    ax.imshow(values, extent=(minx, maxx, miny, maxy), origin="lower")
    ax.autoscale(False)

    # Mask outside cells
    rect_bound.overlay(cell_shapes, how="difference").plot(
        ax=ax, facecolor="black", edgecolor="none"
    )
    cell_shapes.plot(ax=ax, facecolor='none', edgecolor='white', alpha=0.5)
    # ax.set_facecolor(facecolor)
    # ax.set_xlim(points["x"].min(), points["x"].max())
    # ax.set_ylim(points["y"].min(), points["y"].max())
    # ax.set_xticks([])
    # ax.set_yticks([])
    # sns.despine(ax=ax, top=False, right=False)

    return adata if copy else None
