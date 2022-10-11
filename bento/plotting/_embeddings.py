import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import rgb_to_hsv, to_hex
from sklearn.preprocessing import quantile_transform


def nn_embed(data, transform="pca", color_style="rgb", ax=None, copy=False):
    """Color points by first 3 components of embedding. Useful for visualizing embeddings.
    Must run `bento.tl.nn_embed` first.
    """
    adata = data.copy() if copy else data

    if "nn_embed" not in adata.uns:
        raise ValueError("Must run `bento.tl.nn_embed` first.")

    # Get point embeddings
    embed=adata.uns["nn_embed"]
    ndata = AnnData(embed)

    # Transform to 3D space
    print("Normalizing embedding...")
    sc.pp.log1p(ndata, base=2)

    if transform == "pca":
        sc.pp.pca(ndata, n_comps=3)
    elif transform == "umap":
        sc.pp.neighbors(ndata, n_neighbors=30)
        sc.tl.umap(ndata, n_components=3)

    colors = ndata.obsm[f"X_{transform}"][:, :3]
    colors = quantile_transform(colors)

    # Map to color space
    if color_style == "rgb":
        pass
    elif color_style == "hsv":
        colors = rgb_to_hsv(colors)

    colors = [to_hex(c) for c in colors]

    # Save back to points
    color_key = f"{transform}_color"
    adata.uns["points"][color_key] = colors

    points = adata.uns["points"]

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.scatter(x=points["x"], y=points["y"], c=points[color_key], s=1, linewidths=0)
    ax.set_facecolor("black")

    return adata if copy else None
