import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

import dask.dataframe as dd
import geopandas
import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision
from joblib import Parallel, delayed
from rasterio import features
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from umap import UMAP

from ..preprocessing import get_points


def gene_leiden(data, copy=False):
    adata = data.copy() if copy else data

    coloc_sim = (
        adata.uns["coloc_sim_agg"]
        .pivot_table(index="g1", columns="g2", values="coloc_sim")
        .fillna(0)
    )
    coloc_sim = coloc_sim.dropna()

    genes = coloc_sim.index

    # Z scale features
    coloc_sim = zscore(coloc_sim, axis=0)

    nn = NearestNeighbors().fit(coloc_sim)
    connectivity = nn.kneighbors_graph(coloc_sim, n_neighbors=5).toarray()

    loc_umap = UMAP().fit_transform(connectivity)
    loc_umap = pd.DataFrame(loc_umap, index=genes)
    adata.varm["loc_umap"] = loc_umap.reindex(adata.var_names)

    return adata if copy else None


def coloc_cluster_genes(data, resolution=1, copy=False):
    adata = data.copy() if copy else data

    coloc_sim = (
        adata.uns["coloc_sim_agg"]
        .pivot_table(index="g1", columns="g2", values="coloc_sim")
        .fillna(0)
    )

    genes = coloc_sim.index.tolist()
    coloc_sim = coloc_sim.values

    # Z scale features
    coloc_sim = zscore(coloc_sim, axis=0)

    nn = NearestNeighbors().fit(coloc_sim)
    connectivity = nn.kneighbors_graph(coloc_sim, n_neighbors=5).toarray()

    g = ig.Graph.Adjacency(connectivity)
    g.es["weight"] = connectivity[connectivity != 0]
    g.vs["label"] = genes
    partition = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=resolution,
    )
    gene_clusters = pd.Series(partition.membership, dtype=int, index=g.vs["label"])

    adata.var["coloc_group"] = gene_clusters.reindex(adata.var_names)
    return adata if copy else None


def coloc_sim(data, radius=3, min_count=5, n_cores=1, copy=False):
    """Calculate pairwise gene colocalization similarity with the cross L function.

    Parameters
    ----------
    adata : AnnData
        Anndata formatted spatial data.
    radius : int
        Max radius to search for neighboring points, by default 3
    min_count : int
        Minimum points needed to be eligible for analysis.
    Returns
    -------
    adata : AnnData
        .uns['coloc_sim']: Pairwise gene colocalization similarity within each cell formatted as a long dataframe.
    """
    adata = data.copy() if copy else data

    # Filter points and counts by min_count
    counts = adata.to_df()

    # Helper function to apply per cell
    def cell_coloc_sim(p, g_density, name):

        # Get xy coordinates
        xy = p[["x", "y"]].values

        # Get neighbors within fixed outer_radius for every point
        nn = NearestNeighbors(radius=radius).fit(xy)
        distances, point_index = nn.radius_neighbors(xy, return_distance=True)

        # Enumerate point-wise gene labels
        gene_index = p["gene"].reset_index(drop=True)

        # Convert to adjacency list of points, no double counting
        neighbor_pairs = []
        for g1, neighbors, n_dists in zip(gene_index.values, point_index, distances):
            for g2, d in zip(neighbors, n_dists):
                neighbor_pairs.append([g1, g2, d])

        # Calculate pair-wise gene similarity
        neighbor_pairs = pd.DataFrame(neighbor_pairs, columns=["g1", "g2", "p_dist"])

        # Keep minimum distance to g2 point
        neighbor_pairs = neighbor_pairs.groupby(["g1", "g2"]).agg("min").reset_index()
        neighbor_pairs.columns = ["g1", "g2", "point_dist"]

        # Map to gene index
        neighbor_pairs["g2"] = neighbor_pairs["g2"].map(gene_index)

        # Count number of points within distance of increasing radius
        r_step = 0.5
        expected_counts = [
            lambda dists: (dists <= r).sum()
            for r in np.arange(r_step, radius + r_step, r_step)
        ]
        metrics = (
            neighbor_pairs.groupby(["g1", "g2"])
            .agg({"point_dist": expected_counts})
            .reset_index()
        )

        # Colocalization metric: max of L_ij(r) for r <= radius
        g2_density = g_density.loc[metrics["g2"].tolist()].values
        metrics["sim"] = (
            (metrics["point_dist"].divide(g2_density * np.pi, axis=0))
            .pow(0.5)
            .max(axis=1)
        )
        metrics["cell"] = name

        # Ignore self colocalization
        # metrics = metrics.loc[metrics["g1"] != metrics["g2"]]

        return metrics[["cell", "g1", "g2", "sim"]]

    # Only keep genes >= min_count in each cell
    gene_densities = []
    counts.apply(lambda row: gene_densities.append(row[row >= min_count]), axis=1)
    # Calculate point density per gene per cell
    gene_densities /= adata.obs["cell_area"]
    gene_densities = gene_densities.values

    cell_metrics = Parallel(n_jobs=n_cores)(
        delayed(cell_coloc_sim)(
            get_points(adata, cells=g_density.name, genes=g_density.index.tolist()),
            g_density,
            g_density.name,
        )
        for g_density in tqdm(gene_densities)
    )

    cell_metrics = pd.concat(cell_metrics)
    cell_metrics.columns = cell_metrics.columns.get_level_values(0)

    # Make symmetric (Lij = Lji)
    cell_metrics['pair'] = cell_metrics.apply(lambda row: '-'.join(sorted([row['g1'], row['g2']])), axis=1)
    cell_symmetric = cell_metrics.groupby(['cell', 'pair']).mean()

    # Retain gene pair names
    cell_symmetric = cell_metrics.set_index(['cell', 'pair']).drop('sim', axis=1).join(cell_symmetric).reset_index()
    
    # Aggregate across cells
    coloc_agg = cell_symmetric.groupby(['pair'])['sim'].mean().to_frame()
    coloc_agg = coloc_agg.join(cell_symmetric.set_index('pair').drop(['sim', 'cell'], axis=1)).reset_index().drop_duplicates()
    
    # Save coloc similarity
    cell_metrics[['cell', 'g1', 'g2', 'pair']].astype('category', copy=False)
    coloc_agg[['g1', 'g2', 'pair']].astype('category', copy=False)
    adata.uns["coloc_sim"] = cell_metrics
    adata.uns["coloc_sim_agg"] = coloc_agg

    return adata if copy else None


def get_gene_coloc(data, gene):
    """
    For a given gene, return its colocalization with all other genes.
    
    Assumes colocalization is precomputed with bento.tl.coloc_sim.
    
    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial data.
    gene : str
        The name of a gene, must be present in data.var.
        
    Returns
    -------
    pd.DataFrame
        Subset of data.uns['coloc_sim_agg'].
    """
    sim = data.uns['coloc_sim_agg']
    return sim.loc[sim['g1'].str.match(gene)]


def get_gene_set_coloc(data, genes):
    """
    For a list of genes, return their pairwise colocalization with each other.
    
    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial data.
    gene : list of str
        The names of genes, must be present in data.var.
        
    Returns
    -------
    pd.DataFrame
    """
    sim = data.uns['coloc_sim_agg']
    return sim.loc[sim['g1'].isin(genes) & sim['g2'].isin(genes)]

# TODO need physical unit size of coordinate system to standardize rendering resolution
def rasterize_cells(
    data,
    imgdir,
    label_layer=None,
    scale_factor=15,
    out_dim=64,
    n_cores=1,
    overwrite=True,
    copy=False,
):
    """Rasterize points and cell masks to grayscale image. Writes directly to file.

    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial data.
    imgdir : str
        Directory where images will be stored.
    """
    adata = data.copy() if copy else data

    os.makedirs(f"{imgdir}", exist_ok=True)

    def write_img(s, n, p, cell_name):

        # Get bounds and size of cell in raw coordinate space
        bounds = s.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # Define top left corner for centering/scaling transform
        west = bounds[0] + width / 2 - (out_dim / 2 * scale_factor)
        north = bounds[3] - height / 2 + (out_dim / 2 * scale_factor)

        # Define transform
        tf_origin = rasterio.transform.from_origin(
            west, north, scale_factor, scale_factor
        )

        # Rasterize cell
        base_raster = features.rasterize(
            [s],
            fill=0,
            default_value=20,
            out_shape=(out_dim, out_dim),
            transform=tf_origin,
        )

        # Rasterize nucleus
        if n is not None:
            features.rasterize(
                [n], default_value=40, transform=tf_origin, out=base_raster
            )

        warnings.filterwarnings(
            action="ignore", category=rasterio.errors.NotGeoreferencedWarning
        )

        # Rasterize and write points
        genes = p["gene"].unique().tolist()

        if label_layer:
            labels = dict(
                zip(genes, list(adata[cell_name, genes].layers[label_layer].flatten()))
            )
        else:
            labels = dict(zip(genes, ["foo"] * len(genes)))

        p = geopandas.GeoDataFrame(p, geometry=geopandas.points_from_xy(p["x"], p["y"]))

        for gene_name in genes:
            label = labels[gene_name]

            os.makedirs(f"{imgdir}/{label}", exist_ok=True)

            # TODO implement overwrite param
            if not overwrite and os.path.exists(
                f"{imgdir}/{label}/{cell_name}_{gene_name}.tif"
            ):
                return

            cg_points = p.loc[p["gene"] == gene_name]

            gene_raster = base_raster.copy()

            # Set base as 40
            gene_raster = features.rasterize(
                shapes=cg_points.geometry,
                default_value=40,
                transform=tf_origin,
                out=gene_raster,
            )

            # Plus 20 per point
            features.rasterize(
                shapes=cg_points.geometry,
                default_value=20,
                transform=tf_origin,
                merge_alg=rasterio.enums.MergeAlg("ADD"),
                out=gene_raster,
            )

            # Convert to tensor
            gene_raster = torch.from_numpy(gene_raster.astype(np.float32) / 255)

            torchvision.utils.save_image(
                gene_raster, f"{imgdir}/{label}/{cell_name}_{gene_name}.tif"
            )

    # Parallelize points
    Parallel(n_jobs=n_cores)(
        delayed(write_img)(
            adata.obs.loc[cell_name, "cell_shape"],
            adata.obs.loc[cell_name, "nucleus_shape"],
            get_points(adata, cells=cell_name),
            cell_name,
        )
        for cell_name in tqdm(adata.obs_names.tolist())
    )

    # TODO write filepaths to adata

    return adata if copy else None
