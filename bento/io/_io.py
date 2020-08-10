import numpy as np
import pandas as pd
import geopandas
from shapely import geometry, wkt
from shapely.ops import unary_union
from pandarallel import pandarallel

from ast import literal_eval

import anndata
from anndata import AnnData

from .._settings import settings

pandarallel.initialize(nb_workers=settings.n_cores, progress_bar=settings.progress_bar, verbose=0)


def read_h5ad(filename):
    """Load bento processed AnnData object from h5ad. Casts DataFrames in adata.uns['masks'] to GeoDataFrame.

    Parameters
    ----------
    filename : str
        File name to load data file.

    Returns
    -------
    AnnData
        AnnData data object.
    """
    adata = anndata.read_h5ad(filename)

    # Converts geometry column from str wkt format back to GeoSeries to enable GeoPandas functionality
    for m in adata.uns['masks']:
        adata.uns['masks'][m]['geometry'] = adata.uns['masks'][m]['geometry'].apply(
            wkt.loads)
        adata.uns['masks'][m] = geopandas.GeoDataFrame(
            adata.uns['masks'][m], geometry='geometry')

    # if 'labels' in adata.uns:
    #     adata.uns['labels'].index = pd.MultiIndex.from_tuples([literal_eval(i) for i in adata.uns['labels'].index], names=['cell', 'gene'])

    return adata

def write_h5ad(adata, filename):
    """Write AnnData to h5ad. Casts each GeoDataFrame in adata.uns['masks'] for h5ad compatibility.

    Parameters
    ----------
    adata : AnnData
        bento loaded AnnData
    filename : str
        File name to write data file.
    """
    # Convert geometry from GeoSeries to list for h5ad serialization compatibility
    for m in adata.uns['masks']:
        adata.uns['masks'][m]['geometry'] = adata.uns['masks'][m]['geometry'].apply(lambda x: x.wkt).astype(str)

    # Write to h5ad
    adata.write(filename)

def read_geodata(points, cell, other={}, index=True):
    """Load spots and masks for many cells.

    Parameters
    ----------
    points : str
        Filepath to spots .shp file. Expects GeoDataFrame with geometry of Points, and 'gene' column at minimum.
    cell : str
        Filepath to cell segmentation masks .shp file. Expects GeoDataFrame with geometry of Polygons.
    other : dict(str)
        Filepaths to all other segmentation masks .shp files; expects GeoDataFrames of same format.
        Use keys of dict to access corresponding outputs.
    index : bool
        do not index if disjoint e.g. cells coordinates are not relative to one another. Assumes points are already indexed.
    Returns
    -------
        AnnData object
    """
    print('Loading points...')
    points = geopandas.read_file(points)

    # Load masks
    print('Loading masks...')
    mask_paths = {'cell': cell, **other}
    masks = pd.Series(mask_paths).parallel_apply(_load_masks)

    if index:
        # Index points for all masks
        print('Indexing points...')
        point_index = masks.parallel_apply(lambda mask: _index_points(points[['geometry']], mask)).T

        # Index masks to cell
        print('Indexing masks...')
        mask_index = _index_masks(masks)
    else:
        # assume points are pre-indexed to masks, and all masks are indexed to cells
        point_index = points['cell']

        other_masks = masks[masks.index != 'cell']
        other_masks = other_masks.to_dict()
        mask_index = {}
        for m, mask in other_masks.items():
            mask_index[m] = mask['cell'].to_frame()

    # Create AnnData object
    X = points[['x', 'y']]
    uns = {'masks': masks.to_dict(), 'mask_index': mask_index}
    obs = pd.DataFrame(point_index)
    obs['gene'] = points['gene']

    adata = AnnData(X=X, obs=obs, uns=uns)

    print('Done.')
    return adata


def _load_masks(path):
    """Load GeoDataFrame from path.

    Parameters
    ----------
    path : str
        Path to .shp file.

    Returns
    -------
    GeoDataFrame
        Contains masks as Polygons.
    """
    mask = geopandas.read_file(path)

    for i, poly in enumerate(mask['geometry']):
        if type(poly) == geometry.MultiPolygon:
            print(f'Object at index={i} is a MultiPolygon.')
            print(poly)
            return

    # Cleanup polygons
    # mask.geometry = mask.geometry.buffer(2).buffer(-2)
    # mask.geometry = mask.geometry.apply(unary_union)

    return mask


def _index_masks(masks):
    cell = masks['cell']

    mask_index = {}
    for m, mask in masks.items():
        if m != 'cell':
            index = geopandas.sjoin(mask.reset_index(), cell, how='left', op='intersects')
            index = index.drop_duplicates(subset='index', keep='first')
            index = index.sort_index()
            index = index.reset_index()['index_right']
            index.name = 'cell'
            index = index.fillna(-1).astype(int)

            mask_index[m] = pd.DataFrame(index)

    return mask_index

def _index_points(points, mask):
    """Index points to each mask item and save. Assumes non-overlapping masks.

    Parameters
    ----------
    points : GeoDataFrame
        Point coordinates.
    mask : GeoDataFrame
        Mask polygons.
    Returns
    -------
    Series
        Return list of mask indices corresponding to each point.
    """
    index = geopandas.sjoin(points.reset_index(), mask, how='left', op='intersects')

    # remove multiple cells assigned to same point
    index = index.drop_duplicates(subset='index', keep="first")
    index = index.sort_index()
    index = index.reset_index()['index_right']
    index = index.fillna(-1).astype(int)

    return pd.Series(index)
