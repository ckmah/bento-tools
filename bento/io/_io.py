import numpy as np
import pandas as pd
import geopandas
from shapely import geometry, wkt
from shapely.ops import unary_union

from ast import literal_eval

import anndata
from anndata import AnnData

from .._settings import settings
from .._settings import pandarallel


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

    adata.obs.index = adata.obs.index.astype(str)

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
        if type(adata.uns['masks'][m]['geometry'][0]) != str:
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

    # Cast cell and indexing references to str
    obs.index = obs.index.astype(str)
    obs['cell'] = obs['cell'].astype(int).astype(str)

    adata = AnnData(X=X, obs=obs, uns=uns)

    # Convert geometry from GeoSeries to list for h5ad serialization compatibility
    for m in adata.uns['masks']:
        if type(adata.uns['masks'][m]['geometry'][0]) != str:
            adata.uns['masks'][m]['geometry'] = adata.uns['masks'][m]['geometry'].apply(lambda x: x.wkt).astype(str)

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
    mask.index = mask.index.astype(str)

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
            index.index = index.index.astype(str)
            index = index.fillna(-1).astype(str)

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
    index = index.fillna(-1).astype(str)

    return pd.Series(index)


def concatenate(adatas):
    for i, adata in enumerate(adatas):
        for mask in adata.uns['masks'].keys():
            
            adata.obs[mask] = [f'{i}-{x}' if x != '-1' else x for x in adata.obs[mask]]
            adata.uns['masks'][mask].index = [f'{i}-{x}' for x in adata.uns['masks'][mask].index]
            
            if mask != 'cell':
                adata.uns['mask_index'][mask].index = [f'{i}-{x}' for x in adata.uns['mask_index'][mask].index]
                adata.uns['mask_index'][mask]['cell'] = [f'{i}-{x}' for x in adata.uns['mask_index'][mask]['cell']]
            
    uns = dict()
    uns['masks'] = dict()
    uns['mask_index'] = dict()
    for mask in adatas[0].uns['masks'].keys():
        # Concat mask GeoDataFrames
        uns['masks'][mask] = pd.concat([adata.uns['masks'][mask] for adata in adatas])
        
        # Concat mask_index DataFrames
        if mask != 'cell':
            uns['mask_index'][mask] = pd.concat([adata.uns['mask_index'][mask] for adata in adatas])

    new_adata = adatas[0].concatenate(adatas[1:])
    new_adata.uns = uns

    return new_adata