import numpy as np
import cv2 as cv
import pandas as pd
import geopandas
from shapely import geometry
from shapely.ops import unary_union
from pandarallel import pandarallel

from .._settings import settings

pandarallel.initialize(nb_workers=settings.n_cores, verbose=0)


def read_geodata(spots_path, cells_path, nucleus_path):
    """Load spots and masks for many cells.


    Parameters
    ----------
    spots_path : str
        Filepath to spots .shp file.
    cells_path : str
        Filepath to cell segmentation masks .shp file.
    nucleus_path : str
        Filepath to nucleus segmentation masks .shp file.

    Returns
    -------
        dict : keys = points, cell_id, cell, nucleus
    """

    points = geopandas.read_file(spots_path)
    cell = geopandas.read_file(cells_path)
    nucleus = geopandas.read_file(nucleus_path)
    nucleus = nucleus.set_geometry(nucleus.buffer(0))
    
    cell = _clean_polygons(cell)
    nucleus = _clean_polygons(nucleus)
    
    # Assign points to cells, default to -1 to denote not in cell
    print('Indexing cell segmentation masks...\r')
    points, cell, nucleus, cell_ids = _assign_cell_id(points, cell, nucleus)
    
    print('Done.')
    return {'points': points, 'cell_id': cell_ids, 'cell': cell, 'nucleus': nucleus}


def _clean_polygons(gdf):
    # remove self-intersections
    gdf.buffer(0)
    
    # simplify MultiPolygon to Polygon (remove artifacts)
    gdf.geometry = gdf.geometry.apply(unary_union)

    return gdf

def _assign_cell_id(points, cell, nucleus):
    
    def assign_cell_id(c):
        c_poly = c['geometry']
        c_id = c['cell_id']
        # Assign cell_id to each point
        subset_pids = geopandas.clip(points, c_poly).index
        points.loc[subset_pids, 'cell_id'] = c_id

        # Assign cell_id to each nucleus
        subset_nucleus = geopandas.clip(nucleus, c_poly).geometry
        subset_nucleus_area = subset_nucleus.apply(lambda n: n.intersection(c_poly).area/n.area)
        subset_nid = subset_nucleus_area.idxmax()
        nucleus.loc[subset_nid, 'cell_id'] = c_id
        
        # TODO assign cell ids by position in fov
    
    cell['cell_id'] = range(0, len(cell))
    points['cell_id'] = -1
    nucleus['cell_id'] = -1
    
    # TODO parallelize
    cell.apply(lambda c: assign_cell_id(c), axis=1)

    return points, cell, nucleus, cell['cell_id'].values

def read_imgs(spots_path, cell_img_path, nucleus_img_path):
    """Load spots and masks for a single cell.

    Parameters
    ----------
    spots_path : str
        Filepath to spots csv file. Columns should follow (dim1, dim2, gene) format.
    cell_img_path : str
        Filepath to cell segmentation image.
    nucleus_img_path : str
        Filepath to nucleus segmentation image.

    Returns
    -------
    dict
    points : GeoDataFrame
        mRNA points labeled by gene.
    polys : GeoSeries
        Cell and nucleus segmentation masks in that order.
    """

    # * Load points csv (x, y, gene)
    df = pd.read_csv(spots_path)
    df.columns = ['x', 'y', 'gene']
    points = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y))

    # Find segmentation contours
    polys = []
    for img_path in [cell_img_path, nucleus_img_path]:
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        contour = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[
            0][0].flatten().reshape(-1, 2)
        poly = geometry.Polygon(contour)
        polys.append(poly)

    return {'points': points, 'cell': polys[0], 'nucleus': polys[1]}
