import numpy as np
import cv2 as cv
import pandas as pd
import geopandas
from shapely import geometry

def read_geodata(spots_path, cells_path, nucleus_path):
    """
    Load spots and masks for many cells.
    """
    points = geopandas.read_file(spots_path)
    cells = geopandas.read_file(cells_path).geometry
    nuclei = geopandas.read_file(nucleus_path).geometry
    cell_ids = range(0, len(cells))
    
    return {'points': points, 'cell_id': cell_ids, 'cell': cells, 'nucleus': nuclei}

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
  