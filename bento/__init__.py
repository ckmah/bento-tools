import logging as log
import math
import os
from collections import defaultdict
from copy import deepcopy

import cv2 as cv
import numpy as np
import pandas as pd
from anndata import AnnData

from . import util

os.environ['KERAS_BACKEND'] = 'tensorflow'



def prepare(spots_path, cell_img_path, nucleus_img_path, space='polar'):
    """Prepare an AnnData object of spots in computed feature space.

    Parameters
    ----------
    spots_path : str
        Filepath to spots csv file. Columns should follow (dim1, dim2, gene) format.
    cell_img_path : str
        Filepath to cell segmentation image.
    nucleus_img_path : str
        Filepath to nucleus segmentation image.
    space : str
        See to_feature_space for parameter options. By default 'polar'.

    Returns
    -------
    AnnData
        [genes x features] (row by column) AnnData object
    """

    # * Load all data
    # sparse style csv (x, y, gene)
    df = pd.read_csv(spots_path)
    cell_img = cv.imread(cell_img_path)
    nucleus_img = cv.imread(nucleus_img_path)

    # * Prepare feature space
    X = to_feature_space(df, cell_img=cell_img,
                         nucleus_img=nucleus_img, space=space)

    X.index = [f'{i}-1' for i in X.index]

    # * Observation annotations [Genes]

    # * Variable annotations [Coordinate system]
    var = pd.DataFrame()
    var_names = X.columns
    var['name'] = var_names
    var['feature_ids'] = var_names  # compatibility formatting for 10x
    var['feature_types'] = space  # for spatial data, coordinate type
    var.set_index('name', inplace=True)
    # * Create AnnData object
    adata = AnnData(X=X, var=var)

    return adata


def to_feature_space(df, cell_img, nucleus_img, space='polar'):
    """[summary]

    Parameters
    ----------
    df : [type]
        [description]
    cell_img : [type]
        [description]
    nucleus_img : [type]
        [description]
    space : str, optional
        Options include 'polar', 'logpolar', 'polar_norm' and 'logpolar_norm'. By default 'polar'.

    Returns
    -------
    [type]
        [description]
    """
    # Compile coordinate lists
    coords = defaultdict()
    coords['df'] = df.to_numpy()[:, :2]

    # Find segmentation contours
    for img, img_name in zip([cell_img, nucleus_img], ['cell', 'nucleus']):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        contour = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[
            0][0].flatten().reshape(-1, 2)
        coords[img_name] = contour

    # Center all to centroid
    m = cv.moments(coords['nucleus'])
    centroid = [m['m10']/m['m00'], m['m01']/m['m00']]
    x = df.iloc[:, 0] - centroid[0]
    y = df.iloc[:, 1] - centroid[1]

    # * Convert to specified space
    for name, c in coords.items():
        x = c[:, 0] - centroid[0]
        y = c[:, 1] - centroid[1]

        # Polar
        polar = np.array(util.cart2pol(x, y))
        polar = np.transpose(polar)
        polar = pd.DataFrame(polar, columns=['radius', 'angle'])

        # Log polar
        if space == 'logpolar' or space == 'logpolar_norm':
            polar['radius'] = np.log2(polar['radius'])

        coords[name] = polar

    # * Normalize everything to cell boundary
    if space == 'polar_norm' or space == 'logpolar_norm':
        cell_radius = deepcopy(coords['cell'].radius)
        cell_angle = deepcopy(coords['cell'].angle)

        # Normalize each set of coordinates
        # ? very slow
        for name, c in coords.items():
            c.radius = c.apply(
                lambda x: x.radius / cell_radius[abs(x.angle - cell_angle) < 0.1].mean(), axis=1)
            coords[name] = c

    # * Perform convolution / blur
    GRID_SIZE = 20
    c = coords['df']
    c[df.columns[2]] = df.iloc[:, 2]
    ANGLE_BIN_SIZE = 6.3 / GRID_SIZE
    RADIUS_BIN_SIZE = (c.radius.max() -
                       c.radius.min()) / GRID_SIZE

    # Bin radius
    c.radius = c.radius.apply(lambda r: min(
        GRID_SIZE - 1, math.floor(r / RADIUS_BIN_SIZE)))

    # Bin angle
    c.angle = c.angle - c.angle.min()
    c.angle = c.angle.apply(lambda a: min(
        GRID_SIZE - 1, math.floor(a / ANGLE_BIN_SIZE)))

    # Count in each bin
    counts = c.groupby(c.columns.tolist()).size()

    # Populate grid
    grids = defaultdict()
    for (i, j, gene), count in counts.items():
        if gene not in grids:
            grids[gene] = np.zeros((GRID_SIZE, GRID_SIZE))
        grids[gene][i][j] = count

    for gene, g in grids.items():
        grids[gene] = g.flatten()
    # Return converted coordinate sets
    return pd.DataFrame.from_dict(grids).T
