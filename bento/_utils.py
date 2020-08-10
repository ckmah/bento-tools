import matplotlib.path as mplPath
import numpy as np
from astropy.stats import RipleysKEstimator


def quantify_variable(data, mask_name, variable):
    """Quantify variable on points binned by mask Polygons.

    Parameters
    ----------
    points : GeoDataFrame
        Geometry must be point coordinates. Must have 'cell' column for binning.
    variable : str
        Must be one of ['npoints', 'ngenes']
    """
    adata = data.copy()
    if variable == 'npoints':
        adata.uns['masks'][mask_name][variable] = adata.obs.groupby(
            mask_name).apply(len)
    elif variable == 'ngenes':
        adata.uns['masks'][mask_name][variable] = adata.obs.groupby(mask_name)['gene'].apply(
            lambda gene_labels: len(gene_labels.unique()))
    else:
        print(f'Variable {variable} not recognized.')
        return

    return adata


def _poly2grid(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    x, y = np.meshgrid(np.arange(minx, maxx, step=np.float(20)),
                       np.arange(miny, maxy, step=np.float(20)))
    x = x.flatten()
    y = y.flatten()
    xy = np.array([x, y]).T
    polygon_path = mplPath.Path(np.array(polygon.exterior.xy).T)
    polygon_cell_mask = polygon_path.contains_points(xy)
    xy = xy[polygon_cell_mask]

    return xy


def _ripley(points, mask, radii=None):
    """
    Calculate estimation for Ripley H function (0-centered L function).

    Parameters
    ----------
    points : 2d array
    mask : shapely.Polygon
    radii : float list

    Returns
    -------
    list
        ripley l function
    radii
    """
    estimator = RipleysKEstimator(area=mask.area,
                                  x_min=float(points[:, 0].min()), x_max=float(points[:, 0].max()),
                                  y_min=float(points[:, 1].min()), y_max=float(points[:, 1].max()))

    # https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator
    # if radii is None:
    #     radii = np.linspace(1, np.sqrt(mask.area / 2) ** 0.5, 200)
    ripley = estimator.Hfunction(data=points, radii=radii, mode='none')

    return ripley, radii
