from abc import ABCMeta, abstractmethod

import matplotlib.path as mplPath
from astropy.stats.spatial import RipleysKEstimator
from scipy.spatial.kdtree import distance_matrix
from scipy.stats.stats import spearmanr
from scipy.spatial import distance

import dask_geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from .. import tools as tl
from .._utils import track
from ..preprocessing import get_points

from tqdm.auto import tqdm


class AbstractFeature(metaclass=ABCMeta):

    __point_metadata = pd.Series(["x", "y", "cell", "gene", "nucleus"])

    @abstractmethod
    def __init__(self, shape_name):
        self._shape_name = shape_name
        self.metadata = None

    @property
    def shape_name(self):
        return self._shape_name

    @shape_name.setter
    def shape_name(self, s):
        self._shape_name = s

    @abstractmethod
    def extract(self, points):
        """Given a set of points, extract and return one or more values.

        Parameters
        ----------
        points : DataFrame
            Point coordinates.

        Returns
        -------
        values
            An interable of values.
        """
        # Cast to GeoDataFrame for spatial operations
        points = gpd.GeoDataFrame(
            points, geometry=gpd.points_from_xy(points.x, points.y)
        )
        return points

    @abstractmethod
    def precompute(self, data, points):
        """Precompute cell-level features. This prevents recomputing once per sample. Also map shape to points."""
        # Save shape to points column
        points = (
            points.set_index("cell")
            .join(data.obs[self.shape_name])
            .reset_index()
            .sort_values(["cell", "gene"])
        )

        # Cast to GeoDataFrame for spatial operations
        points = gpd.GeoDataFrame(
            points, geometry=gpd.points_from_xy(points.x, points.y)
        )

        # Cast categorical type to save memory
        cat_vars = ["cell", "gene", "nucleus"]
        for v in cat_vars:
            points[v] = points[v].astype("category").cat.as_ordered()

        points = points.set_index("cell")
        return data, points

    @track
    def transform(self, data, copy=False):
        """Applies self.extract() to all points grouped by cell and gene.

        Parameters
        ----------
        data : spatial formatted AnnData
            data.uns['points'] must be DataFrame with minimum columns x, y, cell, and gene.
        shape_name : str
            Column name in data.obs referring to shape to use in calculation.
        """
        adata = data.copy() if copy else data

        # Only include points for samples in data.obs_names and data.var_names
        points = get_points(adata)

        # Check points DataFrame for missing columns
        if not self.__point_metadata.isin(points.columns).all():
            raise KeyError(
                f"'points' DataFrame needs to have all columns: {self.__point_metadata.tolist()}."
            )

        adata, points = self.precompute(adata, points)

        ngroups = points.groupby(["cell", "gene"]).ngroups
        if ngroups > 100:
            npartitions = min(1000, ngroups)

            out = (
                dask_geopandas.from_geopandas(points, npartitions=npartitions)
                .groupby(["cell", "gene"])
                .apply(self.extract, meta=self.metadata)
            )

            #             from dask.diagnostics import Profiler, CacheProfiler, ResourceProfiler, visualize
            #             with ProgressBar() as p, Profiler() as prof, CacheProfiler() as cprof, ResourceProfiler() as rprof:
            with ProgressBar():
                feature_df = out.compute()
        #                 visualize([prof, cprof, rprof])

        else:
            tqdm.pandas()
            feature_df = points.groupby(["cell", "gene"]).progress_apply(self.extract)

        feature_df = feature_df.reset_index().drop("tmp_index", axis=1)

        # Save results to data layers
        feature_names = feature_df.columns[~feature_df.columns.isin(["cell", "gene"])]
        for feature_name in feature_names:
            adata.layers[feature_name] = (
                feature_df.pivot(index="cell", columns="gene", values=feature_name)
                .reindex(index=adata.obs_names, columns=adata.var_names)
                .astype(float)
            )

        return adata


class ShapeProximity(AbstractFeature):
    def __init__(self, shape_name):
        super().__init__(shape_name)
        shape_prefix = shape_name.split(sep="_shape")[0]
        self.metadata = {
            f"{shape_prefix}_inner_proximity": float,
            f"{shape_prefix}_outer_proximity": float,
        }

    def extract(self, points):
        """Given a set of points, calculate and return the average proximity between points outside to the shape boundary, and points inside to the shape boundary.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.

        Return
        ------
        inner_proximity : float
            Value ranges between [0, 1]. Value closer to 0 denotes farther from shape boundary, value closer to 1 denotes close to shape boundary.
        outer_proximity : float
            Value ranges between [0, 1]. Value closer to 0 denotes farther from shape boundary, value closer to 1 denotes close to shape boundary.
        """
        points = super().extract(points)

        # Get shape polygon
        shape = points[self.shape_name].values[0]

        # Get points outside shape
        inner = points.within(shape)
        outer = ~inner

        inner_dist = points[inner].distance(shape.boundary).mean()
        outer_dist = points[outer].distance(shape.boundary).mean()

        # Scale from [0, 1], where 1 is close and 0 is far.
        cell_radius = points["cell_radius"].values[0]
        inner_proximity = (cell_radius - inner_dist) / cell_radius
        outer_proximity = (cell_radius - outer_dist) / cell_radius

        if np.isnan(inner_proximity):
            inner_proximity = 0

        if np.isnan(outer_proximity):
            outer_proximity = 0

        shape_prefix = self.shape_name.split("_shape")[0]
        return pd.DataFrame(
            {
                f"{shape_prefix}_inner_proximity": inner_proximity,
                f"{shape_prefix}_outer_proximity": outer_proximity,
            },
            index=pd.Index([0], name="tmp_index"),
        )

    def precompute(self, data, points):
        data, points = super().precompute(data, points)

        tl.cell_radius(data)

        # Join attributes to points
        shape_prefix = self.shape_name.split("_shape")[0]
        points = points.join(data.obs["cell_radius"])

        return data, points


class ShapeAsymmetry(AbstractFeature):
    def __init__(self, shape_name):
        super().__init__(shape_name)
        shape_prefix = shape_name.split(sep="_shape")[0]
        self.metadata = {
            f"{shape_prefix}_inner_asymmetry": float,
            f"{shape_prefix}_outer_asymmetry": float,
        }

    def extract(self, points):
        """Given a set of points, calculate and return the measure of symmetry of points outside the shape boundary, and points inside the shape boundary.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.

        Return
        ------
        asymmetry : float
            Values [0, 1], where 1 is asymmetrical and 0 is symmetrical.
        """
        points = super().extract(points)

        # Get shape polygon
        shape = points[self.shape_name].values[0]

        # Get points in same major subcellular compartment
        shape_prefix = self.shape_name.split("_shape")[0]

        # Get points outside shape
        inner = points.within(shape)
        outer = ~inner

        inner_to_centroid = points[inner].distance(shape.centroid).mean()
        outer_to_centroid = points[outer].distance(shape.centroid).mean()

        # Values [0, 1], where 1 is asymmetrical and 0 is symmetrical.
        cell_radius = points["cell_radius"].values[0]
        inner_asymmetry = inner_to_centroid / cell_radius
        outer_asymmetry = outer_to_centroid / cell_radius

        if np.isnan(inner_asymmetry):
            inner_asymmetry = 0

        if np.isnan(outer_asymmetry):
            outer_asymmetry = 0

        return pd.DataFrame(
            {
                f"{shape_prefix}_inner_asymmetry": inner_asymmetry,
                f"{shape_prefix}_outer_asymmetry": outer_asymmetry,
            },
            index=pd.Index([0], name="tmp_index"),
        )

    def precompute(self, data, points):
        data, points = super().precompute(data, points)

        tl.cell_radius(data)

        # Join attributes to points
        shape_prefix = self.shape_name.split("_shape")[0]
        points = points.join(data.obs["cell_radius"])

        return data, points


class PointDispersion(AbstractFeature):
    def __init__(self, shape_name=None):
        shape_name = "cell_shape"
        super().__init__("cell_shape")
        self.metadata = {
            "point_dispersion": float,
        }

    def extract(self, points):
        """Given a set of points, calculate the normalized central second moment.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates.

        Return
        ------

        """
        points = super().extract(points)

        # Get precomputed centroid and cell moment
        pt_centroid = points[["x", "y"]].values.mean(axis=0).reshape(1, 2)
        cell_coords = points["cell_coords"].values[0]

        # calculate points moment
        point_moment = _second_moment(pt_centroid, points[["x", "y"]].values)
        cell_moment = _second_moment(pt_centroid, cell_coords)

        # Normalize by cell moment
        norm_moment = point_moment / cell_moment

        return pd.DataFrame(
            {
                f"point_dispersion": norm_moment,
            },
            index=pd.Index([0], name="tmp_index"),
        )

    def precompute(self, data, points):
        data, points = super().precompute(data, points)

        # Second moment of cell relative to shape centroid
        cell_coords = data.obs["cell_shape"].apply(_raster_polygon)

        cell_coords = pd.Series(
            cell_coords,
            index=data.obs_names,
            name="cell_coords",
        )

        # Join attributes to points
        points = points.join(cell_coords)

        return data, points


class ShapeDispersion(AbstractFeature):
    def __init__(self, shape_name):
        super().__init__(shape_name)
        shape_prefix = shape_name.split(sep="_shape")[0]
        self.metadata = {
            f"{shape_prefix}_dispersion": float,
        }

    def extract(self, points):
        """Given a set of points, calculate the normalized central second moment (analogous to MSE) in reference to a shape.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates.

        Return
        ------

        """
        points = super().extract(points)
        shape_prefix = self.shape_name.split("_shape")[0]

        # Get precomputed centroid and cell moment
        ref_centroid = points[f"{shape_prefix}_centroid"].iloc[0]
        cell_moment = points["cell_moments"].iloc[0]

        # calculate points moment
        point_moment = _second_moment(ref_centroid, points[["x", "y"]].values)

        # Normalize by cell moment
        norm_moment = point_moment / cell_moment

        return pd.DataFrame(
            {
                f"{shape_prefix}_dispersion": norm_moment,
            },
            index=pd.Index([0], name="tmp_index"),
        )

    def precompute(self, data, points):
        data, points = super().precompute(data, points)
        shape_prefix = self.shape_name.split("_shape")[0]

        # Second moment of cell relative to shape centroid
        cell_rasters = data.obs["cell_shape"].apply(_raster_polygon)
        shape_centroids = gpd.GeoSeries(data.obs[self.shape_name]).centroid
        cell_moments = [
            _second_moment(np.array(centroid.xy).reshape(1, 2), cell_coords)
            for centroid, cell_coords in zip(shape_centroids, cell_rasters)
        ]
        cell_stats = pd.DataFrame(
            {"cell_moments": cell_moments, f"{shape_prefix}_centroid": shape_centroids},
            index=data.obs_names,
        )

        # Join attributes to points
        points = points.join(cell_stats)

        return data, points


def _second_moment(centroid, pts):
    """
    Calculate second moment of points with centroid as reference.

    Parameters
    ----------
    centroid : [1 x 2] float
    pts : [n x 2] float
    """
    centroid = np.array(centroid).reshape(1, 2)
    radii = distance.cdist(centroid, pts)
    second_moment = np.sum(radii * radii / len(pts))
    return second_moment


def _raster_polygon(polygon):
    """
    Rasterize polygon and return list of coordinates in body of polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    x, y = np.meshgrid(
        np.arange(minx, maxx, step=np.float(1)),
        np.arange(miny, maxy, step=np.float(1)),
    )
    x = x.flatten()
    y = y.flatten()
    xy = np.array([x, y]).T
    polygon_path = mplPath.Path(np.array(polygon.exterior.xy).T)
    polygon_cell_mask = polygon_path.contains_points(xy)
    xy = xy[polygon_cell_mask]

    return xy


class Ripley(AbstractFeature):
    def __init__(self, shape_name=None):
        # Not used but keep for consistency
        shape_name = "cell_shape"
        super().__init__(shape_name)
        self.metadata = {
            "l_max": float,
            "l_max_gradient": float,
            "l_min_gradient": float,
            "l_monotony": float,
            "l_half_radius": float,
        }

    def extract(self, points):
        """Given a set of points, calculate statistics of Ripley's L-function.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates.

        Return
        ------
        l_max : float
        l_max_gradient : float
        l_min_gradient : float
        l_monotony : float
        l_half_radius : float
        """
        points = super().extract(points)

        # Get shape polygon
        shape = points[self.shape_name].values[0]
        minx, miny, maxx, maxy = shape.bounds
        shape_coo = np.array(shape.exterior.coords.xy).T

        estimator = RipleysKEstimator(
            area=shape.area,
            x_min=minx,
            x_max=maxx,
            y_min=miny,
            y_max=maxy,
        )

        # Compute ripley function for r=(1, cell diameter / 2), step size = 1 pixel
        l_4_dist = int(distance_matrix(shape_coo, shape_coo).max() / 4)
        radii = np.linspace(1, l_4_dist * 2, num=l_4_dist * 2)
        stats = estimator.Hfunction(
            data=points[["x", "y"]].values, radii=radii, mode="none"
        )

        # Max value of the L-function
        l_max = max(stats)

        # Max and min value of the gradient of L
        ripley_smooth = pd.Series(stats).rolling(5).mean()
        ripley_smooth.dropna(inplace=True)

        # Can't take gradient of single number
        if len(ripley_smooth) < 2:
            ripley_smooth = np.array([0, 0])

        ripley_gradient = np.gradient(ripley_smooth)
        l_max_gradient = ripley_gradient.max()
        l_min_gradient = ripley_gradient.min()

        # Monotony of L-function in the interval
        l_monotony = spearmanr(radii, stats)[0]

        # L-function at L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
        l_half_radius = estimator.Hfunction(
            data=points[["x", "y"]].values, radii=[l_4_dist], mode="none"
        )[0]

        return pd.DataFrame(
            {
                "l_max": l_max,
                "l_max_gradient": l_max_gradient,
                "l_min_gradient": l_min_gradient,
                "l_monotony": l_monotony,
                "l_half_radius": l_half_radius,
            },
            index=pd.Index([0], name="tmp_index"),
        )

    def precompute(self, data, points):
        data, points = super().precompute(data, points)

        return data, points


class ShapeEnrichment(AbstractFeature):
    def __init__(self, shape_name=None):
        super().__init__(shape_name)
        shape_prefix = shape_name.split(sep="_shape")[0]
        self.metadata = {
            f"{shape_prefix}_enrichment": float,
        }

    def extract(self, points):
        """Given a set of points, count the fraction of transcripts inside.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates.

        Return
        ------
        shape_enrichment : float
            Value ranges between [0, 1] 1 denotes all points in shape.

        """
        points = super().extract(points)
        shape_prefix = self.shape_name.split(sep="_shape")[0]

        # Get shape polygon
        shape = points[self.shape_name].values[0]

        # Get points inside shape
        n_points = sum(points.within(shape))
        enrichment = n_points / len(points)

        if np.isnan(enrichment):
            enrichment = 0
            
        return pd.DataFrame(
            {
                f"{shape_prefix}_enrichment": enrichment,
            },
            index=pd.Index([0], name="tmp_index"),
        )

    def precompute(self, data, points):
        data, points = super().precompute(data, points)

        return data, points
