from abc import ABCMeta, abstractmethod

from astropy.stats.spatial import RipleysKEstimator
from scipy.spatial.kdtree import distance_matrix
from scipy.stats.stats import spearmanr

import dask_geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm

from .. import tools as tl
from .._utils import track


class AbstractFeature(metaclass=ABCMeta):

    __point_metadata = pd.Series(["x", "y", "cell", "gene", "nucleus"])

    @abstractmethod
    def __init__(self, s_name):
        self._shape_name = s_name
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
        points[cat_vars] = points[cat_vars].astype("category")

        return data, points

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

        points = adata.uns["points"].copy()

        # Check points DataFrame for missing columns
        if not self.__point_metadata.isin(points.columns).all():
            raise KeyError(
                f"'points' DataFrame needs to have all columns: {self.__point_metadata.tolist()}."
            )

        adata, points = self.precompute(adata, points)

        out = (
            dask_geopandas.from_geopandas(points, chunksize=10000)
            .groupby(["cell", "gene"])
            .apply(self.extract, meta=self.metadata)
            .reset_index()
        )

        with ProgressBar():
            feature_df = out.compute()

        feature_df = feature_df.drop("tmp_index", axis=1)

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

        Only considers points inside the same major subcellular compartment (cytoplasm or nucleus).

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

        # Get points in same major subcellular compartment
        shape_prefix = self.shape_name.split("_shape")[0]
        if points[f"{shape_prefix}_in_nucleus"].values[0]:
            in_compartment = points["nucleus"] != "-1"
        else:
            in_compartment = points["nucleus"] == "-1"

        # Get points outside shape
        inner = points.within(shape)
        outer = ~inner

        inner_dist = points[in_compartment & inner].distance(shape.boundary).mean()
        outer_dist = points[in_compartment & outer].distance(shape.boundary).mean()

        # Scale from [0, 1], where 1 is close and 0 is far.
        cell_radius = points["cell_radius"].values[0]
        inner_proximity = (cell_radius - inner_dist) / cell_radius
        outer_proximity = (cell_radius - outer_dist) / cell_radius

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
        tl.is_nuclear(data, self.shape_name)

        # Join attributes to points
        shape_prefix = self.shape_name.split("_shape")[0]
        points = (
            points.set_index("cell")
            .join(data.obs[["cell_radius", f"{shape_prefix}_in_nucleus"]])
            .reset_index()
        )

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

        Only considers points inside the same major subcellular compartment (cytoplasm or nucleus).

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.

        Return
        ------
        asymmetry : float
            Value ranges between [0, inf]. 0 denotes symmetry, values between 0 and 1 denotes strong asymmetry, >1 denotes very strong asymmetry.
        """
        points = super().extract(points)

        # Get shape polygon
        shape = points[self.shape_name].values[0]

        # Get points in same major subcellular compartment
        shape_prefix = self.shape_name.split("_shape")[0]
        if points[f"{shape_prefix}_in_nucleus"].values[0]:
            in_compartment = points["nucleus"] != "-1"
        else:
            in_compartment = points["nucleus"] == "-1"

        # Get points outside shape
        inner = points.within(shape)
        outer = ~inner

        inner_to_centroid = (
            points[in_compartment & inner].distance(shape.centroid).mean()
        )
        outer_to_centroid = (
            points[in_compartment & outer].distance(shape.centroid).mean()
        )

        # Values [0, 1], where 1 is asymmetrical and 0 is symmetrical.
        cell_radius = points["cell_radius"].values[0]
        inner_asymmetry = inner_to_centroid / cell_radius
        outer_asymmetry = outer_to_centroid / cell_radius

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
        tl.is_nuclear(data, self.shape_name)

        # Join attributes to points
        shape_prefix = self.shape_name.split("_shape")[0]
        points = (
            points.set_index("cell")
            .join(data.obs[["cell_radius", f"{shape_prefix}_in_nucleus"]])
            .reset_index()
        )

        return data, points


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


class CellOpenEnrichment(AbstractFeature):
    def __init__(self):
        # Not used but keep for consistency
        shape_name = "cell_shape"
        super().__init__(shape_name)
        self.metadata = {
            "cell_open_0.05_enrichment": float,
            "cell_open_0.1_enrichment": float,
        }

    def extract(self, points):
        """Given a set of points, calculate the enrichment ratio of transcripts in morphological openings of the cell.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates.

        Return
        ------
        cell_open_0.05_enrichment : float
            Value ranges between [0, 1]. 0 denotes no points in opening with distance = 5% of cell radius, 1 denotes all points in opening.
        cell_open_0.10_enrichment : float
            Value ranges between [0, 1]. 0 denotes no points in opening with distance = 10% of cell radius, 1 denotes all points in opening.
        """
        points = super().extract(points)

        # Get shape polygon
        shape05 = points["cell_open_0.05_shape"].values[0]
        shape10 = points["cell_open_0.1_shape"].values[0]

        # Get points outside shape
        n_in_05 = sum(points.within(shape05))
        n_in_10 = sum(points.within(shape10))

        enrichment_05 = n_in_05 / len(points)
        enrichment_10 = n_in_10 / len(points)

        return pd.DataFrame(
            {
                f"cell_open_0.05_enrichment": enrichment_05,
                f"cell_open_0.1_enrichment": enrichment_10,
            },
            index=pd.Index([0], name="tmp_index"),
        )

    def precompute(self, data, points):
        data, points = super().precompute(data, points)

        # Calculate morphological openings
        tl.cell_morph_open(data, 0.05)
        tl.cell_morph_open(data, 0.1)

        # Join attributes to points
        points = (
            points.set_index("cell")
            .join(data.obs[["cell_open_0.05_shape", "cell_open_0.1_shape"]])
            .reset_index()
        )

        return data, points
