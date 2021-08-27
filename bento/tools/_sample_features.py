from abc import ABCMeta, abstractmethod

import dask_geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm

from .. import tools as tl
from ..utils import track


@track
def proximity(data, shape_name, position="inner", n_jobs=1, copy=False):
    """
    Compute proximity of points to shape.

    Parameters
    ----------
    data : spatial formatted AnnData
        data.uns['points'] must be DataFrame with minimum columns x, y, cell, and gene.
    shape_name : str
        Column name in data.obs referring to shape to use in calculation.
    position : str
        Expect one of 'inner', or 'outer', by default 'inner'. 'Outer' refers to points outside the shape but within the same overall cellular compartment; 'inner' refers to points inside shape.
    n_jobs : int
        Number of jobs to run in parallel.
    copy : bool
        Whether to return a copy of data or to modify in place, by default False.

    """
    adata = data.copy() if copy else data
    if shape_name not in data.obs.columns:
        raise ValueError("Not a valid shape")

    layer = f'{shape_name.split(sep="_shape")[0]}_{position}_proximity'

    if position == "inner":
        # TODO switch to indexed shapes
        InnerToShapeDist().transform(data, shape_name, layer, n_jobs)
    elif position == "outer":
        if shape_name == "cell_shape":
            raise ValueError("Extracellular points not supported")
        OuterToShapeDist().transform(data, shape_name, layer, n_jobs)
    else:
        raise ValueError("Not a valid position.")

    return adata if copy else None


@track
def asymmetry(data, shape_name, position="inner", n_jobs=1, copy=False):
    """
    Compute asymmetry of points relative to shape.

    Parameters
    ----------
    data : spatial formatted AnnData
        data.uns['points'] must be DataFrame with minimum columns x, y, cell, and gene.
    shape_name : str
        Column name in data.obs referring to shape to use in calculation.
    position : str
        Expect one of 'inner', or 'outer', by default 'inner'. 'Outer' refers to points outside the shape but within the same overall cellular compartment; 'inner' refers to points inside shape.
    n_jobs : int
        Number of jobs to run in parallel.
    copy : bool
        Whether to return a copy of data or to modify in place, by default False.

    """
    adata = data.copy() if copy else data
    if shape_name not in data.obs.columns:
        raise ValueError("Not a valid shape")

    layer = f'{shape_name.split(sep="_shape")[0]}_{position}_asymmetry'

    if position == "inner":
        # TODO switch to indexed shapes
        InnerToShapeAsymmetry().transform(data, shape_name, layer, n_jobs)
    elif position == "outer":
        if shape_name == "cell_shape":
            raise ValueError("Extracellular points not supported")
        OuterToShapeAsymmetry().transform(data, shape_name, layer, n_jobs)
    else:
        raise ValueError("Not a valid position.")

    return adata if copy else None


class AbstractFeature(metaclass=ABCMeta):

    __point_metadata = pd.Series(["x", "y", "cell", "gene", "nucleus"])

    @abstractmethod
    def extract(self, points, shape_name):
        """Given a set of points, extract and return a single feature value.

        Parameters
        ----------
        points : DataFrame
            Point coordinates.
        shape : Polygon (Shapely)

        """
        shape = points[shape_name].values[0]
        return points, shape, shape_name

    @abstractmethod
    def precompute(self, data, points, shape_name):
        """Any cell-level feature to precompute. This prevents recomputing once per sample."""
        # GeoDataFrame for spatial operations
        points = gpd.GeoDataFrame(
            points, geometry=gpd.points_from_xy(points.x, points.y)
        )

        # Save shape to points column
        points = (
            points.set_index("cell")
            .join(data.obs[shape_name])
            .reset_index()
            .sort_values(["cell", "gene"])
        )

        return data, points, shape_name

    @classmethod
    def transform(self, data, shape_name, layer, n_jobs):
        """Applies self.extract() to all points grouped by cell and gene.

        Parameters
        ----------
        data : spatial formatted AnnData
            data.uns['points'] must be DataFrame with minimum columns x, y, cell, and gene.
        shape_name : str
            Column name in data.obs referring to shape to use in calculation.
        """

        points = data.uns["points"].copy()

        # Check points DataFrame for missing columns
        if not self.__point_metadata.isin(points.columns).all():
            raise KeyError(
                f"'points' DataFrame needs to have all columns: {self.__point_metadata.tolist()}."
            )

        data, points, shape_name = self.precompute(self, data, points, shape_name)

        # Cast categorical type to save memory
        cat_vars = ["cell", "gene", "nucleus"]
        points[cat_vars] = points[cat_vars].astype("category")

        out = (
            dask_geopandas.from_geopandas(points, chunksize=10000)
            .groupby(["cell", "gene"])
            .apply(lambda df: self.extract(self, df, shape_name), meta=(layer, "float"))
        )

        with ProgressBar():
            feature_df = out.compute().reset_index()

        # Save results to data layer
        data.layers[layer] = (
            feature_df.pivot(index="cell", columns="gene", values=layer)
            .reindex(index=data.obs_names, columns=data.var_names)
            .astype(float)
        )

        return


class OuterToShapeDist(AbstractFeature):
    def extract(self, points, shape_name):
        """Given a set of points, calculate and return the average proximity between points outside to the shape boundary.

        Only considers points inside the same major subcellular compartment (cytoplasm or nucleus).

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.

        Return
        ------
        proximity : float
            Value ranges between [0, 1]. Value closer to 0 denotes farther from shape boundary, value closer to 1 denotes close to shape boundary.
        """
        points, shape, shape_name = super().extract(self, points, shape_name)

        # Get points in same major subcellular compartment
        shape_prefix = shape_name.split("_shape")[0]
        if points[f"{shape_prefix}_in_nucleus"].values[0]:
            in_compartment = points["nucleus"] != "-1"
        else:
            in_compartment = points["nucleus"] == "-1"

        # Get points outside shape
        outer = ~points.within(shape)

        dist = points[in_compartment & outer].distance(shape.boundary).mean()

        # Scale from [0, 1], where 1 is close and 0 is far.
        cell_radius = points["cell_radius"].values[0]
        proximity = (cell_radius - dist) / cell_radius
        return proximity

    def precompute(self, data, points, shape_name):
        data, points, shape_name = super().precompute(self, data, points, shape_name)

        tl.cell_radius(data)
        tl.is_nuclear(data, shape_name)

        # Join attributes to points
        shape_prefix = shape_name.split("_shape")[0]
        points = (
            points.set_index("cell")
            .join(data.obs[["cell_radius", f"{shape_prefix}_in_nucleus"]])
            .reset_index()
        )

        return data, points, shape_name


class InnerToShapeDist(AbstractFeature):
    def extract(self, points, shape_name):
        """Given a set of points, calculate and return the average proximity between points inside to the shape boundary.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.

        Return
        ------
        proximity : float
            Value ranges between [0, 1]. Value closer to 0 denotes farther from shape boundary, value closer to 1 denotes close to shape boundary.
        """
        points, shape, shape_name = super().extract(self, points, shape_name)
        points = gpd.GeoDataFrame(points)
        inner = points.within(shape)

        dist = points[inner].distance(shape.boundary).mean()

        # Scale from [0, 1], where 1 is close and 0 is far.
        cell_radius = points["cell_radius"].values[0]
        proximity = (cell_radius - dist) / cell_radius
        return proximity

    def precompute(self, data, points, shape_name):
        data, points, shape_name = super().precompute(self, data, points, shape_name)

        tl.cell_radius(data)
        tl.is_nuclear(data, shape_name)

        # Join attributes to points
        shape_prefix = shape_name.split("_shape")[0]
        points = (
            points.set_index("cell")
            .join(data.obs[["cell_radius", f"{shape_prefix}_in_nucleus"]])
            .reset_index()
        )

        return data, points, shape_name


class OuterToShapeAsymmetry(AbstractFeature):
    def extract(self, points, shape_name):
        """Given a set of points, calculate and return the measure of symmetry relative to shape.

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
        points, shape, shape_name = super().extract(self, points, shape_name)

        # Get points in same major subcellular compartment
        shape_prefix = shape_name.split("_shape")[0]
        if points[f"{shape_prefix}_in_nucleus"].values[0]:
            in_compartment = points["nucleus"] != "-1"
        else:
            in_compartment = points["nucleus"] == "-1"

        # Get points outside shape
        outer = ~points.within(shape)

        dist_to_centroid = (
            points[in_compartment & outer].distance(shape.centroid).mean()
        )

        # Values [0, 1], where 1 is asymmetrical and 0 is symmetrical.
        cell_radius = points["cell_radius"].values[0]
        asymmetry = dist_to_centroid / cell_radius
        return asymmetry

    def precompute(self, data, points, shape_name):
        data, points, shape_name = super().precompute(self, data, points, shape_name)

        tl.cell_radius(data)
        tl.is_nuclear(data, shape_name)

        # Join attributes to points
        shape_prefix = shape_name.split("_shape")[0]
        points = (
            points.set_index("cell")
            .join(data.obs[["cell_radius", f"{shape_prefix}_in_nucleus"]])
            .reset_index()
        )

        return data, points, shape_name


class InnerToShapeAsymmetry(AbstractFeature):
    def extract(self, points, shape_name):
        """Given a set of points, calculate and return the measure of symmetry relative to shape.

        Parameters
        ----------
        points : GeoDataFrame
            Point coordinates. Assumes "nuclear" column is present.

        Return
        ------
        asymmetry : float
            Value ranges between [0, inf]. 0 denotes symmetry, values between 0 and 1 denotes strong asymmetry, >1 denotes very strong asymmetry.
        """
        points, shape, shape_name = super().extract(self, points, shape_name)
        inner = points.within(shape)

        dist_to_centroid = points[inner].distance(shape.centroid).mean()

        # Values [0, 1], where 1 is asymmetrical and 0 is symmetrical.
        cell_radius = points["cell_radius"].values[0]
        asymmetry = dist_to_centroid / cell_radius
        return asymmetry

    def precompute(self, data, points, shape_name):
        data, points, shape_name = super().precompute(self, data, points, shape_name)

        tl.cell_radius(data)
        tl.is_nuclear(data, shape_name)

        # Join attributes to points
        shape_prefix = shape_name.split("_shape")[0]
        points = (
            points.set_index("cell")
            .join(data.obs[["cell_radius", f"{shape_prefix}_in_nucleus"]])
            .reset_index()
        )

        return data, points, shape_name
